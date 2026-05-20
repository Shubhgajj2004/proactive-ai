# ADR-001: Proactive AI — System Architecture

**Status:** Proposed  
**Date:** 2026-05-13  
**Deciders:** Engineering team  

---

## Context

Wearable client device (e.g., smart glasses). Thin client: audio capture → Daily WebRTC → server; audio playback from server. All AI computation is server-side.

Before first use, the wearer completes a **one-time enrollment flow**: speaks a 10-second phrase. Server extracts a **d-vector** (192-dim speaker embedding) via SpeechBrain ECAPA-TDNN and stores it in PostgreSQL per `user_id`. This is the authoritative wearer identity across all sessions and reconnects.

At runtime, the server:
1. Detects speech boundaries via VAD (60s hard cap)
2. Transcribes audio with timestamped diarization via `gemini-3-flash-preview`
3. Segments audio by speaker using timestamps; extracts per-speaker d-vectors; matches against enrolled voiceprint
4. Routes: wake word → reactive path; ambient → proactive evaluation path
5. Proactively **suggests** when confidence > 75%; requires user consent before any tool execution
6. Reactive ("hey jarvis"): intent → plan → execute (no consent gate)

---

## ⚠ Pre-Implementation Verification Required

**Before building the audio segmentation pipeline, verify:** Does `gemini-3-flash-preview` return per-speaker timestamps (`start_ms`, `end_ms`) alongside speaker labels in its diarization output? The voiceprint reconciliation pipeline depends on slicing the raw audio by timestamp to extract per-speaker d-vectors. If Gemini returns only `{speaker_label, text}` without timestamps, the segmentation approach must be redesigned (e.g., run a separate diarization model before STT).

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                          CLIENT DEVICE                          │
│   Mic → Audio Capture → Daily WebRTC Client ← TTS Audio Out     │
└─────────────────────────────┬────────────────────────────────────┘
                              │ Daily WebRTC (bidirectional audio)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│         PIPECAT PIPELINE (per session, Daily WebRTC transport)  │
│                                                                  │
│   Audio Frames                                                   │
│       ↓                                                          │
│   [Audio Gain Amplifier]  ×AUDIO_GAIN_FACTOR (default 5×)       │
│       custom Pipecat processor; NumPy int16 clip to ±32767       │
│       ↓                                                          │
│   [Pipecat SileroVADAnalyzer]  (built-in; emits VAD events)     │
│   [vad_processor.py FSM]  wraps VAD events with custom logic:    │
│       speech → START                                             │
│       silence > 3000ms → STOP (emit)                            │
│       duration >= 60000ms → FORCE_STOP (emit + reset)           │
│       ↓                                                          │
│   [Audio Accumulator]                                            │
│       < 1.5s accumulated speech → DROP (no STT call)            │
│       else → emit utterance blob                                │
│       ↓                                                          │
│   [Gemini STT]  gemini-3-flash-preview                           │
│       → [{start_ms, end_ms, speaker_label, text}] per segment   │
│       ↓                                                          │
│   [Audio Segmenter]  slice raw audio by timestamps              │
│       overlap rule: if segments overlap, assign overlap region  │
│       to segment with higher energy (torchaudio RMS comparison) │
│       ↓                                                          │
│   [Audio Resampler]  torchaudio Resample 48kHz→16kHz per seg   │
│       ↓                                                          │
│   [ECAPA-TDNN Embedder]  192-dim d-vector per segment           │
│       via run_in_executor (CPU-bound; must not block asyncio)   │
│       ↓                                                          │
│   [Voiceprint Matcher]  cosine sim vs user_voiceprints.d_vector │
│       > 0.75 → is_wearer=True                                    │
│       < 0.50 (or single speaker) → is_wearer=Unknown (skip mem0)│
│       else → is_wearer=False                                     │
│       ↓                                                          │
│   [Barge-In Handler]  UserStartedSpeakingFrame → stop TTS       │
│       ↓                                                          │
│   [SESSION ROUTER]                                               │
│       ├── "hey jarvis" found → REACTIVE → ACK chime + spawn     │
│       ├── ACTIVE_SESSION → route to action agent queue           │
│       └── AMBIENT → cost governor check → Ambient Processor     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Transport | **Daily WebRTC** via Pipecat | Low-latency audio; native Pipecat integration |
| Audio pipeline | **Pipecat** `pipecat-ai[daily,silero]` | VAD inference, TTS streaming, barge-in, Daily transport |
| Audio gain | **`audio_gain.py`** (NumPy, first stage) | ×5 PCM amplifier; hardware mic is low-volume; clips to int16 range |
| VAD | **Pipecat `SileroVADAnalyzer`** (built-in) | Already integrated + tested; emits VAD START/STOP events; custom FSM layered on top in Python |
| STT | **`gemini-3-flash-preview`** (sole provider) | Multilingual + timestamped diarization; predictable single provider |
| Speaker embedding | **SpeechBrain ECAPA-TDNN** | `speechbrain` + `speechbrain/ecapa_tdnn` weights; 192-dim d-vector; ~60ms CPU inference |
| Text embeddings | **`text-embedding-004`** via `google-generativeai` | 768-dim; single SDK; consistent provider |
| Ambient LLM | **Gemini 2.5 Flash** via **OpenRouter** | Combined structured output; swappable to any provider via `LLMClient` abstraction |
| Action LLM (standard) | **Gemini 2.5 Flash** via **OpenRouter** | Proactive sessions; swappable |
| Action LLM (premium) | **Claude Sonnet 4.6** via **OpenRouter** | "hey jarvis" sessions; swappable |
| LLM transport | **OpenRouter** (`openai` SDK, custom `base_url`) | Single API key covers all LLM models; direct provider swap requires only config change |
| Action graphs | **LangGraph** + **`langgraph-checkpoint-postgres`** | Checkpoints in PostgreSQL; Redis is read cache only |
| Long-term memory | **`mem0ai`** with pgvector backend, 768-dim | Wearer facts only; explicit dim config |
| Tool protocol | **MCP** over **HTTP/SSE** | Networked; independently scalable; `mcp` PyPI package |
| Context store | **PostgreSQL JSONB** (weekly partitioned) | No MongoDB; single store |
| Session state | **PostgreSQL `sessions`** (source of truth) | LangGraph checkpoints via `langgraph-checkpoint-postgres` |
| Read cache | **Redis** (5-min TTL snapshots) | Fast session reads; NOT checkpoint store |
| TTS | **Gemini TTS** streaming via Pipecat | <500ms first chunk |
| Reactive ACK | **Pre-recorded PCM** (`ack_chime.wav`) injected as `AudioRawFrame` | <50ms; no LLM cost |
| Primary DB | **PostgreSQL 16 + pgvector** | Single store for all data |

---

## Enrollment Flow

```
Setup:
  User speaks 10-second phrase
  → Server receives audio blob
  → ECAPA-TDNN extracts 192-dim d-vector
  → INSERT INTO user_voiceprints (user_id, d_vector, enrolled_at)

Later reconnects use same row — voiceprint never expires unless user re-enrolls.
```

---

## VAD State Machine

```
┌──────────┐  speech > 200ms  ┌──────────────┐
│   IDLE   │ ───────────────► │ ACCUMULATING │
│isStarted │                  │ isStarted    │
│ = False  │ ◄─────────────── │   = True     │
└──────────┘  silence > 3000ms│              │
              OR               │              │
              duration>=60000ms│              │
              → emit utterance │              │
              → isStarted=False└──────────────┘

Accumulator rules (before STT call):
  < 1500ms total speech → DROP silently (no STT API call)
  ≥ 1500ms and < 60000ms → emit to STT
  ≥ 60000ms → force-emit; reset accumulator
```

---

## Cost Governor

```python
class CostGovernor:
    def __init__(self):
        self.vad_timestamps = deque(maxlen=200)  # rolling 60s window
        self.ambient_paused_until: float = 0

    def on_vad_start(self) -> None:
        self.vad_timestamps.append(time.time())

    @property
    def ambient_allowed(self) -> bool:
        if time.time() < self.ambient_paused_until:
            return False
        recent = sum(1 for t in self.vad_timestamps if t > time.time() - 60)
        if recent > 60:  # 60 fires/min threshold (adaptive in v2)
            self.ambient_paused_until = time.time() + 30
            return False
        return True
    # STT always runs; wake word check always runs regardless of ambient_allowed
```

Daily token budget tracked per `user_id` in `user_settings` table. On breach, `ambient_allowed` returns False; reactive wake word path still works.

---

## Session State Machine

```
┌──────────┐  confidence > 75%  ┌────────────────────────────┐
│          │ ─────────────────► │ ACTIVE (proactive)         │
│  AMBIENT │                    │ timeout: 25s→prompt, 35s→abort│
│          │  "hey jarvis"      │ turn limit: 5              │
│          │ ─────────────────► └────────────────────────────┘
│          │                    ┌────────────────────────────┐
│          │                    │ ACTIVE (reactive)          │
│          │◄─────────────────  │ timeout: 75s→prompt, 90s→abort│
│          │  DONE / timeout    │ turn limit: 8              │
└──────────┘                    └────────────────────────────┘
```

**PostgreSQL `sessions` (source of truth):**
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('AMBIENT', 'ACTIVE')),
    trigger_source TEXT CHECK (trigger_source IN ('proactive_confidence', 'wake_word')),
    model_tier TEXT CHECK (model_tier IN ('standard', 'premium')),
    langgraph_thread_id TEXT,
    pending_memory_ops JSONB DEFAULT '[]',
    initial_proposed_action TEXT,
    initial_consent_prompt TEXT,
    initial_reasoning TEXT,
    turn_count INT DEFAULT 0,
    last_activity_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now()
);
```

**Redis (read cache only, 5-min TTL):**
```python
redis.setex(f"proactive:session:{session_id}", 300, json.dumps(row))
# On miss → read PostgreSQL sessions table, warm cache
```

---

## Session Router Logic

```python
def route(transcript: str, state: dict, cost_governor: CostGovernor) -> str:
    if "hey jarvis" in transcript.lower():
        return "REACTIVE"                # always, even during backoff
    if state["state"] == "ACTIVE":
        return "ACTIVE"                  # route to action agent queue
    if cost_governor.ambient_allowed:
        return "AMBIENT"                 # run ambient processor
    return "SKIP"                        # backoff active, no ambient LLM
```

---

## Ambient Processor

```python
class MemoryOp(BaseModel):
    op: Literal["add", "update", "delete"]
    fact: str
    memory_id: str | None = None

class AmbientAnalysis(BaseModel):
    memory_operations: list[MemoryOp]   # wearer facts only (is_wearer=True turns)
    summary: str
    extracted_facts: list[str]
    tags: list[str]
    should_act: bool
    confidence: float
    proposed_action: str                # e.g. "offer to book a cab to the airport"
    consent_prompt: str                 # e.g. "Looks like you need a ride — want me to book one?"
    reasoning: str
```

Pre-call: `memory.search(wearer_text_only, user_id=user_id, limit=5)` + capability manifest in system prompt.

Post-call (asyncio, parallel):
1. `memory_operations` → queued; written ONLY if no session spawned
2. `summary + facts` → `context_summaries` + `raw_transcripts`
3. Full object → `ambient_logs` → capture returned `ambient_log_id`
4. `confidence > 0.75` → spawn ACTIVE_SESSION; hand off `proposed_action`, `consent_prompt`, `reasoning`, `pending_memory_ops`, `ambient_log_id`

---

## Action Agent — Two LangGraph Graphs

### Proactive Graph

```
[Suggest node]
  TTS: consent_prompt from handoff (no LLM call needed)
  → await user reply
  ├── "no" / "not now" / timeout → flush memory ops → DONE → AMBIENT
  └── "yes" / "sure" / "go ahead" → [Shared Plan node]
  (log outcome to ambient_logs in DONE handler)
```

### Reactive Graph

```
[Intent node]
  LLM: parse "hey jarvis + request" transcript
  if complete → [Shared Plan node]
  if ambiguous → one clarifying question → [Shared Plan node]
  (no consent gate — user explicitly invoked)
```

### Shared Nodes (Plan → Execute loop)

```
[Plan node]
  → {next_step: str, need_clarification: bool, question: str | None, done: bool}
  ├── need_clarification → TTS question → WAIT → [Plan node]
  ├── done → [Respond node] → TTS → flush memory ops → DONE → AMBIENT
  └── next_step defined →
        [Tool Select node]
          embed(next_step) via text-embedding-004
          pgvector top-2 schemas from mcp_tools
        [Execute node]
          write tool → 1 attempt only + X-Idempotency-Key header; on failure → error in history → [Plan node]
          read tool → httpx, 3 retries, 2s base backoff
          → results appended to history → [Plan node]
```

**Memory flush — all termination paths:**
```python
async def terminate_session(session_id: str, reason: str) -> None:
    try:
        # Load pending ops from sessions table (ambient handoff ops)
        session = await db.sessions.get(session_id)
        # Load session ops accumulated during active turns from LangGraph checkpoint
        checkpoint = await langgraph_checkpointer.aget({"configurable": {"thread_id": session.langgraph_thread_id}})
        session_ops = checkpoint["channel_values"].get("session_memory_ops", []) if checkpoint else []

        all_ops = session.pending_memory_ops + session_ops
        for op in all_ops:
            if op["op"] == "add":
                memory.add(op["fact"], user_id=session.user_id)
            elif op["op"] == "update":
                memory.update(op["memory_id"], op["fact"])
            elif op["op"] == "delete":
                memory.delete(op["memory_id"])
    finally:
        await cleanup_session_records(session_id)
```

---

## Tool Layer

**Capability manifest** (Tier 1): tool name + one-liner in Ambient Processor + Action Agent system prompt. Regenerated on tool change, cached in Redis.

**Per-step schema RAG** (Tier 2): called at each Execute step with fresh `embed(next_step)` → pgvector top-2 full schemas.

**MCP executor:**
- Read tools: 3 retries, exponential backoff
- Write tools: 1 attempt, `X-Idempotency-Key: {session_id}:{step_count}:{tool_name}`; failure → error context in Plan node history

---

## mem0 Configuration

```python
from mem0 import Memory

memory = Memory.from_config({
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "connection_string": DATABASE_URL,
            "embedding_model_dims": 768,          # text-embedding-004
            "table_name": "mem0_memories"          # avoid naming collision with custom tables
        }
    }
})
```

Only `is_wearer=True` turns are passed to `memory.add()`. Bystander speech is never ingested into personal memory.

---

## Provider Abstraction Layer

**Every AI capability — LLM, STT, TTS, and embeddings — has its own ABC + factory.** No module outside its own `server/<capability>/` directory ever imports from a vendor SDK directly. Swapping the underlying model or provider is a config change only.

```
server/
├── llm/         client.py (ABC)  openrouter.py  factory.py
├── stt/         client.py (ABC)  gemini.py      factory.py
├── tts/         client.py (ABC)  gemini.py      factory.py
└── embeddings/  client.py (ABC)  google.py      factory.py
```

---

### LLM

All LLM calls (ambient processor, action agent plan/intent/respond nodes) go through `LLMClient`. Default backend: OpenRouter.

### Interface

```python
# server/llm/client.py
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any

class LLMResponse(BaseModel):
    content: str
    usage_input_tokens: int
    usage_output_tokens: int

class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        response_format: type[BaseModel] | None = None,   # structured output
        temperature: float = 0.3,
    ) -> LLMResponse: ...
```

### OpenRouter Implementation (default)

```python
# server/llm/openrouter.py
from openai import AsyncOpenAI
from server.llm.client import LLMClient, LLMResponse

class OpenRouterClient(LLMClient):
    def __init__(self, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages, response_format=None, temperature=0.3) -> LLMResponse:
        kwargs = {"model": self._model, "messages": messages, "temperature": temperature}
        if response_format is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": response_format.model_json_schema()}
        resp = await self._client.chat.completions.create(**kwargs)
        return LLMResponse(
            content=resp.choices[0].message.content,
            usage_input_tokens=resp.usage.prompt_tokens,
            usage_output_tokens=resp.usage.completion_tokens,
        )
```

### Factory (reads from config)

```python
# server/llm/factory.py
from server.config import settings
from server.llm.openrouter import OpenRouterClient

def make_llm_client(tier: str) -> "LLMClient":
    """tier: 'ambient' | 'standard' | 'premium'"""
    model = {
        "ambient":   settings.LLM_AMBIENT_MODEL,
        "standard":  settings.LLM_ACTION_STANDARD_MODEL,
        "premium":   settings.LLM_ACTION_PREMIUM_MODEL,
    }[tier]
    return OpenRouterClient(
        model=model,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )
```

---

### STT

```python
# server/stt/client.py
class STTSegment(BaseModel):
    start_ms: int; end_ms: int; speaker_label: str; text: str; language: str

class STTClient(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> list[STTSegment]: ...
    @property
    @abstractmethod
    def usage_tokens(self) -> int: ...    # for token_counter after each call

# server/stt/gemini.py  — default implementation using google-generativeai SDK
class GeminiSTTClient(STTClient): ...

# server/stt/factory.py
def make_stt_client() -> STTClient:
    return GeminiSTTClient(model=settings.STT_MODEL, api_key=settings.GEMINI_API_KEY)
```

---

### TTS

```python
# server/tts/client.py
class TTSClient(ABC):
    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]: ...
    # Streams PCM chunks; Pipecat wraps each in AudioRawFrame

# server/tts/gemini.py  — default implementation using google-generativeai SDK
class GeminiTTSClient(TTSClient): ...

# server/tts/factory.py
def make_tts_client() -> TTSClient:
    return GeminiTTSClient(model=settings.TTS_MODEL, api_key=settings.GEMINI_API_KEY)
```

---

### Embeddings

```python
# server/embeddings/client.py
class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

# server/embeddings/google.py  — default: text-embedding-004 via google-generativeai
class GoogleEmbeddingClient(EmbeddingClient): ...

# server/embeddings/factory.py
def make_embedding_client() -> EmbeddingClient:
    return GoogleEmbeddingClient(model=settings.EMBEDDING_MODEL, api_key=settings.GEMINI_API_KEY)
```

---

### Unified Config

All provider and model settings live in one place — change `.env`, restart, done:

```python
# server/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ── LLM via OpenRouter ───────────────────────────────────────────────────
    OPENROUTER_API_KEY: str
    LLM_BASE_URL: str           = "https://openrouter.ai/api/v1"
    LLM_AMBIENT_MODEL: str      = "inclusionai/ring-2.6-1t:free"   # default
    LLM_ACTION_STANDARD_MODEL: str = "inclusionai/ring-2.6-1t:free"
    LLM_ACTION_PREMIUM_MODEL: str  = "inclusionai/ring-2.6-1t:free"
    # ── Example alternatives (swap in .env, no code change) ─────────────────
    # LLM_AMBIENT_MODEL      = "google/gemini-2.5-flash"
    # LLM_ACTION_PREMIUM_MODEL = "anthropic/claude-sonnet-4-6"
    # LLM_ACTION_PREMIUM_MODEL = "openai/gpt-4o"
    # To use direct Gemini API instead of OpenRouter:
    #   LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    #   OPENROUTER_API_KEY = <your-gemini-key>

    # ── STT (direct Gemini SDK) ──────────────────────────────────────────────
    GEMINI_API_KEY: str
    STT_MODEL: str              = "gemini-3-flash-preview"
    # Example alternative: STT_MODEL = "gemini-2.0-flash"

    # ── TTS (direct Gemini SDK) ──────────────────────────────────────────────
    TTS_MODEL: str              = "gemini-2.5-flash-preview-tts"
    # Example alternative: TTS_MODEL = "gemini-2.0-flash-tts"

    # ── Embeddings (direct Google SDK) ──────────────────────────────────────
    EMBEDDING_MODEL: str        = "text-embedding-004"   # 768-dim
    # Example alternative: EMBEDDING_MODEL = "text-embedding-005"

    # ── Infrastructure ───────────────────────────────────────────────────────
    DATABASE_URL: str
    REDIS_URL: str              = "redis://localhost:6379"
    DAILY_API_KEY: str

    # ── Audio ────────────────────────────────────────────────────────────────
    AUDIO_GAIN_FACTOR: float    = 5.0   # hardware mic volume boost (first pipeline stage)

    class Config:
        env_file = ".env"

settings = Settings()
```

**The rule:** every `factory.py` reads from `settings`. Business logic reads from factories. No vendor import leaks past its own `server/<capability>/` directory.

### Token counting

```python
# token_counter.py — two paths, same Redis counter
async def track_llm_tokens(user_id: str, response: LLMResponse) -> bool:
    """After every LLMClient.complete() call (OpenRouter)."""
    tokens = response.usage_input_tokens + response.usage_output_tokens
    return await _increment(user_id, tokens)

async def track_tokens(user_id: str, usage_metadata) -> bool:
    """After Gemini STT and TTS calls (google-generativeai usage_metadata)."""
    return await _increment(user_id, usage_metadata.total_token_count)

async def _increment(user_id: str, tokens: int) -> bool:
    key = f"tokens:{user_id}:{date.today().isoformat()}"
    new_total = await redis.incrby(key, tokens)
    await redis.expire(key, 86400 * 2)
    row = await db.fetchrow("SELECT daily_token_budget FROM user_settings WHERE user_id = $1", user_id)
    return new_total <= (row["daily_token_budget"] if row else 1_000_000)
```

---

## Daily Room Lifecycle

```python
DAILY_API_BASE = "https://api.daily.co/v1"
DAILY_API_KEY = config.DAILY_API_KEY

async def daily_create_room(session_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{DAILY_API_BASE}/rooms",
            headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
            json={"name": session_id, "properties": {"exp": int(time.time()) + 7200}}
        )
        r.raise_for_status()
        return r.json()

async def daily_create_token(room_name: str, user_id: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{DAILY_API_BASE}/meeting-tokens",
            headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
            json={"properties": {"room_name": room_name, "user_id": user_id}}
        )
        r.raise_for_status()
        return r.json()["token"]

async def daily_delete_room(room_name: str) -> None:
    async with httpx.AsyncClient() as client:
        await client.delete(
            f"{DAILY_API_BASE}/rooms/{room_name}",
            headers={"Authorization": f"Bearer {DAILY_API_KEY}"}
        )

@app.post("/api/session/create")
async def create_session(user_id: str):
    session_id = str(uuid.uuid4())
    room = await daily_create_room(session_id)
    token = await daily_create_token(room["name"], user_id)
    await db.execute("INSERT INTO sessions (session_id, user_id) VALUES ($1, $2)", session_id, user_id)
    asyncio.create_task(inactivity_timer(session_id, timeout_seconds=300))
    return {"room_url": room["url"], "token": token}

# Enrollment endpoint (separate from ambient pipeline)
@app.post("/api/enroll")
async def enroll_user(user_id: str, audio: UploadFile):
    audio_bytes = await audio.read()
    waveform = load_audio_bytes(audio_bytes, target_sr=16000)   # torchaudio resample
    d_vector = await asyncio.get_event_loop().run_in_executor(
        None, lambda: ecapa_model.encode_batch(waveform)
    )
    await db.execute(
        "INSERT INTO user_voiceprints (user_id, d_vector) VALUES ($1, $2) "
        "ON CONFLICT (user_id) DO UPDATE SET d_vector = $2, enrolled_at = now()",
        user_id, d_vector.tolist()
    )
    return {"status": "enrolled"}

async def inactivity_timer(session_id: str, timeout_seconds: int):
    while True:
        await asyncio.sleep(60)
        row = await db.fetchrow("SELECT last_activity_at FROM sessions WHERE session_id = $1", session_id)
        if not row or time.time() - row["last_activity_at"].timestamp() > timeout_seconds:
            await teardown_session(session_id)
            return

async def teardown_session(session_id: str):
    await terminate_session(session_id, reason="inactivity")
    await daily_delete_room(session_id)

# Pipecat on_client_disconnected hook also calls teardown_session
```

---

## Barge-In Handling

```python
# In session_pipeline.py
@pipeline.on(UserStartedSpeakingFrame)
async def on_user_speaking(frame):
    await pipeline.emit(BotInterruptionFrame())   # Pipecat built-in: halts TTS
    # new utterance accumulates normally
    # session_router adds user_interrupted=True to action agent context
```

*(Confirmed: `UserStartedSpeakingFrame` and `BotInterruptionFrame` are real Pipecat frame classes — verified against Pipecat reference docs. Pin version and grep source to confirm no renames in your specific release.)*

---

## PostgreSQL Schema

```sql
-- Wearer identity
CREATE TABLE user_voiceprints (
    user_id TEXT PRIMARY KEY,
    d_vector vector(192) NOT NULL,    -- ECAPA-TDNN output
    enrolled_at TIMESTAMPTZ DEFAULT now()
);

-- Tool registry (768-dim)
CREATE TABLE mcp_tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,        -- context-specific; differentiates similar tools
    schema JSONB NOT NULL,
    schema_version INT DEFAULT 1,
    call_type TEXT NOT NULL CHECK (call_type IN ('read', 'write')),
    domain TEXT,
    embedding vector(768),            -- text-embedding-004
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON mcp_tools USING hnsw (embedding vector_cosine_ops);

-- Context summaries (WEEKLY partitioned)
CREATE TABLE context_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    summary TEXT,
    extracted_facts JSONB,
    tags JSONB,
    speaker_labels JSONB
) PARTITION BY RANGE (created_at);
CREATE INDEX ON context_summaries (user_id, created_at DESC);
-- Weekly partitions: context_summaries_2026w20, etc.

-- Raw transcripts (separate table; no FK to partitioned context_summaries)
CREATE TABLE raw_transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    raw_transcript TEXT
);
CREATE INDEX ON raw_transcripts (created_at);
CREATE INDEX ON raw_transcripts (user_id, created_at);
-- Cleaned by asyncio background job (DELETE WHERE created_at < now() - 24h)

-- Session state (source of truth)
-- Note: langgraph_thread_id is set equal to session_id at spawn time.
-- Verify against langgraph-checkpoint-postgres checkpoint table schema before finalizing.
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('AMBIENT', 'ACTIVE')),
    trigger_source TEXT CHECK (trigger_source IN ('proactive_confidence', 'wake_word')),
    model_tier TEXT CHECK (model_tier IN ('standard', 'premium')),
    langgraph_thread_id TEXT,              -- set to session_id at spawn; used by langgraph-checkpoint-postgres
    pending_memory_ops JSONB DEFAULT '[]', -- from ambient processor handoff
    triggering_ambient_log_id UUID,        -- links back to ambient_logs row for outcome update
    initial_proposed_action TEXT,
    initial_consent_prompt TEXT,
    initial_reasoning TEXT,
    turn_count INT DEFAULT 0,
    last_activity_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Confidence calibration log (90-day retention)
CREATE TABLE ambient_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    confidence FLOAT,
    should_act BOOL,
    proposed_action TEXT,
    session_spawned BOOL,
    user_outcome TEXT CHECK (user_outcome IN ('confirmed', 'declined', 'no_feedback', 'timeout'))
);

-- Cost governance
CREATE TABLE user_settings (
    user_id TEXT PRIMARY KEY,
    daily_token_budget INT DEFAULT 1000000,
    ambient_enabled BOOL DEFAULT true
);

-- Monthly rollup (permanent wearer insights)
CREATE TABLE user_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    month DATE NOT NULL,              -- first of month
    insight TEXT NOT NULL,
    source_count INT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (user_id, month, insight)  -- prevents duplicate rollup runs
);
```

---

## Token Counting

All Gemini SDK calls return `usage_metadata`. Intercept at the call site and increment a Redis counter per user per UTC day:

```python
# token_counter.py
async def track_tokens(user_id: str, usage_metadata) -> bool:
    """Returns True if within budget, False if exceeded."""
    tokens = usage_metadata.total_token_count
    key = f"tokens:{user_id}:{date.today().isoformat()}"
    new_total = await redis.incrby(key, tokens)
    await redis.expire(key, 86400 * 2)   # 2-day TTL (covers midnight boundary)

    row = await db.fetchrow("SELECT daily_token_budget FROM user_settings WHERE user_id = $1", user_id)
    budget = row["daily_token_budget"] if row else 1_000_000
    return new_total <= budget
```

**Two token tracking paths — both feed the same Redis counter per user per day:**
- `track_tokens(user_id, usage_metadata)` — called after Gemini STT and TTS (reads `usage_metadata.total_token_count`)
- `track_llm_tokens(user_id, llm_response)` — called after every `LLMClient.complete()` (reads `LLMResponse.usage_input_tokens + usage_output_tokens` from OpenRouter)

If budget exceeded, `cost_governor.ambient_allowed` returns False; reactive path still works.

**Gemini audio pricing note:** Audio input to Gemini is tokenized at ~32 tokens/second. A 10s utterance = ~320 audio tokens + text prompt + output tokens. At current audio input pricing ($0.15/M tokens after 40% discount), a 10s utterance costs ~$0.000048. At 60 utterances/hour that's ~$0.003/hour per user — well within budget for normal use. Set `daily_token_budget` conservatively and monitor actual spend in the first week.

---

## Weekly Partition Automation

pg_partman is not in standard apt repositories — it requires compilation from source. Use the **startup script approach** instead (no custom Dockerfile needed):

```python
# scripts/create_partitions.py — run at server startup and weekly via cleanup_job
async def ensure_weekly_partitions(db, weeks_ahead: int = 4):
    for i in range(weeks_ahead):
        week_start = (date.today() + timedelta(weeks=i)).strftime("...monday...")
        week_end   = (week_start + timedelta(weeks=1))
        await db.execute(f"""
            CREATE TABLE IF NOT EXISTS context_summaries_{week_start.strftime('%Yw%W')}
            PARTITION OF context_summaries
            FOR VALUES FROM ('{week_start}') TO ('{week_end}')
        """)
```

Called at server startup and every 6h in `cleanup_job`. Simple, no extension needed.

*(If your team later decides to adopt pg_partman, build a custom Dockerfile from `postgres:16` with pgvector + pg_partman compiled from source — but this is not required for v1.)*

---

## Pipecat Version Pin

Pin to a specific release before starting implementation. Verify barge-in frame names:

```toml
"pipecat-ai[daily,silero]==0.0.74"  # replace with latest stable at start date
```

```bash
# Verify frame class names in pinned version:
python -c "from pipecat.frames.frames import UserStartedSpeakingFrame, BotInterruptionFrame; print('OK')"
```

If `BotInterruptionFrame` is not found, search Pipecat source for the correct name before implementing `barge_in_handler.py`.

---

## Retention + Cleanup

No pg_cron dependency. Asyncio background task in `server/cleanup.py`, runs every 6 hours:

```python
async def cleanup_job():
    await db.execute("DELETE FROM raw_transcripts WHERE created_at < now() - interval '24 hours'")
    await db.execute("DELETE FROM ambient_logs WHERE created_at < now() - interval '90 days'")
    # Drop weekly partitions older than 30 days via pg_partman or manual DROP
    await db.execute("SELECT partman.run_maintenance('public.context_summaries')")
    # Run monthly rollup aggregate → user_insights (INSERT ... ON CONFLICT DO NOTHING)
```

---

## Latency Targets (Realistic, Tiered)

| Path | Target (p50) |
|---|---|
| Ambient processing, no action | < 1500ms from utterance end |
| Proactive suggestion TTS | < 4000ms from utterance end |
| Reactive response TTS | < 5000ms from utterance end |

*Session timeout (25-35s proactive, 75-90s reactive) is silence-to-abort, not response latency.*

---

## File Structure

```
proactive-ai/
├── server/
│   ├── main.py                           # FastAPI + Daily room lifecycle + inactivity timer
│   ├── config.py                         # Pydantic settings
│   ├── cleanup.py                        # asyncio background retention job
│   ├── assets/
│   │   └── ack_chime.wav                 # 50ms pre-recorded reactive ACK
│   ├── pipeline/
│   │   ├── session_pipeline.py           # Pipecat assembly (per Daily room)
│   │   ├── audio_gain.py                 # FIRST STAGE: ×5 PCM amplifier (clips to int16 range)
│   │   ├── vad_processor.py              # Silero VAD + FSM + 60s FORCE_STOP
│   │   ├── audio_accumulator.py          # 1.5s min gate + 60s cap
│   │   ├── stt_processor.py              # calls stt/factory.py → STTClient.transcribe()
│   │   ├── audio_segmenter.py            # slice by timestamps + overlap energy resolution
│   │   ├── audio_resampler.py            # torchaudio Resample 48kHz→16kHz
│   │   ├── speaker_embedder.py           # ECAPA-TDNN via run_in_executor (SpeechBrain)
│   │   ├── voiceprint_matcher.py         # cosine sim vs user_voiceprints
│   │   ├── barge_in_handler.py           # UserStartedSpeakingFrame → BotInterruptionFrame
│   │   ├── session_router.py             # wake word → REACTIVE; ACTIVE/AMBIENT routing
│   │   └── tts_processor.py              # calls tts/factory.py → TTSClient.synthesize_stream()
│   ├── prompts/
│   │   ├── stt.py                        # TRANSCRIBE prompt (language auto-detected, never hardcoded)
│   │   ├── ambient.py                    # SYSTEM + USER_TEMPLATE + RETRY_SCHEMA_REMINDER
│   │   ├── action.py                     # INTENT_*, PLAN_*, RESPOND_*, SUGGEST_FALLBACK
│   │   └── tools.py                      # CAPABILITY_MANIFEST_HEADER + NO_TOOLS_AVAILABLE
│   ├── llm/
│   │   ├── client.py                     # LLMClient ABC + LLMResponse model
│   │   ├── openrouter.py                 # OpenRouterClient (default backend, openai SDK)
│   │   └── factory.py                    # make_llm_client(tier) → LLMClient
│   ├── stt/
│   │   ├── client.py                     # STTClient ABC + STTSegment model
│   │   ├── gemini.py                     # GeminiSTTClient (default, google-generativeai)
│   │   └── factory.py                    # make_stt_client() → STTClient
│   ├── tts/
│   │   ├── client.py                     # TTSClient ABC
│   │   ├── gemini.py                     # GeminiTTSClient (default, google-generativeai)
│   │   └── factory.py                    # make_tts_client() → TTSClient
│   ├── embeddings/
│   │   ├── client.py                     # EmbeddingClient ABC
│   │   ├── google.py                     # GoogleEmbeddingClient (text-embedding-004)
│   │   └── factory.py                    # make_embedding_client() → EmbeddingClient
│   ├── ambient/
│   │   ├── processor.py                  # Combined LLM call → AmbientAnalysis
│   │   ├── memory_writer.py              # mem0 write (wearer facts only)
│   │   ├── context_writer.py             # context_summaries + raw_transcripts insert
│   │   └── cost_governor.py              # 60/min VAD threshold + daily budget
│   ├── action/
│   │   ├── proactive_graph.py            # LangGraph: Suggest → Shared nodes
│   │   ├── reactive_graph.py             # LangGraph: Intent → Shared nodes
│   │   ├── shared_nodes.py               # plan, tool_select, execute, respond
│   │   ├── state.py                      # ActionSessionState TypedDict
│   │   └── session_manager.py            # spawn/resume/terminate (all paths flush memory)
│   ├── tools/
│   │   ├── registry.py                   # text-embedding-004 embed (768-dim) + upsert
│   │   ├── selector.py                   # per-step top-2 pgvector RAG
│   │   ├── manifest.py                   # capability manifest + Redis cache
│   │   └── executor.py                   # MCP HTTP/SSE; read=3 retries; write=1+idempotency
│   ├── token_counter.py                  # track_tokens (Gemini STT/TTS) + track_llm_tokens (OpenRouter)
│   └── db/
│       ├── postgres.py                   # asyncpg pool
│       └── redis.py                      # cache client
├── client/
│   ├── client.py                         # Reference Daily WebRTC client
│   └── audio_capture.py
├── scripts/
│   ├── register_tools.py                 # Embed + upsert MCP tools
│   └── enroll_user.py                    # CLI: audio file → ECAPA-TDNN → user_voiceprints
├── docker-compose.yml                    # PostgreSQL 16 + pgvector ext, Redis
├── pyproject.toml
└── .env.example
```

---

## pyproject.toml

```toml
[project]
dependencies = [
    "pipecat-ai[daily,silero]",     # silero: Pipecat's built-in SileroVADAnalyzer
    "fastapi",
    "uvicorn",
    "langgraph",
    "langgraph-checkpoint-postgres",
    "mem0ai",
    "asyncpg",
    "redis",
    "google-generativeai",       # ONLY in: stt/gemini.py, tts/gemini.py, embeddings/google.py
    "openai",                    # ONLY in: llm/openrouter.py (OpenRouter OpenAI-compat API)
    "numpy",                     # audio_gain.py — PCM int16 amplification
    "mcp",
    "speechbrain",               # ECAPA-TDNN
    "torch",
    "torchaudio",
    "httpx",                     # MCP HTTP transport + Daily REST API calls
    "python-multipart",          # FastAPI file upload for /api/enroll
    "pydantic",
    "pydantic-settings",
]
```

> **Import rule:**
> - `google-generativeai` → only in `server/stt/gemini.py`, `server/tts/gemini.py`, `server/embeddings/google.py`
> - `openai` → only in `server/llm/openrouter.py`
> - All callers use the relevant `factory.py`. To add a new provider: add one file + update factory.

---

## Verification Plan

| Test | Pass Criteria |
|---|---|
| Audio gain (×5) | Silent WAV → 5× louder; 0dBFS input → clipped to int16 max; no distortion on normal speech |
| Provider swap (LLM) | Change `LLM_BASE_URL` + `LLM_AMBIENT_MODEL` → ambient processor uses new model; no code diff |
| Provider swap (STT) | Change `STT_MODEL` → `stt_processor.py` uses new model; no code diff |
| Enrollment | `user_voiceprints` row inserted; d_vector is 192-dim; CPU inference < 100ms |
| Voiceprint match | Same wearer, new session → cosine_sim > 0.85 |
| Voiceprint reject | Bystander segment → cosine_sim < 0.60 |
| Unknown chunk | Single speaker, low similarity → `is_wearer=Unknown`; NOT written to mem0 |
| Gemini STT timestamps | Feed known 2-speaker audio → segments have non-overlapping `start_ms`/`end_ms` |
| Audio segmenter | 10s blob + 2 segments → sliced WAVs match expected durations ±50ms |
| VAD 60s cap | 65s continuous speech → force-emit at 60s; remainder starts new utterance |
| VAD 1.5s gate | 1.2s utterance → dropped silently; no STT call made |
| Cost governor threshold | 65 VAD fires in 60s → 61st fire sets backoff; STT still runs on all |
| Wake word during backoff | "hey jarvis" during ambient pause → REACTIVE spawned; ambient skipped |
| Reactive ACK | "hey jarvis" → `ack_chime.wav` plays via `AudioRawFrame` within 100ms of routing |
| Proactive suggest + decline | Suggest TTS → "no" → DONE; zero MCP calls; outcome logged in ambient_logs |
| Proactive suggest + confirm | "yes" → Plan → tool execute → TTS result → DONE |
| Reactive path | "hey jarvis book cab" → Intent node → Plan (no consent) → execute |
| Barge-in | TTS playing → user speaks → `BotInterruptionFrame` → TTS stops |
| Write idempotency | `uber_book_ride` network error → NO retry; error in Plan node history |
| Read retry | `weather_fetch` fails twice → 3rd attempt succeeds |
| Memory flush (timeout) | 35s proactive timeout → `terminate_session` finally block flushes ops to mem0 |
| Memory flush (turn limit) | 5 proactive turns → same flush behavior |
| Redis miss | Delete Redis session key → next turn reads PostgreSQL; resumes correctly |
| Weekly partition | Rows written to current week's partition; partition pruning on query |
| Raw transcript cleanup | Background job → rows > 24h deleted; context_summaries unaffected |
| Daily teardown | 5min no audio → `daily_rest_api.delete_room()` called; pipeline torn down |
| Latency (ambient no action) | < 1500ms utterance end → ambient LLM complete |
| Latency (proactive TTS) | < 4000ms utterance end → first TTS byte |
| Latency (reactive TTS) | < 5000ms utterance end → first TTS byte |

---

## Phased Build & Test Plan

Build one block at a time. Each block is verified with mocks before moving on. Real APIs are only called once the block is independently confirmed working. Integration happens only after all blocks pass individually.

---

### Phase 0 — Project Setup (no tests, just scaffolding)

**Steps:**
1. Pin Pipecat version; run `python -c "from pipecat.frames.frames import UserStartedSpeakingFrame, BotInterruptionFrame; print('OK')"` to confirm frame names
2. `docker-compose.yml`: PostgreSQL 16 + pgvector, Redis → `docker compose up -d`
3. `pyproject.toml` with all deps → `uv sync`
4. DB migrations: run `scripts/migrate.py` → verify all tables exist with `\dt` in psql
5. `scripts/create_partitions.py` → verify weekly partitions created

**You know it's done when:** `docker compose ps` shows all services healthy; `psql` lists all tables; partition script creates `context_summaries_YYYY_wNN`.

---

### Phase 1 — Building Blocks (block-by-block, mock-first)

Each block is a standalone module with a test script in `tests/`. You run the test script, I tell you the exact command, exact input, and exactly what to look for in the output.

---

> **Performance design note:** Audio pipeline stages (gain, accumulation) use clean byte-in / byte-out signatures. VAD uses Pipecat's battle-tested `SileroVADAnalyzer` — adding the Rust-based `silero-vad-rust` would save ~1-2ms per chunk while STT + LLM calls take 500-3000ms, so it's not worth the build complexity in v1. If a genuine hot path emerges later, any stage can be dropped into a PyO3 Rust extension without touching the rest of the system.

> **Canonical test fixture location:** All blocks read from `tests/fixtures/input.wav` and write to `tests/output/`. Put your audio file at `tests/fixtures/input.wav` once — every block reuses it.

---

#### Block 1: Audio Gain Amplifier

**Files:** `pipeline/audio_gain.py`  
**No external APIs. No Pipecat. Pure NumPy. Simplest possible block.**

**What it does:** reads `tests/fixtures/input.wav`, applies ×5 gain, saves to `tests/output/gain_test.wav`. You listen and verify it's louder with no wrap-around distortion.

**Put your audio file at:**
```
tests/fixtures/input.wav   ← drop any WAV here (mono or stereo, any sample rate)
```

**How to run:**
```bash
python tests/test_audio_gain.py \
  --input  tests/fixtures/input.wav \
  --output tests/output/gain_test.wav \
  --gain   5.0
```

**Expected terminal output:**
```
[GAIN] Input  → peak amplitude: 1340 / 32767  (4.1% of max)
[GAIN] Gain   → ×5.0
[GAIN] Output → peak amplitude: 6700 / 32767  (20.5% of max)  — no clipping
[GAIN] Saved: tests/output/gain_test.wav
────────────────────────────────────────
Clipping test (synthetic 32000-peak input):
[GAIN] Input peak: 32000  →  Output peak: 32767  (hard-clipped, no wrap-around)  ✓
```

**You then open `tests/output/gain_test.wav`** in any audio player — it should sound the same as the original but clearly louder. If it sounds distorted or garbled, the clipping logic has a bug.

**Pass criteria:** `output_peak == min(input_peak * gain, 32767)` for both tests; saved WAV is audibly louder.

---

#### Block 2: VAD + Audio Accumulator

**Files:** `pipeline/vad_processor.py`  
**Uses Pipecat's built-in `SileroVADAnalyzer`. Custom FSM logic (60s cap, 1.5s gate, accumulation) layered on top in Python. No external APIs. Two run modes: file and real-time mic.**

**File mode** (automated, uses same `tests/fixtures/input.wav`):
```bash
python tests/test_vad.py --audio tests/fixtures/input.wav
```
The script reads the WAV, feeds it through VAD frame-by-frame, and logs utterance boundaries.

**Expected output** (varies with your actual audio — the important thing is the structure):
```
[VAD] UTTERANCE_START at 0.8s
[VAD] UTTERANCE_STOP  at 4.2s  →  emitting 3.4s chunk  (passes 1.5s gate)
[VAD] UTTERANCE_START at 6.1s
[VAD] UTTERANCE_STOP  at 7.9s  →  emitting 1.8s chunk  (passes 1.5s gate)
```

**Short-utterance gate test** (generate synthetic 0.8s tone):
```bash
python tests/test_vad.py --synthetic-short
# Expected: [VAD] DROP: utterance 0.8s < 1.5s minimum — no emit
```

**60s force-emit test** (generate synthetic 65s tone):
```bash
python tests/test_vad.py --synthetic-long
# Expected: [VAD] FORCE_EMIT at 60.0s — accumulator reset; new utterance starts
```

**Real-time mic mode** (testing only — speak into your microphone, see VAD fire live):
```bash
pip install sounddevice   # one-time, test-only dep — not in pyproject.toml
python tests/test_vad.py --mic
```
Speak normally. You should see:
```
[VAD] UTTERANCE_START  ← when you start speaking
[VAD] UTTERANCE_STOP   ← ~3s after you go silent
[VAD] emitting Xs chunk
```
Press Ctrl+C to stop. This mode is purely for human verification that VAD fires correctly on real hardware mic input.

**Pass criteria:** Gate drops short utterances; force-emit fires at 60s; mic mode logs START/STOP in real time.

---

#### Block 3: STT Provider Abstraction + Gemini STT

**Files:** `stt/client.py`, `stt/gemini.py`, `stt/factory.py`, `pipeline/stt_processor.py`  
**Mock first, then real API. Uses same `tests/fixtures/input.wav`. `stt_processor.py` calls `make_stt_client()` — never imports from `google-generativeai` directly.**

**Step 3a — Test with real API (one call to verify output format):**
```bash
python tests/test_stt.py --audio tests/fixtures/input.wav --real-api
```
*Use a WAV with two distinct voices for best results — the same file from Block 1/2.*

**Expected output (verify the format, not the exact words):**
```json
[
  {"start_ms": 0, "end_ms": 3200, "speaker_label": "Speaker A", "text": "...", "language": "en"},
  {"start_ms": 3400, "end_ms": 7100, "speaker_label": "Speaker B", "text": "...", "language": "en"}
]
```
**⚠️ If timestamps are missing** (Gemini returns only `speaker_label + text` without `start_ms`/`end_ms`), stop here and flag — the audio segmentation design must be revised.

**Step 3b — Save the output as a fixture:**
```bash
python tests/test_stt.py --audio tests/fixtures/input.wav --real-api --save-fixture tests/fixtures/stt_output.json
```

**Step 3c — Test with mock (all subsequent blocks use this fixture):**
```bash
python tests/test_stt.py --mock  # reads stt_output.json, prints same format without API call
```

**Pass criteria:** Format matches schema exactly; fixture saved; mock returns identical structure.

---

#### Block 4: Audio Segmenter + Resampler

**Files:** `pipeline/audio_segmenter.py`, `pipeline/audio_resampler.py`  
**No external APIs — uses same `tests/fixtures/input.wav` + STT fixture from Block 3.**

```bash
python tests/test_segmenter.py \
  --audio tests/fixtures/input.wav \
  --stt-fixture tests/fixtures/stt_output.json
```

**Expected output:**
```
Segment 0: speaker_label=Speaker A  duration=3.2s  sample_rate=16000  shape=(51200,)
Segment 1: speaker_label=Speaker B  duration=3.7s  sample_rate=16000  shape=(59200,)
Saved: tests/output/seg_SpeakerA.wav
Saved: tests/output/seg_SpeakerB.wav
```

Play `seg_SpeakerA.wav` and `seg_SpeakerB.wav` to confirm each contains the right voice.

**Also test overlap:**  
Manually add overlapping timestamps to the fixture (e.g., segment 0 ends at 3500ms, segment 1 starts at 3200ms). Expect the overlap region to go to the louder segment, and log: `[OVERLAP] 3200-3500ms → Speaker B (higher RMS)`.

**Pass criteria:** Saved WAVs are audibly correct; overlap test logs the resolution decision; all output is 16kHz.

---

#### Block 5: Speaker Embedder + Voiceprint Matcher

**Files:** `pipeline/speaker_embedder.py`, `pipeline/voiceprint_matcher.py`, `scripts/enroll_user.py`  
**Local ECAPA-TDNN model — no external APIs.**

**Step 4a — Enrollment:**
```bash
python scripts/enroll_user.py --user-id test_user --audio tests/fixtures/seg_SpeakerA.wav
```
**Expected:**
```
Extracted d-vector: shape=(192,)  inference_time=58ms
Saved to user_voiceprints: user_id=test_user
```

**Step 4b — Match (same speaker):**
```bash
python tests/test_voiceprint.py --user-id test_user --audio tests/fixtures/seg_SpeakerA_take2.wav
```
*Record a second clip of the same person saying something different.*  
**Expected:** `cosine_sim=0.89  is_wearer=True`

**Step 4c — Reject (different speaker):**
```bash
python tests/test_voiceprint.py --user-id test_user --audio tests/fixtures/seg_SpeakerB.wav
```
**Expected:** `cosine_sim=0.31  is_wearer=False`

**Step 4d — Uncertain (single speaker, low confidence):**  
Feed a very short segment (< 0.5s) where similarity falls between 0.50–0.75.  
**Expected:** `cosine_sim=0.62  is_wearer=Unknown  (skipping mem0 ingestion)`

**Pass criteria:** All four scenarios produce the expected `is_wearer` label.

---

#### Block 6: Session Router (pure logic)

**Files:** `pipeline/session_router.py`  
**No external APIs — pure Python logic.**

```bash
python tests/test_router.py
```

The test script runs a table of inputs:

| Input transcript | Session state | Backoff active | Expected route |
|---|---|---|---|
| `"I need to be at the airport by 3pm"` | AMBIENT | No | `AMBIENT` |
| `"hey jarvis book me a cab"` | AMBIENT | No | `REACTIVE` |
| `"hey jarvis book me a cab"` | AMBIENT | **Yes** | `REACTIVE` (still works) |
| `"sure, 2pm works for me"` | ACTIVE | No | `ACTIVE` |
| `"just some noise"` | AMBIENT | **Yes** | `SKIP` |

**Expected output:**
```
[ROUTER] transcript="I need to be at the airport..."  → AMBIENT  ✓
[ROUTER] transcript="hey jarvis book me a cab"        → REACTIVE ✓
[ROUTER] transcript="hey jarvis book me a cab" (backoff active) → REACTIVE ✓
[ROUTER] transcript="sure, 2pm works for me" (ACTIVE) → ACTIVE  ✓
[ROUTER] transcript="just some noise" (backoff)        → SKIP    ✓
All 5/5 passed.
```

**Pass criteria:** 5/5 correct routes with no external calls.

---

#### Block 7: Ambient Processor (LLM mocked)

**Files:** `ambient/processor.py`  
**Mock: LLM returns a hardcoded `AmbientAnalysis` fixture.**

**Step 6a — Test with mock LLM:**
```bash
python tests/test_ambient_processor.py --mock
```
*The mock returns a pre-written AmbientAnalysis JSON whenever called.*

**Input (printed by test script):**
```
Transcript: "Speaker A: I should book a table at that Italian place for tonight"
Memories: ["User likes Ristorante Roma", "User usually books for 2"]
```

**Expected output:**
```json
{
  "memory_operations": [{"op": "add", "fact": "User wants to book Ristorante Roma tonight"}],
  "summary": "User is planning to dine at an Italian restaurant tonight.",
  "extracted_facts": ["User mentioned booking a restaurant for tonight"],
  "confidence": 0.83,
  "proposed_action": "offer to book a table at Ristorante Roma",
  "consent_prompt": "Want me to book a table at Ristorante Roma for tonight?",
  "should_act": true
}
```

**Step 6b — Test with real LLM (one call to validate prompt quality):**
```bash
python tests/test_ambient_processor.py --real-api \
  --transcript "Speaker A: I should book a table at that Italian place for tonight"
```
Compare real output structure against schema — all fields present? Confidence plausible? Summary specific (not vague)?

**Pass criteria:** Mock returns valid schema; real API output matches schema and is non-vague.

---

#### Block 8: Memory Writer (mem0 + real local DB)

**Files:** `ambient/memory_writer.py`  
**Uses real mem0 + local PostgreSQL — no Gemini call needed.**

```bash
python tests/test_memory_writer.py
```

**What the test does:**
1. Calls `memory_writer.write([{op:"add", fact:"User prefers window seats"}], user_id="test_user")`
2. Calls `memory.search("seating preferences", user_id="test_user")` → prints results
3. Calls update on the returned memory ID
4. Calls delete → verifies fact is gone

**Expected output:**
```
[MEM0] ADD: "User prefers window seats"  → memory_id=abc123
[MEM0] SEARCH "seating preferences" → ["User prefers window seats"]
[MEM0] UPDATE abc123 → "User strongly prefers window seats at restaurants"
[MEM0] DELETE abc123
[MEM0] SEARCH "seating preferences" → []  (empty — confirmed deleted)
All 4 operations successful.
```

**Pass criteria:** Correct round-trip: add → search → update → delete → gone.

---

#### Block 9: Context Writer (PostgreSQL)

**Files:** `ambient/context_writer.py`  
**Uses real local PostgreSQL.**

```bash
python tests/test_context_writer.py
```

Feed a hardcoded AmbientAnalysis fixture. Verify rows in both tables:

**Expected output:**
```
[CTX] Inserted context_summaries id=xyz  user_id=test_user  partition=context_summaries_2026w20
[CTX] Inserted raw_transcripts id=abc  linked to summary
[CTX] SELECT summary FROM context_summaries WHERE id='xyz':
      "User is planning to dine at an Italian restaurant tonight."  ✓
[CTX] SELECT raw_transcript FROM raw_transcripts WHERE id='abc':
      "Speaker A: I should book a table at..."  ✓
```

**Pass criteria:** Both rows present; summary matches input; raw transcript matches input.

---

#### Block 10: Tool Registry + Selector (pgvector)

**Files:** `tools/registry.py`, `tools/selector.py`, `scripts/register_tools.py`

**Step 9a — Register mock tools:**
```bash
python scripts/register_tools.py --tools tests/fixtures/mock_tools.json
```
`mock_tools.json` contains 10 fake tools with names and descriptions (e.g., `book_restaurant`, `book_cab`, `send_whatsapp`, `play_music`, etc.)

**Expected:**
```
Registered book_restaurant  embedding_dim=768  ✓
Registered book_cab         embedding_dim=768  ✓
... (10 total)
```

**Step 9b — Test selector:**
```bash
python tests/test_tool_selector.py --query "book a table at a restaurant for tonight"
```

**Expected:**
```
Top-2 tools for "book a table at a restaurant for tonight":
  1. book_restaurant  similarity=0.91
  2. maps_search      similarity=0.72
```

```bash
python tests/test_tool_selector.py --query "send a message to my contact"
```
**Expected:**
```
  1. send_whatsapp   similarity=0.89
  2. contacts_lookup similarity=0.81
```

**Pass criteria:** Semantically correct top-2 for each query; different queries return different tools.

---

#### Block 11: Tool Executor (MCP mocked)

**Files:** `tools/executor.py`  
**Mock: A tiny FastAPI server that echoes requests — no real tool integrations.**

**Start mock MCP server:**
```bash
python tests/fixtures/mock_mcp_server.py --port 8888
# Responds to any tool call with {"status": "success", "result": "mock result for <tool_name>"}
```

**Test read tool (retry):**
```bash
python tests/test_tool_executor.py --tool weather_fetch --type read --fail-first-2
```
**Expected:**
```
[EXECUTOR] Attempt 1: weather_fetch → FAILED (mock failure)
[EXECUTOR] Attempt 2: weather_fetch → FAILED (mock failure)
[EXECUTOR] Attempt 3: weather_fetch → SUCCESS  result={"status":"success","result":"mock result for weather_fetch"}
```

**Test write tool (no retry, idempotency key):**
```bash
python tests/test_tool_executor.py --tool book_cab --type write --fail-on-call
```
**Expected:**
```
[EXECUTOR] Attempt 1: book_cab (write) → FAILED
[EXECUTOR] No retry for write tools. Returning error to caller.
error={"type": "MCPCallError", "tool": "book_cab", "message": "..."}
```

**Pass criteria:** Read tool retries 3 times then succeeds; write tool fails once and surfaces error without retrying.

---

#### Block 12: Proactive LangGraph

**Files:** `action/proactive_graph.py`, `action/shared_nodes.py`, `action/state.py`  
**Mock: LLM returns scripted responses; tool executor uses mock MCP server from Block 11.**

```bash
python tests/test_proactive_graph.py --scenario decline
python tests/test_proactive_graph.py --scenario approve-single-step
python tests/test_proactive_graph.py --scenario approve-multi-step
```

**Scenario: decline**  
Input: AmbientAnalysis fixture with consent_prompt = "Want me to book a table at Ristorante Roma?"  
Mock user reply: "No thanks"  
**Expected output:**
```
[GRAPH] Suggest node: TTS → "Want me to book a table at Ristorante Roma?"
[GRAPH] User replied: "No thanks"
[GRAPH] DONE → AMBIENT  (zero tool calls made)
ambient_logs.user_outcome = "declined"  ✓
```

**Scenario: approve-single-step**  
Mock user reply: "Yes please"  
Mock Plan node output: `{done: false, next_step: "book table at Ristorante Roma for tonight, 2 people"}`  
Mock Plan node output (turn 2): `{done: true}`  
**Expected:**
```
[GRAPH] Suggest → "Yes" → Plan node
[GRAPH] Tool Select: top-2 for "book table..." → [book_restaurant, maps_search]
[GRAPH] Execute: book_restaurant → mock result
[GRAPH] Plan (turn 2): done=true
[GRAPH] Respond: TTS → "Done! Booked Ristorante Roma for 2 tonight."
[GRAPH] DONE → AMBIENT  memory_ops flushed ✓
```

**Scenario: approve-multi-step**  
3 tool steps (book table → send WhatsApp confirmation → set reminder). Verify different tool schemas at each step.

**Pass criteria:** Each scenario produces the exact log sequence above; memory flush happens on DONE.

---

#### Block 13: Reactive LangGraph

**Files:** `action/reactive_graph.py`  
**Same mock setup as Block 12.**

```bash
python tests/test_reactive_graph.py --scenario direct
python tests/test_reactive_graph.py --scenario clarify-then-execute
```

**Scenario: direct** (intent is clear, no clarification needed)  
Input transcript: `"hey jarvis, play some jazz music"`  
**Expected:**
```
[GRAPH] Intent node: intent_clear=True  → Plan node directly (no consent TTS)
[GRAPH] Tool Select: top-2 for "play jazz music" → [spotify_play, music_search]
[GRAPH] Execute: spotify_play → mock result
[GRAPH] Respond: TTS → "Playing jazz on Spotify."
[GRAPH] DONE → AMBIENT
```

**Scenario: clarify-then-execute** (intent needs one question)  
Input: `"hey jarvis, book the usual"`  
**Expected:**
```
[GRAPH] Intent node: ambiguous → one question: "Book a cab or a restaurant?"
[GRAPH] User replied: "A cab to the office"
[GRAPH] Plan node → Tool Select → Execute → Respond → DONE
```

**Pass criteria:** No consent prompt in reactive path; clarification fires exactly once when ambiguous.

---

#### Block 14: TTS Processor + ACK Frame

**Files:** `pipeline/tts_processor.py`, `assets/ack_chime.wav`

**Step 13a — Test ACK frame (no API):**
```bash
python tests/test_tts.py --ack-only
# Injects ack_chime.wav as AudioRawFrame into mock transport; plays locally
```
**Expected:** You hear the chime sound immediately (< 50ms).

**Step 13b — Test Gemini TTS:**
```bash
python tests/test_tts.py --text "Want me to book a table at Ristorante Roma for tonight?" --save tests/output/tts_test.wav
```
**Expected:** `tts_test.wav` exists; audibly contains the sentence; first byte received in < 600ms (logged).

**Pass criteria:** Chime plays; TTS WAV is audibly correct; latency logged.

---

### Phase 2 — Integration Testing

Run only after all 14 blocks pass individually. Each integration test wires 2+ blocks together and verifies the seam.

---

#### Integration 1: VAD → STT (STT still mocked)

```bash
python tests/integration/test_vad_stt.py \
  --audio tests/fixtures/input.wav \
  --stt-mock
```

**What it tests:** VAD emits an utterance chunk → STT mock receives the exact bytes → returns fixture transcript.

**Expected:**
```
[VAD] UTTERANCE_STOP  → emitting 3.0s chunk  (48000 bytes @ 16kHz)
[STT-MOCK] received 48000 bytes → returning fixture transcript
[RESULT] [{start_ms:0, speaker_label:"Speaker A", text:"..."}]
```

**Pass criteria:** Byte count from VAD matches what STT mock received; transcript returned.

---

#### Integration 2: STT → Segmenter → Embedder → Matcher

```bash
python tests/integration/test_speaker_pipeline.py \
  --audio tests/fixtures/input.wav \
  --user-id test_user \
  --stt-mock   # uses saved fixture
```

**Expected:**
```
[STT-MOCK] → 2 segments
[SEGMENTER] Segment 0: SpeakerA 3.2s  Segment 1: SpeakerB 3.7s
[RESAMPLER] both → 16kHz
[EMBEDDER] SpeakerA d-vector: (192,)  inference=57ms
[EMBEDDER] SpeakerB d-vector: (192,)  inference=54ms
[MATCHER] SpeakerA: cosine_sim=0.88  is_wearer=True
[MATCHER] SpeakerB: cosine_sim=0.29  is_wearer=False
```

**Pass criteria:** Wearer correctly identified in both segments.

---

#### Integration 3: Router → Ambient Processor (LLM mocked)

```bash
python tests/integration/test_router_ambient.py \
  --transcript "Speaker A: I should book the Italian place for tonight" \
  --session-state AMBIENT \
  --llm-mock
```

**Expected:**
```
[ROUTER] → AMBIENT
[AMBIENT] mem0 search → 2 memories found
[AMBIENT-MOCK-LLM] → AmbientAnalysis: confidence=0.83, should_act=True
[AMBIENT] ambient_logs INSERT ✓  ambient_log_id=uuid123
[AMBIENT] context_summaries INSERT ✓
[AMBIENT] memory_ops queued (not yet written — no session spawned in this test)
```

**Also test:** Router → REACTIVE path (skip ambient entirely):
```bash
python tests/integration/test_router_ambient.py \
  --transcript "hey jarvis, book me a cab" \
  --session-state AMBIENT
```
**Expected:** `[ROUTER] → REACTIVE  (ambient processor NOT called)`

---

#### Integration 4: Ambient → Action Agent (LLM still mocked)

```bash
python tests/integration/test_ambient_to_action.py \
  --confidence 0.83 \
  --llm-mock \
  --user-reply "yes please"
```

**Expected:**
```
[AMBIENT] confidence=0.83 > 0.75 → spawning ACTIVE_SESSION(proactive)
[SESSION] Redis + PG sessions row created  langgraph_thread_id=test_session
[ACTION] Suggest node: TTS text="Want me to book Ristorante Roma for tonight?"
[ACTION] User: "yes please" → Plan node
[ACTION] Tool Select + Execute (mock) → book_restaurant
[ACTION] Respond: TTS text="Booked for 2 tonight!"
[ACTION] DONE → flush memory ops → PG sessions.state=AMBIENT
ambient_logs.user_outcome="confirmed" ✓
```

---

#### Integration 5: Full Pipeline (real STT, real LLM, mock tools)

```bash
python tests/integration/test_full_pipeline.py \
  --audio tests/fixtures/input.wav \
  --user-id test_user \
  --tool-mock   # only tools remain mocked
```

Wire everything: VAD → STT (real API) → Segmenter → Embedder → Matcher → Router → Ambient (real LLM) → Action Agent (real LLM) → TTS (real API).

**Expected:** End-to-end conversation plays through; memory written; context stored; TTS audio saved to file.

---

#### Integration 6: Full System with WebRTC Client

```bash
# Terminal 1: start server
uvicorn server.main:app --reload

# Terminal 2: run reference client
python client/client.py --user-id test_user --mic
```

Speak into microphone. Hear response. Verify DB rows written.

---

### Phase 3 — Production Readiness (after integration passes)

- Token counter validated against real API usage in logs
- Barge-in tested live (speak while agent is responding)
- Timeout tested (stay silent for 35s mid-session)
- Memory flush on crash tested (kill server mid-session, restart, verify mem0 flushed from LangGraph checkpoint)
- Weekly partition auto-creation tested (advance system clock, run cleanup_job)
- Load test: 5 concurrent WebRTC sessions, verify no asyncio blocking from ECAPA-TDNN inference
