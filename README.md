# Proactive AI

Wearable AI assistant — audio streams via Daily WebRTC to a server that runs VAD, STT with diarization, speaker identification, and a proactive/reactive AI agent.

---

## Setup

```bash
# Start PostgreSQL + Redis
docker compose up -d

# Install dependencies
uv sync

# Copy and fill in secrets
cp .env.example .env
```

Put your test audio file at `tests/fixtures/input.wav` (any WAV, any sample rate — mono or stereo).

---

## Testing Building Blocks

Each block has a standalone test script. Run them independently to verify that stage works before moving on.

### Block 1 — Audio Gain Amplifier

Reads your WAV, applies ×5 gain, saves output. Open the output WAV and verify it sounds louder with no distortion.

```bash
python tests/test_audio_gain.py \
  --input  tests/fixtures/input.wav \
  --output tests/output/gain_test.wav \
  --gain   5.0
```

---

### Block 2 — VAD + Audio Accumulator

Three modes:

**Synthetic tests** — verifies FSM logic (gate drop, force-emit, two utterances):
```bash
python tests/test_vad.py --synthetic
```
Expected: `4/4 passed`

**File mode** — runs real Silero VAD on your audio:
```bash
python tests/test_vad.py --audio tests/fixtures/input.wav
```
Expected: utterance count and durations printed.

**Mic mode** — speak into your microphone and see VAD fire live:
```bash
uv pip install sounddevice   # one-time, test-only
python tests/test_vad.py --mic
```
Speak → see `UTTERANCE_START`. Go silent for ~3s → see `UTTERANCE_STOP` with duration. Press Ctrl+C to stop.

---

### Block 3 — STT (Gemini)

**Step 1 — Real API call** (needs `GEMINI_API_KEY` in `.env`):
```bash
python tests/test_stt.py --real-api --save-fixture
```
Sends your `input.wav` to Gemini, prints each speaker segment with timestamps, and saves the result to `tests/fixtures/stt_output.json`.

**Critical check:** every segment must have `start_ms` and `end_ms`. If timestamps are missing, stop — Block 4 depends on them.

**Step 2 — Mock mode** (no API call, uses saved fixture):
```bash
python tests/test_stt.py --mock
```
All subsequent blocks use this fixture instead of hitting the API.

---

### Block 4 — Audio Segmenter + Resampler

Slices the utterance PCM by STT timestamps. Needs `tests/fixtures/stt_output.json` from Block 3.

```bash
python tests/test_segmenter.py
```

Expected: `2/2 passed`. Per-speaker WAVs saved to `tests/output/` — open them and verify each contains the right voice.

---

### Block 5 — Speaker Embedder + Voiceprint Matcher

Needs segment WAVs from Block 4 in `tests/output/`. First run downloads the ECAPA-TDNN model (~100MB).

```bash
python tests/test_voiceprint.py
```

Expected: `5/5 passed`. Enroll a real user into the DB:

```bash
python scripts/enroll_user.py --user-id alice --audio tests/fixtures/input.wav
```

---

### Block 10 — Tool Registry + Selector

Requires Docker + migrated DB.

```bash
# Register mock tools
python scripts/register_tools.py --tools tests/fixtures/mock_tools.json

# Run semantic search test
python tests/test_tool_selector.py
```

Expected: `6/6 passed`

---

### Block 9 — Context Writer

Requires Docker + migrated DB.

```bash
python tests/test_context_writer.py
```

Expected: `4/4 passed`

---

### Block 8 — Memory Writer (mem0 self-hosted)

Requires Docker running and DB migrated.

```bash
docker compose up -d
python scripts/migrate.py
python tests/test_memory_writer.py
```

Expected: `5/5 passed` — full add → search → update → delete round-trip.

---

### Block 7 — Ambient Processor

**Mock (no API call):**
```bash
python tests/test_ambient_processor.py
```

**Real API** (needs `OPENROUTER_API_KEY` in `.env`):
```bash
python tests/test_ambient_processor.py --real-api
python tests/test_ambient_processor.py --real-api --transcript "I should book the Italian place tonight"
```

---

### Block 6 — Session Router + Cost Governor

Pure logic — no APIs, no DB.

```bash
python tests/test_router.py
```

Expected: `5/5 passed`

---

### Utterance Queue

Bounded async queue between VAD (fast) and downstream pipeline (slow). Drops newest utterances when full — never blocks VAD.

```bash
python tests/test_utterance_queue.py
```

Expected: `4/4 passed`

---

## Project Structure

```
server/
├── pipeline/          # Audio processing stages
│   ├── audio_gain.py      # Stage 1: ×5 PCM amplifier
│   └── vad_processor.py   # Stage 2: Silero VAD + 60s cap + 1.5s gate
├── stt/               # STT abstraction (ABC + Gemini impl + factory)
├── tts/               # TTS abstraction
├── llm/               # LLM abstraction (OpenRouter backend)
├── embeddings/        # Embedding abstraction
├── prompts/           # All prompt strings centralised here
├── ambient/           # Ambient processor + memory/context writers
├── action/            # LangGraph proactive + reactive graphs
└── config.py          # All provider/model config (edit .env to swap)

tests/
├── fixtures/          # input.wav + generated fixtures
├── output/            # Test output files (gitignored)
└── test_*.py          # One test script per block

scripts/
├── migrate.py         # Create all DB tables (idempotent)
└── create_partitions.py  # Create weekly partitions for context_summaries
```
