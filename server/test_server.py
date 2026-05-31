"""
Proactive AI — Test Server.

A standalone FastAPI server for local testing of all pipeline blocks.
Runs without Pipecat / Daily WebRTC.

Live session uses WebSocket + server-side Silero VAD (continuous audio).
Enrollment still uses HTTP multipart upload.

Start:
    .venv/bin/python -m uvicorn server.test_server:app --reload --port 8000

Then open: http://localhost:8000
"""
import asyncio
import base64
import io
import json
import logging
import os
import re
import tempfile
import uuid
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Proactive AI — Test Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIENT_DIR = Path(__file__).parent.parent / "client"


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def frontend():
    index = CLIENT_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Place client/index.html in the project root</h1>", status_code=404)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Insights & context ────────────────────────────────────────────────────────

@app.get("/api/context/{user_id}")
async def get_context(user_id: str, limit: int = 50):
    """Recent context summaries with extracted facts and tags, grouped by day."""
    from server.ambient.context_writer import get_recent_summaries
    rows = await get_recent_summaries(user_id, limit=limit)
    # Deserialise JSONB strings
    for r in rows:
        for k in ("extracted_facts", "tags"):
            if isinstance(r.get(k), str):
                try:
                    r[k] = json.loads(r[k])
                except Exception:
                    r[k] = []
        if r.get("created_at"):
            r["created_at"] = r["created_at"].isoformat()
    return {"summaries": rows}


@app.get("/api/memories/{user_id}")
async def get_memories(user_id: str):
    """All stored mem0 memories (facts) for this user."""
    from server.ambient.memory_writer import get_all_memories
    mems = await get_all_memories(user_id)
    return {"memories": mems}


@app.get("/api/rollup/{user_id}")
async def get_daily_rollup(user_id: str):
    """Aggregate today's context summaries into a daily rollup (facts + tags + summary count)."""
    import asyncpg
    from server.config import settings

    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            rows = await conn.fetch(
                """
                SELECT summary, extracted_facts, tags, created_at
                FROM context_summaries
                WHERE user_id = $1
                  AND created_at >= CURRENT_DATE
                ORDER BY created_at ASC
                """,
                user_id,
            )
        finally:
            await conn.close()
    except Exception as e:
        return {"error": str(e), "summaries": [], "all_facts": [], "all_tags": [], "count": 0}

    summaries = []
    all_facts: list[str] = []
    all_tags: set[str] = set()

    for r in rows:
        row = dict(r)
        for k in ("extracted_facts", "tags"):
            if isinstance(row.get(k), str):
                try:
                    row[k] = json.loads(row[k])
                except Exception:
                    row[k] = []
        if row.get("created_at"):
            row["created_at"] = row["created_at"].isoformat()
        summaries.append(row)
        all_facts.extend(row.get("extracted_facts") or [])
        all_tags.update(row.get("tags") or [])

    # Deduplicate facts preserving order
    seen: set[str] = set()
    unique_facts = [f for f in all_facts if f not in seen and not seen.add(f)]  # type: ignore[func-returns-value]

    return {
        "date": str(__import__("datetime").date.today()),
        "count": len(summaries),
        "summaries": summaries,
        "all_facts": unique_facts,
        "all_tags": sorted(all_tags),
    }


# ── Enrollment (HTTP multipart) ───────────────────────────────────────────────

@app.post("/api/enroll")
async def enroll(
    user_id: str        = Form(...),
    audio:   UploadFile = File(...),
):
    """Record wearer voice → extract 256-dim d-vector → store in PostgreSQL."""
    import asyncpg, subprocess
    from server.config import settings
    from server.pipeline.speaker_embedder import extract_embedding

    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        # Decode to 16kHz mono int16 PCM via ffmpeg — avoids torchcodec/libavutil dependency
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp, "-f", "s16le", "-ar", "16000", "-ac", "1", "-"],
            capture_output=True,
        )
    finally:
        os.unlink(tmp)

    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        raise HTTPException(500, f"Audio decode failed: {stderr[-300:]}")

    pcm = proc.stdout
    num_samples = len(pcm) // 2
    sr = 16000
    duration_s = num_samples / sr
    if duration_s < 2.0:
        raise HTTPException(400, f"Too short ({duration_s:.1f}s) — speak for at least 2 seconds")

    d_vector = await extract_embedding(pcm, src_rate=sr)
    if d_vector is None:
        raise HTTPException(400, "Audio too quiet — speak louder and try again")

    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        await conn.execute(
            """
            INSERT INTO user_voiceprints (user_id, d_vector)
            VALUES ($1, $2::vector)
            ON CONFLICT (user_id)
            DO UPDATE SET d_vector = EXCLUDED.d_vector, enrolled_at = now()
            """,
            user_id,
            "[" + ",".join(f"{x:.6f}" for x in d_vector.tolist()) + "]",
        )
    finally:
        await conn.close()

    logger.info("[ENROLL] user=%s  duration=%.1fs  d_vector=%s", user_id, duration_s, d_vector.shape)
    return {"status": "enrolled", "user_id": user_id, "duration_s": round(duration_s, 2)}


# ── Shared pipeline ───────────────────────────────────────────────────────────

async def _run_pipeline(pcm: bytes, user_id: str, session_id: str, sr: int = 16000) -> dict:
    """
    Full pipeline on a complete utterance blob (int16 PCM at 16kHz).
    Returns a result dict suitable for JSON serialisation.
    """
    from server.config import settings
    from server.pipeline.stt_processor import STTProcessor
    from server.pipeline.audio_segmenter import segment_audio
    from server.pipeline.speaker_embedder import extract_embedding
    from server.pipeline.voiceprint_matcher import identify_wearer
    from server.pipeline.session_router import route
    from server.ambient.cost_governor import CostGovernor
    from server.ambient.processor import AmbientProcessor
    from server.ambient.context_writer import write_context
    from server.ambient.memory_writer import apply_memory_ops, search_memories
    from server.llm.factory import make_llm_client
    from server.tts.factory import make_tts_client

    # ── STT ───────────────────────────────────────────────────────────────────
    try:
        segments = await STTProcessor().transcribe(pcm)
    except Exception as e:
        err = str(e)
        if "nodename nor servname" in err or "ConnectError" in type(e).__name__ or "ConnectError" in err:
            logger.error("[PIPELINE] STT network error: %s", err)
            return {"error": "STT unavailable — check internet connection"}
        logger.error("[PIPELINE] STT failed: %s", err)
        return {"error": f"STT error: {err[:120]}"}

    if not segments:
        return {"route": "SKIP", "reason": "no speech"}

    full_transcript = " | ".join(f"{s.speaker_label}: {s.text}" for s in segments)
    logger.info("[PIPELINE] transcript: %s", full_transcript[:120])

    # ── Voiceprint matching ───────────────────────────────────────────────────
    enrolled = await _fetch_dvec(user_id, settings)
    wearer_label = None

    if enrolled is not None:
        audio_segs = segment_audio(pcm, sr, segments)
        pairs = []
        for seg in audio_segs:
            emb = await extract_embedding(seg.pcm_bytes)
            pairs.append((seg.speaker_label, emb))
        if pairs:
            ranked = identify_wearer(pairs, enrolled)
            top = ranked[0] if ranked else None
            if top and top.is_wearer == "True":
                wearer_label = top.speaker_label
                logger.info("[PIPELINE] wearer=%s sim=%.3f", wearer_label, top.cosine_sim)

    wearer_text = " ".join(
        s.text for s in segments if s.speaker_label == wearer_label
    ) if wearer_label else " ".join(s.text for s in segments)

    # ── Session router ────────────────────────────────────────────────────────
    governor = CostGovernor()
    governor.on_vad_start()
    decision = route(full_transcript, "AMBIENT", governor)
    logger.info("[PIPELINE] route=%s", decision)

    result: dict = {
        "transcript":   full_transcript,
        "segments":     [s.model_dump() for s in segments],
        "wearer_label": wearer_label,
        "wearer_text":  wearer_text,
        "route":        decision,
        "analysis":     None,
        "tts_audio":    None,
        "tts_text":     None,
    }

    # ── Ambient analysis ──────────────────────────────────────────────────────
    spoken = None
    if decision == "AMBIENT" and wearer_text.strip():
        try:
            from server.tools.manifest import get_manifest
            capability_manifest = await get_manifest()

            # Fetch relevant memories for this utterance
            memories = await search_memories(wearer_text, user_id)

            analysis = await AmbientProcessor(
                client=make_llm_client("ambient")
            ).analyse(transcript=wearer_text, memories=memories, capability_manifest=capability_manifest)
            result["analysis"] = analysis.model_dump()

            # Fire-and-forget: persist context summary + raw transcript
            speaker_labels = list({s.speaker_label for s in segments})
            asyncio.create_task(write_context(
                analysis=analysis,
                raw_transcript=full_transcript,
                user_id=user_id,
                session_id=session_id,
                speaker_labels=speaker_labels,
            ))

            # Fire-and-forget: apply memory ops (add/update/delete facts)
            if analysis.memory_operations:
                asyncio.create_task(apply_memory_ops(analysis.memory_operations, user_id))

            if analysis.should_act and analysis.consent_prompt:
                spoken = analysis.consent_prompt
                logger.info("[PIPELINE] proactive suggestion: %r", spoken)
        except Exception as e:
            logger.warning("[PIPELINE] ambient failed: %s", e)

    elif decision == "REACTIVE":
        spoken = "Hey! I'm listening — what do you need?"

    # ── TTS ───────────────────────────────────────────────────────────────────
    if spoken:
        try:
            tts    = make_tts_client()
            chunks = []
            async for chunk in tts.synthesize_stream(spoken):
                chunks.append(chunk)
            pcm_out = b"".join(chunks)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(tts.sample_rate)
                wf.writeframes(pcm_out)

            result["tts_audio"] = base64.b64encode(buf.getvalue()).decode()
            result["tts_text"]  = spoken
        except Exception as e:
            logger.warning("[PIPELINE] TTS failed: %s", e)

    return result


# ── Consent detection ────────────────────────────────────────────────────────

_YES = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|please|absolutely|correct|right|proceed|sounds good)\b",
    re.IGNORECASE,
)
_NO = re.compile(
    r"\b(no|nope|nah|don't|dont|stop|cancel|never mind|nevermind|forget it|not now|skip)\b",
    re.IGNORECASE,
)


def _detect_consent(transcript: str) -> str:
    """Return 'yes', 'no', or 'unclear'."""
    if _YES.search(transcript):
        return "yes"
    if _NO.search(transcript):
        return "no"
    return "unclear"


async def _handle_consent(pcm: bytes, user_id: str, pending: dict) -> dict:
    """
    Transcribe the user's consent response and decide yes/no.
    Returns a result dict like _run_pipeline.
    """
    from server.pipeline.stt_processor import STTProcessor
    from server.tts.factory import make_tts_client

    try:
        segments = await STTProcessor().transcribe(pcm)
    except Exception as e:
        logger.warning("[CONSENT] STT failed: %s", e)
        segments = []

    transcript = " ".join(s.text for s in segments) if segments else ""
    consent = _detect_consent(transcript)
    logger.info("[CONSENT] transcript=%r  decision=%s", transcript, consent)

    if consent == "yes":
        spoken = f"Got it! I'll {pending['proposed_action']}."
    elif consent == "no":
        spoken = "No problem, I'll leave it."
    else:
        spoken = "Sorry, I didn't catch that — did you want me to go ahead? Just say yes or no."

    tts_audio = None
    try:
        tts = make_tts_client()
        chunks = []
        async for chunk in tts.synthesize_stream(spoken):
            chunks.append(chunk)
        pcm_out = b"".join(chunks)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts.sample_rate)
            wf.writeframes(pcm_out)
        tts_audio = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning("[CONSENT] TTS failed: %s", e)

    return {
        "route":        "CONSENT",
        "transcript":   f"[Consent] {transcript}",
        "segments":     [s.model_dump() for s in segments],
        "wearer_label": None,
        "wearer_text":  transcript,
        "analysis":     None,
        "tts_text":     spoken,
        "tts_audio":    tts_audio,
        "consent":      consent,
    }


# ── WebSocket — continuous VAD session ───────────────────────────────────────

@app.websocket("/ws/session/{user_id}")
async def session_ws(websocket: WebSocket, user_id: str):
    """
    Continuous session endpoint.

    Browser streams 1024-byte chunks (512 int16 samples @ 16 kHz) produced
    by an AudioWorkletProcessor.  Server feeds them through Silero VAD.
    When VAD emits a complete utterance the full pipeline runs and the result
    is pushed back as a JSON message.

    Message types sent TO the browser:
      {"type": "vad_status", "status": "speaking"|"processing"|"listening"}
      {"type": "result",     ...pipeline result fields...}
      {"type": "error",      "message": "..."}
    """
    from server.pipeline.vad_processor import VadProcessor, MIN_SPEECH_MS, MIN_SPEECH_MS_CONSENT

    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("[WS] connected  user=%s  session=%s", user_id, session_id)

    vad = VadProcessor()

    # Consent state — set when ambient fires a high-confidence suggestion
    consent_state: dict = {"mode": "AMBIENT", "pending": None}
    # pending = {"proposed_action": str, "consent_prompt": str}

    async def handle_utterance(pcm: bytes) -> None:
        """Run pipeline on a complete utterance and push result to browser."""
        try:
            await websocket.send_text(json.dumps({"type": "vad_status", "status": "processing"}))

            if consent_state["mode"] == "AWAITING_CONSENT":
                result = await _handle_consent(pcm, user_id, consent_state["pending"])
                # On yes/no clear the state; on unclear stay in AWAITING_CONSENT
                if result["consent"] in ("yes", "no"):
                    consent_state["mode"] = "AMBIENT"
                    consent_state["pending"] = None
                    vad.min_speech_ms = MIN_SPEECH_MS  # restore normal gate
                    logger.info("[WS] consent=%s — returning to AMBIENT", result["consent"])
            else:
                result = await _run_pipeline(pcm, user_id, session_id)
                # Enter consent mode if ambient made a high-confidence suggestion
                analysis = result.get("analysis") or {}
                if (
                    analysis.get("should_act")
                    and analysis.get("confidence", 0) >= 0.75
                    and analysis.get("consent_prompt")
                ):
                    consent_state["mode"] = "AWAITING_CONSENT"
                    consent_state["pending"] = {
                        "proposed_action": analysis.get("proposed_action", "help you"),
                        "consent_prompt":  analysis.get("consent_prompt", ""),
                    }
                    vad.min_speech_ms = MIN_SPEECH_MS_CONSENT  # accept short yes/no replies
                    logger.info("[WS] entering AWAITING_CONSENT — action: %s", consent_state["pending"]["proposed_action"])

            await websocket.send_text(json.dumps({"type": "result", **result}))
        except Exception as e:
            logger.error("[WS] pipeline error: %s", e)
            try:
                await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
        finally:
            try:
                await websocket.send_text(json.dumps({"type": "vad_status", "status": "listening"}))
            except Exception:
                pass

    _speaking = False

    try:
        while True:
            data = await websocket.receive_bytes()

            # Feed 1024-byte chunk (512 int16 samples) to VAD
            utterance = vad.process_chunk(data)

            # Detect start/end transitions to send status to UI
            # VADProcessor logs internally; we infer state by watching the buffer
            # A simple heuristic: if we just got an utterance, we were speaking
            if utterance is not None:
                _speaking = False
                asyncio.create_task(handle_utterance(utterance))
            else:
                # Check whether VAD thinks we're in speech (accumulator active)
                currently_speaking = vad._state.active
                if currently_speaking and not _speaking:
                    _speaking = True
                    await websocket.send_text(json.dumps({"type": "vad_status", "status": "speaking"}))
                elif not currently_speaking and _speaking:
                    _speaking = False
                    # Don't send "listening" here — the handle_utterance task
                    # will send it after the pipeline completes (or VAD dropped it)
                    await websocket.send_text(json.dumps({"type": "vad_status", "status": "listening"}))

    except WebSocketDisconnect:
        logger.info("[WS] disconnected  user=%s", user_id)
    except Exception as e:
        logger.error("[WS] error: %s", e)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _fetch_dvec(user_id: str, settings) -> np.ndarray | None:
    try:
        import asyncpg
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            row = await conn.fetchrow(
                "SELECT d_vector::text FROM user_voiceprints WHERE user_id = $1", user_id
            )
        finally:
            await conn.close()
        if not row:
            return None
        vals = [float(x) for x in row["d_vector"].strip("[]").split(",")]
        return np.array(vals, dtype="float32")
    except Exception as e:
        logger.warning("[FETCH_DVEC] %s", e)
        return None
