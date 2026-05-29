"""
Proactive AI — Test Server.

A standalone FastAPI server for local testing of all pipeline blocks.
Runs without Pipecat / Daily WebRTC — uses simple HTTP push-to-talk.

Start:
    .venv/bin/python -m uvicorn server.test_server:app --reload --port 8000

Then open: http://localhost:8000
"""
import asyncio
import base64
import io
import logging
import os
import tempfile
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

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


# ── Enrollment ────────────────────────────────────────────────────────────────

@app.post("/api/enroll")
async def enroll(
    user_id: str        = Form(...),
    audio:   UploadFile = File(...),
):
    """Record wearer voice → extract 256-dim d-vector → store in PostgreSQL."""
    import asyncpg, torchaudio
    from server.config import settings
    from server.pipeline.speaker_embedder import extract_embedding

    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        waveform, sr = torchaudio.load(tmp)
    finally:
        os.unlink(tmp)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    pcm = (waveform.squeeze().numpy() * 32767).astype("int16").tobytes()

    duration_s = waveform.shape[1] / sr
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


# ── Full pipeline ─────────────────────────────────────────────────────────────

@app.post("/api/process")
async def process_audio(
    user_id: str        = Form(...),
    audio:   UploadFile = File(...),
):
    """
    Full pipeline: audio → STT → speaker ID → router → ambient analysis → TTS.
    Returns JSON with transcript, analysis, and base64 TTS audio.
    """
    import asyncpg, torchaudio
    from server.config import settings
    from server.pipeline.stt_processor import STTProcessor
    from server.pipeline.audio_segmenter import segment_audio
    from server.pipeline.speaker_embedder import extract_embedding
    from server.pipeline.voiceprint_matcher import identify_wearer
    from server.pipeline.session_router import route
    from server.ambient.cost_governor import CostGovernor
    from server.ambient.processor import AmbientProcessor
    from server.llm.factory import make_llm_client
    from server.tts.factory import make_tts_client

    audio_bytes = await audio.read()

    # Guard: browser sent empty blob (0s recording)
    if len(audio_bytes) < 1000:
        return JSONResponse({"transcript": "", "route": "SKIP", "reason": "recording too short"})

    # Save with .webm suffix — browser MediaRecorder produces WebM, not WAV
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        waveform, sr = torchaudio.load(tmp)
    finally:
        os.unlink(tmp)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    pcm = (waveform.squeeze().numpy() * 32767).astype("int16").tobytes()

    # ── STT ───────────────────────────────────────────────────────────────────
    logger.info("[PROCESS] STT starting user=%s", user_id)
    segments = await STTProcessor().transcribe(pcm)
    if not segments:
        return JSONResponse({"transcript": "", "route": "SKIP", "reason": "no speech"})

    full_transcript = " | ".join(f"{s.speaker_label}: {s.text}" for s in segments)
    logger.info("[PROCESS] transcript: %s", full_transcript[:120])

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
                logger.info("[PROCESS] wearer=%s sim=%.3f", wearer_label, top.cosine_sim)

    wearer_text = " ".join(
        s.text for s in segments if s.speaker_label == wearer_label
    ) if wearer_label else " ".join(s.text for s in segments)

    # ── Session router ────────────────────────────────────────────────────────
    governor = CostGovernor()
    governor.on_vad_start()
    decision = route(full_transcript, "AMBIENT", governor)
    logger.info("[PROCESS] route=%s", decision)

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
            analysis = await AmbientProcessor(
                client=make_llm_client("ambient")
            ).analyse(transcript=wearer_text, memories=[], capability_manifest="")
            result["analysis"] = analysis.model_dump()
            if analysis.should_act and analysis.consent_prompt:
                spoken = analysis.consent_prompt
                logger.info("[PROCESS] proactive suggestion: %r", spoken)
        except Exception as e:
            logger.warning("[PROCESS] ambient failed: %s", e)

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

            # Wrap in WAV for browser playback
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm_out)

            result["tts_audio"] = base64.b64encode(buf.getvalue()).decode()
            result["tts_text"]  = spoken
        except Exception as e:
            logger.warning("[PROCESS] TTS failed: %s", e)

    return JSONResponse(result)


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
