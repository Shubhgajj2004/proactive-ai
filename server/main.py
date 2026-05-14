"""
FastAPI entrypoint.
Handles: Daily room lifecycle, session creation, enrollment, inactivity timer.
"""
import asyncio
import logging
import time
import uuid

import httpx
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from server.cleanup import cleanup_job
from server.config import settings
from server.db.postgres import close_db, init_db
from server.db.redis import close_redis

logger = logging.getLogger(__name__)

app = FastAPI(title="Proactive AI")

DAILY_API_BASE = "https://api.daily.co/v1"


# ── Lifespan ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    await init_db()
    asyncio.create_task(cleanup_job())
    logger.info("Server started")


@app.on_event("shutdown")
async def shutdown():
    await close_db()
    await close_redis()


# ── Daily REST helpers ────────────────────────────────────────────────────────

async def daily_create_room(session_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{DAILY_API_BASE}/rooms",
            headers={"Authorization": f"Bearer {settings.DAILY_API_KEY}"},
            json={"name": session_id, "properties": {"exp": int(time.time()) + 7200}},
        )
        r.raise_for_status()
        return r.json()


async def daily_create_token(room_name: str, user_id: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{DAILY_API_BASE}/meeting-tokens",
            headers={"Authorization": f"Bearer {settings.DAILY_API_KEY}"},
            json={"properties": {"room_name": room_name, "user_id": user_id}},
        )
        r.raise_for_status()
        return r.json()["token"]


async def daily_delete_room(room_name: str) -> None:
    async with httpx.AsyncClient() as client:
        await client.delete(
            f"{DAILY_API_BASE}/rooms/{room_name}",
            headers={"Authorization": f"Bearer {settings.DAILY_API_KEY}"},
        )


# ── Session lifecycle ─────────────────────────────────────────────────────────

@app.post("/api/session/create")
async def create_session(user_id: str):
    from server.db import postgres as db

    session_id = str(uuid.uuid4())
    room = await daily_create_room(session_id)
    token = await daily_create_token(room["name"], user_id)

    await db.execute(
        "INSERT INTO sessions (session_id, user_id, state) VALUES ($1, $2, 'AMBIENT')",
        session_id, user_id,
    )
    asyncio.create_task(inactivity_timer(session_id, timeout_seconds=300))

    return {"room_url": room["url"], "token": token, "session_id": session_id}


async def inactivity_timer(session_id: str, timeout_seconds: int) -> None:
    from server.db import postgres as db

    while True:
        await asyncio.sleep(60)
        row = await db.fetchrow(
            "SELECT last_activity_at FROM sessions WHERE session_id = $1", session_id
        )
        if not row:
            return  # session already cleaned up
        idle = time.time() - row["last_activity_at"].timestamp()
        if idle > timeout_seconds:
            await teardown_session(session_id)
            return


async def teardown_session(session_id: str) -> None:
    from server.action.session_manager import terminate_session

    await terminate_session(session_id, reason="inactivity")
    await daily_delete_room(session_id)


# ── Enrollment ────────────────────────────────────────────────────────────────

@app.post("/api/enroll")
async def enroll_user(user_id: str, audio: UploadFile):
    """One-time enrollment: record 10s phrase → extract d-vector → store."""
    import asyncio

    import torchaudio

    from server.db import postgres as db

    audio_bytes = await audio.read()

    # Lazy import to avoid loading torch at startup
    from server.pipeline.speaker_embedder import extract_d_vector

    d_vector = await asyncio.get_event_loop().run_in_executor(
        None, extract_d_vector, audio_bytes
    )

    await db.execute(
        "INSERT INTO user_voiceprints (user_id, d_vector) VALUES ($1, $2) "
        "ON CONFLICT (user_id) DO UPDATE SET d_vector = $2, enrolled_at = now()",
        user_id, d_vector.tolist(),
    )
    return {"status": "enrolled", "user_id": user_id, "d_vector_dims": len(d_vector)}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}
