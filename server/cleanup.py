"""
Asyncio background retention job. Runs every 6 hours at startup.
No pg_cron dependency.
"""
import asyncio
import logging
from datetime import date, timedelta

from scripts.create_partitions import ensure_weekly_partitions
from server.db import postgres as db

logger = logging.getLogger(__name__)


async def cleanup_job() -> None:
    """Runs indefinitely. Call as asyncio.create_task(cleanup_job())."""
    while True:
        try:
            await _run_once()
        except Exception as exc:
            logger.exception("cleanup_job error: %s", exc)
        await asyncio.sleep(6 * 3600)  # 6 hours


async def _run_once() -> None:
    logger.info("cleanup_job: starting")

    # Delete raw transcripts older than 24h
    deleted = await db.execute(
        "DELETE FROM raw_transcripts WHERE created_at < now() - interval '24 hours'"
    )
    logger.info("cleanup_job: deleted raw_transcripts %s", deleted)

    # Delete ambient logs older than 90 days
    deleted = await db.execute(
        "DELETE FROM ambient_logs WHERE created_at < now() - interval '90 days'"
    )
    logger.info("cleanup_job: deleted ambient_logs %s", deleted)

    # Ensure next 4 weekly partitions exist
    import asyncpg
    from server.config import settings
    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        await ensure_weekly_partitions(conn, weeks_ahead=4)
    finally:
        await conn.close()

    logger.info("cleanup_job: done")
