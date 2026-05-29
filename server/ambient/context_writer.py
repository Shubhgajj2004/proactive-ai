"""
Context Writer — persists conversation context to PostgreSQL.

Two writes per utterance (both async, run in parallel):
  1. context_summaries  — summary, extracted_facts, tags, speaker_labels
                          weekly-partitioned table; drives long-term insights
  2. raw_transcripts    — verbatim transcript text
                          deleted by cleanup_job after 24h

Both are fire-and-forget from the pipeline — failures are logged but
never propagate to the caller.
"""
import json
import logging
from dataclasses import dataclass

import asyncpg

from server.ambient.processor import AmbientAnalysis
from server.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ContextWriteResult:
    summary_id:    str | None = None
    transcript_id: str | None = None


async def write_context(
    analysis:        AmbientAnalysis,
    raw_transcript:  str,
    user_id:         str,
    session_id:      str,
    speaker_labels:  list[str] | None = None,
) -> ContextWriteResult:
    """
    Write summary + raw transcript to PostgreSQL in a single connection.

    Args:
        analysis:       AmbientAnalysis from the ambient processor
        raw_transcript: Full verbatim transcript for this utterance
        user_id:        Wearer's user_id
        session_id:     Current session_id
        speaker_labels: STT speaker labels found in this utterance

    Returns:
        ContextWriteResult with inserted row IDs (None on failure)
    """
    result = ContextWriteResult()

    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            result = await _write(conn, analysis, raw_transcript, user_id, session_id, speaker_labels)
        finally:
            await conn.close()

    except Exception as e:
        logger.error("[CONTEXT] write failed: %s", e, exc_info=True)

    return result


async def _write(
    conn:            asyncpg.Connection,
    analysis:        AmbientAnalysis,
    raw_transcript:  str,
    user_id:         str,
    session_id:      str,
    speaker_labels:  list[str] | None,
) -> ContextWriteResult:
    result = ContextWriteResult()

    # ── context_summaries ─────────────────────────────────────────────────────
    summary_row = await conn.fetchrow(
        """
        INSERT INTO context_summaries
            (user_id, session_id, summary, extracted_facts, tags, speaker_labels)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id::text
        """,
        user_id,
        session_id,
        analysis.summary,
        json.dumps(analysis.extracted_facts),
        json.dumps(analysis.tags),
        json.dumps(speaker_labels or []),
    )
    result.summary_id = summary_row["id"]
    logger.info(
        "[CONTEXT] summary written id=%s user=%s session=%s",
        result.summary_id, user_id, session_id,
    )

    # ── raw_transcripts ───────────────────────────────────────────────────────
    transcript_row = await conn.fetchrow(
        """
        INSERT INTO raw_transcripts (user_id, session_id, raw_transcript)
        VALUES ($1, $2, $3)
        RETURNING id::text
        """,
        user_id,
        session_id,
        raw_transcript,
    )
    result.transcript_id = transcript_row["id"]
    logger.info(
        "[CONTEXT] transcript written id=%s user=%s",
        result.transcript_id, user_id,
    )

    return result


async def get_recent_summaries(
    user_id:    str,
    limit:      int = 10,
) -> list[dict]:
    """Fetch recent context summaries for a user (for debugging/testing)."""
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            rows = await conn.fetch(
                """
                SELECT id::text, session_id, created_at, summary, extracted_facts, tags
                FROM context_summaries
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id, limit,
            )
            return [dict(r) for r in rows]
        finally:
            await conn.close()
    except Exception as e:
        logger.error("[CONTEXT] get_recent_summaries failed: %s", e)
        return []
