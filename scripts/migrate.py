"""
Run all DB migrations. Safe to run multiple times (idempotent).

Usage:
    python scripts/migrate.py
"""
import asyncio
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg

from server.config import settings

SCHEMA = """
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Wearer identity ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_voiceprints (
    user_id     TEXT PRIMARY KEY,
    d_vector    vector(256) NOT NULL,
    enrolled_at TIMESTAMPTZ DEFAULT now()
);

-- ── Tool registry (768-dim text-embedding-004) ───────────────────────────────
CREATE TABLE IF NOT EXISTS mcp_tools (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           TEXT UNIQUE NOT NULL,
    description    TEXT NOT NULL,
    schema         JSONB NOT NULL,
    schema_version INT DEFAULT 1,
    call_type      TEXT NOT NULL CHECK (call_type IN ('read', 'write')),
    domain         TEXT,
    embedding      vector(768),
    created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS mcp_tools_embedding_idx
    ON mcp_tools USING hnsw (embedding vector_cosine_ops);

-- ── Context summaries (weekly partitioned) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS context_summaries (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary         TEXT,
    extracted_facts JSONB,
    tags            JSONB,
    speaker_labels  JSONB
) PARTITION BY RANGE (created_at);
CREATE INDEX IF NOT EXISTS context_summaries_user_created_idx
    ON context_summaries (user_id, created_at DESC);

-- ── Raw transcripts (separate table; no FK to partitioned table) ─────────────
CREATE TABLE IF NOT EXISTS raw_transcripts (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    raw_transcript TEXT
);
CREATE INDEX IF NOT EXISTS raw_transcripts_created_idx
    ON raw_transcripts (created_at);
CREATE INDEX IF NOT EXISTS raw_transcripts_user_created_idx
    ON raw_transcripts (user_id, created_at);

-- ── Session state (source of truth) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id               TEXT PRIMARY KEY,
    user_id                  TEXT NOT NULL,
    state                    TEXT NOT NULL CHECK (state IN ('AMBIENT', 'ACTIVE')),
    trigger_source           TEXT CHECK (trigger_source IN ('proactive_confidence', 'wake_word')),
    model_tier               TEXT CHECK (model_tier IN ('standard', 'premium')),
    langgraph_thread_id      TEXT,
    pending_memory_ops       JSONB DEFAULT '[]',
    triggering_ambient_log_id UUID,
    initial_proposed_action  TEXT,
    initial_consent_prompt   TEXT,
    initial_reasoning        TEXT,
    turn_count               INT DEFAULT 0,
    last_activity_at         TIMESTAMPTZ DEFAULT now(),
    created_at               TIMESTAMPTZ DEFAULT now()
);

-- ── Ambient confidence calibration log (90-day retention) ───────────────────
CREATE TABLE IF NOT EXISTS ambient_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT,
    session_id      TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    confidence      FLOAT,
    should_act      BOOL,
    proposed_action TEXT,
    session_spawned BOOL,
    user_outcome    TEXT CHECK (user_outcome IN ('confirmed', 'declined', 'no_feedback', 'timeout'))
);

-- ── Cost governance ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_settings (
    user_id            TEXT PRIMARY KEY,
    daily_token_budget INT DEFAULT 1000000,
    ambient_enabled    BOOL DEFAULT true
);

-- ── Monthly rollup (permanent wearer insights) ───────────────────────────────
CREATE TABLE IF NOT EXISTS user_insights (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      TEXT NOT NULL,
    month        DATE NOT NULL,
    insight      TEXT NOT NULL,
    source_count INT,
    created_at   TIMESTAMPTZ DEFAULT now(),
    UNIQUE (user_id, month, insight)
);

-- ── mem0 memory store (managed by mem0ai; 768-dim) ────────────────────────────
-- mem0 creates its own tables automatically when Memory() is initialised.
-- We use table_name="mem0_memories" in config to avoid collision.
"""


async def run():
    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        await conn.execute(SCHEMA)
        print("✓ Migrations complete — all tables exist")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run())
