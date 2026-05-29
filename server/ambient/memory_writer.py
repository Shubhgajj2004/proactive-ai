"""
Memory Writer — persists wearer facts via mem0 (self-hosted).

Fully self-hosted — no mem0 cloud. All three components run locally:
  - Vector store : our own PostgreSQL + pgvector  (mem0_memories table)
  - LLM          : OpenRouter (poolside/laguna-xs.2:free)
  - Embedder     : Google text-embedding-004 via Gemini SDK

Only is_wearer=True turns are ever written. Bystander speech is filtered
upstream before this module is called.

mem0's Memory API is synchronous — all calls are wrapped in
run_in_executor so they never block the asyncio event loop.
"""
import asyncio
import logging
import os
from functools import lru_cache
from typing import Any

from server.ambient.processor import MemoryOp
from server.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_memory():
    """
    Build and cache the mem0 Memory instance.
    Called once on first use — subsequent calls return the cached instance.
    """
    from mem0 import Memory

    # Gemini embedder reads GOOGLE_API_KEY — set it from our config
    os.environ.setdefault("GOOGLE_API_KEY", settings.GEMINI_API_KEY)

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model":           settings.LLM_AMBIENT_MODEL,
                "api_key":         settings.OPENROUTER_API_KEY,
                "openai_base_url": settings.LLM_BASE_URL,
                "temperature":     0.1,
                "max_tokens":      2000,
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                # gemini-embedding-001 is the mem0-compatible model (v1beta endpoint)
                # text-embedding-004 uses v1 endpoint which mem0's Gemini client doesn't support
                "model":           "models/gemini-embedding-001",
                "api_key":         settings.GEMINI_API_KEY,
                "embedding_dims":  768,
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "connection_string":    settings.DATABASE_URL,
                "collection_name":      "mem0_memories",
                "embedding_model_dims": 768,
            },
        },
    }

    logger.info("[MEMORY] initialising mem0 (self-hosted pgvector)")
    memory = Memory.from_config(config)
    logger.info("[MEMORY] mem0 ready")
    return memory


# ── Public API ────────────────────────────────────────────────────────────────

async def apply_memory_ops(
    ops: list[MemoryOp],
    user_id: str,
) -> None:
    """
    Apply a list of memory operations for the given user.
    Runs mem0 calls on the thread pool — never blocks the event loop.

    Args:
        ops:     list of MemoryOp (add / update / delete) from AmbientAnalysis
        user_id: wearer's user_id
    """
    if not ops:
        return

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _apply_sync, ops, user_id)


def _apply_sync(ops: list[MemoryOp], user_id: str) -> None:
    """Synchronous mem0 operations — runs on thread pool."""
    memory = _get_memory()

    for op in ops:
        try:
            if op.op == "add":
                result = memory.add(op.fact, user_id=user_id)
                logger.info("[MEMORY] ADD user=%s fact=%r → %s", user_id, op.fact[:60], result)

            elif op.op == "update":
                if not op.memory_id:
                    logger.warning("[MEMORY] UPDATE skipped — no memory_id provided")
                    continue
                result = memory.update(op.memory_id, op.fact)
                logger.info("[MEMORY] UPDATE id=%s fact=%r → %s", op.memory_id, op.fact[:60], result)

            elif op.op == "delete":
                if not op.memory_id:
                    logger.warning("[MEMORY] DELETE skipped — no memory_id provided")
                    continue
                memory.delete(op.memory_id)
                logger.info("[MEMORY] DELETE id=%s", op.memory_id)

        except Exception as e:
            logger.error("[MEMORY] op=%s failed: %s", op.op, e, exc_info=True)


async def search_memories(
    query: str,
    user_id: str,
    limit: int = 5,
) -> list[str]:
    """
    Search wearer memories relevant to a query.
    Returns a list of fact strings (for injection into ambient prompt).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _search_sync, query, user_id, limit)


def _search_sync(query: str, user_id: str, limit: int) -> list[str]:
    memory = _get_memory()
    try:
        results = memory.search(query, filters={"user_id": user_id}, top_k=limit)
        # mem0 returns list of dicts with 'memory' key
        facts = [r["memory"] for r in results.get("results", []) if "memory" in r]
        logger.info("[MEMORY] SEARCH user=%s query=%r → %d results", user_id, query[:60], len(facts))
        return facts
    except Exception as e:
        logger.error("[MEMORY] search failed: %s", e, exc_info=True)
        return []


async def get_all_memories(user_id: str) -> list[dict[str, Any]]:
    """Return all stored memories for a user (for debugging/testing)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_all_sync, user_id)


def _get_all_sync(user_id: str) -> list[dict[str, Any]]:
    memory = _get_memory()
    try:
        results = memory.get_all(filters={"user_id": user_id})
        return results.get("results", [])
    except Exception as e:
        logger.error("[MEMORY] get_all failed: %s", e, exc_info=True)
        return []
