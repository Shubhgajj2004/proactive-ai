"""
Capability Manifest — one-liner tool descriptions for the ambient + action prompts.

Injected into every LLM call so the model knows what tools exist.
Cached in Redis (5-min TTL) and regenerated on cache miss or tool change.
"""
import logging

import asyncpg

from server.config import settings
from server.prompts.tools import CAPABILITY_MANIFEST_HEADER, NO_TOOLS_AVAILABLE

logger = logging.getLogger(__name__)

CACHE_TTL = 300   # 5 minutes


async def get_manifest(redis_client=None) -> str:
    """
    Return a formatted capability manifest string.
    Tries Redis cache first, falls back to PostgreSQL.

    Args:
        redis_client: Optional async Redis client. If None, skips cache.

    Returns:
        Multi-line string listing all tools — injected into LLM system prompt.
    """
    cache_key = "proactive:tool_manifest"

    # ── Cache read ────────────────────────────────────────────────────────────
    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug("[MANIFEST] cache hit")
                return cached.decode() if isinstance(cached, bytes) else cached
        except Exception as e:
            logger.warning("[MANIFEST] redis read failed: %s", e)

    # ── Build from DB ─────────────────────────────────────────────────────────
    manifest = await _build_from_db()

    # ── Cache write ───────────────────────────────────────────────────────────
    if redis_client and manifest:
        try:
            await redis_client.setex(cache_key, CACHE_TTL, manifest)
            logger.debug("[MANIFEST] cached for %ds", CACHE_TTL)
        except Exception as e:
            logger.warning("[MANIFEST] redis write failed: %s", e)

    return manifest


async def invalidate_manifest(redis_client) -> None:
    """Call after registering new tools to force cache refresh."""
    try:
        await redis_client.delete("proactive:tool_manifest")
        logger.info("[MANIFEST] cache invalidated")
    except Exception as e:
        logger.warning("[MANIFEST] invalidate failed: %s", e)


async def _build_from_db() -> str:
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            rows = await conn.fetch(
                "SELECT name, description FROM mcp_tools ORDER BY name"
            )
        finally:
            await conn.close()
    except Exception as e:
        logger.error("[MANIFEST] DB read failed: %s", e)
        return NO_TOOLS_AVAILABLE

    if not rows:
        return NO_TOOLS_AVAILABLE

    lines = [CAPABILITY_MANIFEST_HEADER]
    for row in rows:
        lines.append(f"- {row['name']}: {row['description']}")

    manifest = "\n".join(lines)
    logger.info("[MANIFEST] built from DB — %d tools", len(rows))
    return manifest
