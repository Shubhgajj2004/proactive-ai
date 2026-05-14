"""asyncpg connection pool. Call init_db() at FastAPI startup."""
import asyncpg

from server.config import settings

_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        settings.DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised — call init_db() first")
    return _pool


async def execute(query: str, *args) -> str:
    return await get_pool().execute(query, *args)


async def fetch(query: str, *args) -> list[asyncpg.Record]:
    return await get_pool().fetch(query, *args)


async def fetchrow(query: str, *args) -> asyncpg.Record | None:
    return await get_pool().fetchrow(query, *args)


async def fetchval(query: str, *args):
    return await get_pool().fetchval(query, *args)
