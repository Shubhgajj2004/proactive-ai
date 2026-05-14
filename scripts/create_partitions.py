"""
Create weekly partitions for context_summaries (4 weeks ahead by default).
Run at server startup and every 6h in cleanup_job.

Usage:
    python scripts/create_partitions.py
"""
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg

from server.config import settings


def _monday(d: date) -> date:
    """Return the Monday of the week containing d."""
    return d - timedelta(days=d.weekday())


async def ensure_weekly_partitions(conn: asyncpg.Connection, weeks_ahead: int = 4) -> None:
    today = date.today()
    for i in range(weeks_ahead):
        week_start = _monday(today + timedelta(weeks=i))
        week_end = week_start + timedelta(weeks=1)
        partition_name = f"context_summaries_{week_start.strftime('%Yw%W')}"

        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {partition_name}
            PARTITION OF context_summaries
            FOR VALUES FROM ('{week_start}') TO ('{week_end}')
        """)
        print(f"✓ Partition: {partition_name}  ({week_start} → {week_end})")


async def run():
    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        await ensure_weekly_partitions(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run())
