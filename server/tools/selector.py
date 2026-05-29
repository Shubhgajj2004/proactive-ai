"""
Tool Selector — retrieves the top-K most relevant tools for a given step.

Uses pgvector cosine similarity search against the mcp_tools table.
Called at each Execute step in the action agent with a fresh embed of
the current plan step — ensures the right schema is retrieved every time.
"""
import logging
from dataclasses import dataclass
from typing import Any

import asyncpg

from server.config import settings
from server.embeddings.factory import make_embedding_client

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 2


@dataclass
class SelectedTool:
    name:        str
    description: str
    schema:      dict[str, Any]
    call_type:   str
    domain:      str
    similarity:  float


async def select_tools(query: str, top_k: int = DEFAULT_TOP_K) -> list[SelectedTool]:
    """
    Embed query and return top-K most similar tools by cosine similarity.

    Args:
        query:  Natural language description of the current step
                e.g. "book a table at Ristorante Roma for 2 people"
        top_k:  Number of tools to return (default 2)

    Returns:
        list of SelectedTool sorted by similarity descending
    """
    import json

    embedder  = make_embedding_client()
    embedding = await embedder.embed(query)
    vec_str   = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT
                name,
                description,
                schema,
                call_type,
                domain,
                1 - (embedding <=> $1::vector) AS similarity
            FROM mcp_tools
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vec_str,
            top_k,
        )
    finally:
        await conn.close()

    results = []
    for row in rows:
        results.append(SelectedTool(
            name=row["name"],
            description=row["description"],
            schema=json.loads(row["schema"]),
            call_type=row["call_type"],
            domain=row["domain"] or "",
            similarity=float(row["similarity"]),
        ))
        logger.info(
            "[SELECTOR] %s  sim=%.4f  (%s)",
            row["name"], float(row["similarity"]), row["call_type"],
        )

    return results
