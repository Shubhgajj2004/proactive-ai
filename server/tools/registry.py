"""
Tool Registry — embeds MCP tool descriptions and upserts into mcp_tools table.

Each tool stored with:
  - name, description, schema (JSONB), call_type (read/write), domain
  - embedding (vector(768)) — text-embedding-004 of "name: description"

Called at server startup and whenever tools change.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any

import asyncpg

from server.config import settings
from server.embeddings.factory import make_embedding_client

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name:        str
    description: str
    schema:      dict[str, Any]
    call_type:   str            # "read" | "write"
    domain:      str = ""


async def register_tools(tools: list[ToolDefinition]) -> None:
    """
    Embed each tool description and upsert into mcp_tools.
    Safe to run multiple times — conflicts update in place.
    """
    embedder = make_embedding_client()
    conn     = await asyncpg.connect(settings.DATABASE_URL)

    try:
        for tool in tools:
            # Embed "name: description" so semantic search captures both
            embed_text = f"{tool.name}: {tool.description}"
            embedding  = await embedder.embed(embed_text)

            await conn.execute(
                """
                INSERT INTO mcp_tools (name, description, schema, call_type, domain, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                ON CONFLICT (name) DO UPDATE
                    SET description  = EXCLUDED.description,
                        schema       = EXCLUDED.schema,
                        call_type    = EXCLUDED.call_type,
                        domain       = EXCLUDED.domain,
                        embedding    = EXCLUDED.embedding,
                        schema_version = mcp_tools.schema_version + 1
                """,
                tool.name,
                tool.description,
                json.dumps(tool.schema),
                tool.call_type,
                tool.domain,
                "[" + ",".join(f"{x:.6f}" for x in embedding) + "]",
            )
            logger.info("[REGISTRY] registered %s (%s)", tool.name, tool.call_type)

    finally:
        await conn.close()

    logger.info("[REGISTRY] %d tool(s) registered", len(tools))
