"""
Tool Executor — calls MCP tools over HTTP.

Two execution policies:
  read  tools: 3 attempts, exponential backoff (2s, 4s, 8s)
  write tools: 1 attempt only + X-Idempotency-Key header
               on failure → return error context, let Plan node decide

Write tools are never retried to prevent double-execution of
side-effectful operations (booking, sending messages, etc.).
The idempotency key makes it safe to retry at the HTTP layer
if the server supports it, but we don't retry at the client layer.

Idempotency key format: {session_id}:{step_count}:{tool_name}
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Retry config for read tools
READ_MAX_ATTEMPTS = 3
READ_BASE_BACKOFF = 2.0    # seconds; doubles each attempt

# HTTP timeout
TIMEOUT_SECONDS = 30.0


@dataclass
class MCPCallError:
    tool_name:  str
    attempt:    int
    status:     int | None     # HTTP status, None if connection error
    message:    str

    def to_dict(self) -> dict:
        return {
            "type":      "MCPCallError",
            "tool_name": self.tool_name,
            "attempt":   self.attempt,
            "status":    self.status,
            "message":   self.message,
        }


async def call_tool(
    tool_name:   str,
    call_type:   str,               # "read" | "write"
    arguments:   dict[str, Any],
    endpoint:    str,               # MCP server URL e.g. "http://localhost:8888/tools/book_restaurant"
    session_id:  str = "",
    step_count:  int = 0,
) -> dict[str, Any] | MCPCallError:
    """
    Call an MCP tool over HTTP POST.

    Returns the parsed JSON response on success, or MCPCallError on failure.
    Never raises — callers receive the error object and pass it to the Plan node.
    """
    if call_type == "write":
        return await _call_write(tool_name, arguments, endpoint, session_id, step_count)
    else:
        return await _call_read(tool_name, arguments, endpoint)


# ── Read — 3 retries with exponential backoff ─────────────────────────────────

async def _call_read(
    tool_name: str,
    arguments: dict[str, Any],
    endpoint:  str,
) -> dict[str, Any] | MCPCallError:
    backoff = READ_BASE_BACKOFF

    for attempt in range(1, READ_MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                response = await client.post(endpoint, json={"arguments": arguments})

            if response.status_code == 200:
                logger.info(
                    "[EXECUTOR] %s (read) attempt=%d → SUCCESS",
                    tool_name, attempt,
                )
                return response.json()

            logger.warning(
                "[EXECUTOR] %s (read) attempt=%d → HTTP %d",
                tool_name, attempt, response.status_code,
            )
            error = MCPCallError(
                tool_name=tool_name, attempt=attempt,
                status=response.status_code, message=response.text[:200],
            )

        except httpx.RequestError as exc:
            logger.warning(
                "[EXECUTOR] %s (read) attempt=%d → connection error: %s",
                tool_name, attempt, exc,
            )
            error = MCPCallError(
                tool_name=tool_name, attempt=attempt,
                status=None, message=str(exc),
            )

        if attempt < READ_MAX_ATTEMPTS:
            logger.info("[EXECUTOR] retrying in %.1fs…", backoff)
            await asyncio.sleep(backoff)
            backoff *= 2

    logger.error("[EXECUTOR] %s (read) failed after %d attempts", tool_name, READ_MAX_ATTEMPTS)
    return error


# ── Write — 1 attempt only ────────────────────────────────────────────────────

async def _call_write(
    tool_name:  str,
    arguments:  dict[str, Any],
    endpoint:   str,
    session_id: str,
    step_count: int,
) -> dict[str, Any] | MCPCallError:
    idempotency_key = f"{session_id}:{step_count}:{tool_name}"
    headers = {"X-Idempotency-Key": idempotency_key}

    logger.info(
        "[EXECUTOR] %s (write) idempotency_key=%s",
        tool_name, idempotency_key,
    )

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                endpoint, json={"arguments": arguments}, headers=headers,
            )

        if response.status_code == 200:
            logger.info("[EXECUTOR] %s (write) → SUCCESS", tool_name)
            return response.json()

        logger.error(
            "[EXECUTOR] %s (write) → HTTP %d (no retry for write tools)",
            tool_name, response.status_code,
        )
        return MCPCallError(
            tool_name=tool_name, attempt=1,
            status=response.status_code, message=response.text[:200],
        )

    except httpx.RequestError as exc:
        logger.error(
            "[EXECUTOR] %s (write) → connection error (no retry): %s",
            tool_name, exc,
        )
        return MCPCallError(
            tool_name=tool_name, attempt=1,
            status=None, message=str(exc),
        )
