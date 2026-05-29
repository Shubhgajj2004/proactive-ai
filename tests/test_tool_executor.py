"""
Block 11: Tool Executor test.

Starts a mock MCP server in a subprocess, runs three scenarios,
then tears it down.

Scenarios:
  1. Read tool success      — succeeds on first attempt
  2. Read tool retry        — fails twice, succeeds on 3rd attempt
  3. Write tool success     — succeeds, idempotency key present in response
  4. Write tool failure     — fails once, NOT retried, error returned
  5. Read tool exhausted    — fails all 3 attempts, MCPCallError returned

Usage:
    python tests/test_tool_executor.py
"""
import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from server.tools.executor import call_tool, MCPCallError

PORT     = 8889   # use a non-standard port to avoid conflicts
BASE_URL = f"http://127.0.0.1:{PORT}"


# ── Server lifecycle ──────────────────────────────────────────────────────────

def start_server(fail: int = 0) -> subprocess.Popen:
    """Start mock MCP server as a subprocess."""
    proc = subprocess.Popen(
        [sys.executable, "tests/fixtures/mock_mcp_server.py",
         "--port", str(PORT), "--fail", str(fail)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Wait until it's accepting connections
    for _ in range(20):
        time.sleep(0.2)
        try:
            httpx.get(f"{BASE_URL}/health", timeout=1.0)
            return proc
        except Exception:
            pass
    raise RuntimeError("Mock MCP server did not start in time")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    proc.wait(timeout=5)


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_read_success(results: list) -> None:
    print("\n[1] Read tool — success on first attempt")
    proc = start_server(fail=0)
    try:
        result = await call_tool(
            tool_name="weather_fetch", call_type="read",
            arguments={"location": "Mumbai"},
            endpoint=f"{BASE_URL}/tools/weather_fetch",
        )
        ok = isinstance(result, dict) and result.get("status") == "success"
        print(f"   Result: {result}")
        print(f"   {'✓ PASS' if ok else '✗ FAIL'}")
        results.append(("Read success", ok))
    finally:
        stop_server(proc)


async def test_read_retry(results: list) -> None:
    print("\n[2] Read tool — fails twice, succeeds on 3rd attempt")
    proc = start_server(fail=2)
    try:
        result = await call_tool(
            tool_name="weather_fetch", call_type="read",
            arguments={"location": "Mumbai"},
            endpoint=f"{BASE_URL}/tools/weather_fetch",
        )
        ok = isinstance(result, dict) and result.get("status") == "success"
        print(f"   Result: {result}")
        print(f"   {'✓ PASS' if ok else '✗ FAIL'} — recovered after 2 failures")
        results.append(("Read retry recovery", ok))
    finally:
        stop_server(proc)


async def test_write_success(results: list) -> None:
    print("\n[3] Write tool — success with idempotency key")
    proc = start_server(fail=0)
    try:
        result = await call_tool(
            tool_name="book_restaurant", call_type="write",
            arguments={"restaurant": "Ristorante Roma", "party_size": 2},
            endpoint=f"{BASE_URL}/tools/book_restaurant",
            session_id="sess-abc", step_count=1,
        )
        expected_key = "sess-abc:1:book_restaurant"
        ok = (
            isinstance(result, dict)
            and result.get("status") == "success"
            and result.get("idempotency_key") == expected_key
        )
        print(f"   Result: {result}")
        print(f"   Idempotency key: {result.get('idempotency_key')!r}")
        print(f"   {'✓ PASS' if ok else '✗ FAIL'}")
        results.append(("Write success + idempotency", ok))
    finally:
        stop_server(proc)


async def test_write_no_retry(results: list) -> None:
    print("\n[4] Write tool — fails once, NOT retried, MCPCallError returned")
    proc = start_server(fail=5)   # server will always fail during this test
    try:
        result = await call_tool(
            tool_name="book_cab", call_type="write",
            arguments={"destination": "airport"},
            endpoint=f"{BASE_URL}/tools/book_cab",
            session_id="sess-abc", step_count=2,
        )
        ok = (
            isinstance(result, MCPCallError)
            and result.attempt == 1          # only 1 attempt for write
            and result.tool_name == "book_cab"
        )
        print(f"   Result: {result}")
        print(f"   {'✓ PASS' if ok else '✗ FAIL'} — exactly 1 attempt, no retry")
        results.append(("Write no retry", ok))
    finally:
        stop_server(proc)


async def test_read_exhausted(results: list) -> None:
    print("\n[5] Read tool — all 3 attempts fail, MCPCallError returned")
    proc = start_server(fail=10)  # server will always fail during this test
    try:
        result = await call_tool(
            tool_name="weather_fetch", call_type="read",
            arguments={"location": "Mumbai"},
            endpoint=f"{BASE_URL}/tools/weather_fetch",
        )
        ok = (
            isinstance(result, MCPCallError)
            and result.attempt == 3          # exhausted all 3 attempts
            and result.tool_name == "weather_fetch"
        )
        print(f"   Result: {result}")
        print(f"   {'✓ PASS' if ok else '✗ FAIL'} — MCPCallError after 3 attempts")
        results.append(("Read exhausted", ok))
    finally:
        stop_server(proc)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    print(f"\n{'='*60}")
    print("  Block 11: Tool Executor")
    print(f"{'='*60}")

    results: list[tuple[str, bool]] = []
    await test_read_success(results)
    await test_read_retry(results)
    await test_write_success(results)
    await test_write_no_retry(results)
    await test_read_exhausted(results)

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
