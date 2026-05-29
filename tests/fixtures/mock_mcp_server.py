"""
Mock MCP server for Block 11 testing.

Responds to any POST /tools/{tool_name} request.
Supports controlled failures via query params:
  ?fail=N  → fail the first N requests then succeed

Usage:
    python tests/fixtures/mock_mcp_server.py --port 8888
    python tests/fixtures/mock_mcp_server.py --port 8888 --fail 2
"""
import argparse
import sys
from pathlib import Path

# Must import uvicorn and fastapi — install if missing
try:
    import uvicorn
    from fastapi import FastAPI, Request, Response
except ImportError:
    print("Install fastapi + uvicorn: uv pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI()

_fail_remaining = 0   # controlled from startup


@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    global _fail_remaining

    body = await request.json()
    idempotency_key = request.headers.get("X-Idempotency-Key", "")

    if _fail_remaining > 0:
        _fail_remaining -= 1
        print(f"[MOCK] {tool_name} → FAIL (remaining failures: {_fail_remaining})")
        return Response(
            content=f'{{"error": "simulated failure for {tool_name}"}}',
            status_code=500,
            media_type="application/json",
        )

    result = {
        "status": "success",
        "tool":   tool_name,
        "result": f"mock result for {tool_name}",
        "arguments_received": body.get("arguments", {}),
        "idempotency_key": idempotency_key,
    }
    print(f"[MOCK] {tool_name} → SUCCESS  key={idempotency_key or 'none'}")
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock MCP server")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--fail", type=int, default=0,
                        help="Fail the first N requests then succeed")
    args = parser.parse_args()

    _fail_remaining = args.fail
    print(f"Mock MCP server starting on port {args.port}")
    if args.fail:
        print(f"Will fail first {args.fail} request(s)")

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
