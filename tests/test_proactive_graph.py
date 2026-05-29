"""
Block 12: Proactive LangGraph test.

Three scenarios with mock LLM + mock MCP server:
  1. decline  — user says "no thanks" → DONE, zero tool calls
  2. approve  — user says "yes" → Plan → Tool → Respond → DONE
  3. multi-step — 2 tool steps then done

Usage:
    python tests/test_proactive_graph.py
    python tests/test_proactive_graph.py --scenario decline
    python tests/test_proactive_graph.py --scenario approve
    python tests/test_proactive_graph.py --scenario multi
"""
import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from server.action.proactive_graph import run_proactive_session
from server.llm.client import LLMClient, LLMResponse

PORT     = 8890
MCP_BASE = f"http://127.0.0.1:{PORT}/tools"


# ── Mock LLM ──────────────────────────────────────────────────────────────────

class _ScriptedLLM(LLMClient):
    """Returns pre-written responses in sequence."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    async def complete(self, messages, response_format=None, temperature=0.3) -> LLMResponse:
        if self._idx < len(self._responses):
            content = self._responses[self._idx]
            self._idx += 1
        else:
            # Default: mark done
            content = json.dumps({
                "next_step": "", "need_clarification": False,
                "question": "", "done": True,
                "final_response": "Done!",
            })
        return LLMResponse(content=content, usage_input_tokens=50, usage_output_tokens=30)

    @property
    def last_usage_tokens(self) -> int:
        return 80


# ── Server helpers ────────────────────────────────────────────────────────────

def start_mock_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "tests/fixtures/mock_mcp_server.py", "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for _ in range(20):
        time.sleep(0.2)
        try:
            httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=1.0)
            return proc
        except Exception:
            pass
    raise RuntimeError("Mock MCP server did not start")


def stop_mock_server(proc: subprocess.Popen):
    proc.terminate()
    proc.wait(timeout=5)


# ── Scenarios ─────────────────────────────────────────────────────────────────

BASE_STATE = {
    "session_id":       "test-proactive-001",
    "user_id":          "test_user",
    "task_description": "book a table at Ristorante Roma for tonight",
    "consent_prompt":   "Want me to book a table at Ristorante Roma for tonight?",
    "proposed_action":  "book_restaurant",
}


async def scenario_decline() -> bool:
    print("\n[decline] User says 'no thanks' → zero tool calls, outcome=declined")

    llm = _ScriptedLLM([])   # LLM never called if user declines
    final = await run_proactive_session(
        initial_state=BASE_STATE,
        user_reply="no thanks",
        llm=llm,
        mcp_base=MCP_BASE,
    )

    ok = (
        final.get("outcome") == "declined"
        and len(final.get("tool_results", [])) == 0
        and final.get("done") is True
    )
    print(f"   outcome={final.get('outcome')}  tools_used={len(final.get('tool_results',[]))}")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


async def scenario_approve_single(proc) -> bool:
    print("\n[approve] User says 'yes please' → 1 tool call → respond → done")

    # Plan: one step then done
    llm = _ScriptedLLM([
        # Turn 1: plan says next step
        json.dumps({
            "next_step": "book_restaurant for Ristorante Roma, 2 people, tonight 7pm",
            "need_clarification": False, "question": "",
            "done": False, "final_response": "",
        }),
        # Turn 2: plan says done after seeing tool result
        json.dumps({
            "next_step": "", "need_clarification": False, "question": "",
            "done": True, "final_response": "Done! Table booked at Ristorante Roma for 2 tonight.",
        }),
    ])

    final = await run_proactive_session(
        initial_state=BASE_STATE,
        user_reply="yes please",
        llm=llm,
        mcp_base=MCP_BASE,
    )

    ok = (
        final.get("outcome") == "completed"
        and len(final.get("tool_results", [])) >= 1
        and "Ristorante Roma" in final.get("final_response", "")
    )
    print(f"   outcome={final.get('outcome')}  tools_used={len(final.get('tool_results',[]))}")
    print(f"   final_response={final.get('final_response', '')!r}")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


async def scenario_multi_step(proc) -> bool:
    print("\n[multi] User says 'go ahead' → 2 tool steps → respond → done")

    llm = _ScriptedLLM([
        # Step 1: book restaurant
        json.dumps({
            "next_step": "book_restaurant for tonight",
            "need_clarification": False, "question": "",
            "done": False, "final_response": "",
        }),
        # Step 2: set reminder
        json.dumps({
            "next_step": "set_reminder for 6:30pm today",
            "need_clarification": False, "question": "",
            "done": False, "final_response": "",
        }),
        # Step 3: done
        json.dumps({
            "next_step": "", "need_clarification": False, "question": "",
            "done": True,
            "final_response": "Table booked and reminder set for 6:30pm!",
        }),
    ])

    final = await run_proactive_session(
        initial_state=BASE_STATE,
        user_reply="go ahead",
        llm=llm,
        mcp_base=MCP_BASE,
    )

    ok = (
        final.get("outcome") == "completed"
        and len(final.get("tool_results", [])) >= 2
        and final.get("done") is True
    )
    print(f"   outcome={final.get('outcome')}  tools_used={len(final.get('tool_results', []))}")
    print(f"   final_response={final.get('final_response', '')!r}")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(scenario: str):
    print(f"\n{'='*60}")
    print("  Block 12: Proactive LangGraph")
    print(f"{'='*60}")

    results = []

    if scenario in ("decline", "all"):
        ok = await scenario_decline()
        results.append(("Decline", ok))

    proc = None
    if scenario in ("approve", "multi", "all"):
        proc = start_mock_server()
        print(f"   Mock MCP server started on port {PORT}")

    try:
        if scenario in ("approve", "all"):
            ok = await scenario_approve_single(proc)
            results.append(("Approve single-step", ok))

        if scenario in ("multi", "all"):
            ok = await scenario_multi_step(proc)
            results.append(("Approve multi-step", ok))
    finally:
        if proc:
            stop_mock_server(proc)

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all",
                        choices=["all", "decline", "approve", "multi"])
    args = parser.parse_args()
    asyncio.run(main(args.scenario))
