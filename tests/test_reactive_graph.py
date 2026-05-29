"""
Block 13: Reactive LangGraph test.

Two scenarios with mock LLM + mock MCP server:
  1. direct         — clear intent, no clarification needed → plan → tool → respond
  2. clarify        — ambiguous intent → one question → clarification → plan → tool → respond

Usage:
    python tests/test_reactive_graph.py
    python tests/test_reactive_graph.py --scenario direct
    python tests/test_reactive_graph.py --scenario clarify
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

from server.action.reactive_graph import run_reactive_session
from server.llm.client import LLMClient, LLMResponse

PORT     = 8891
MCP_BASE = f"http://127.0.0.1:{PORT}/tools"


# ── Mock LLM ──────────────────────────────────────────────────────────────────

class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    async def complete(self, messages, response_format=None, temperature=0.3) -> LLMResponse:
        if self._idx < len(self._responses):
            content = self._responses[self._idx]
            self._idx += 1
        else:
            content = json.dumps({
                "next_step": "", "need_clarification": False,
                "question": "", "done": True, "final_response": "Done!",
            })
        return LLMResponse(content=content, usage_input_tokens=40, usage_output_tokens=25)

    @property
    def last_usage_tokens(self) -> int:
        return 65


# ── Server helpers ────────────────────────────────────────────────────────────

def start_server() -> subprocess.Popen:
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


def stop_server(proc):
    proc.terminate()
    proc.wait(timeout=5)


# ── Scenarios ─────────────────────────────────────────────────────────────────

async def scenario_direct() -> bool:
    """Clear intent — no clarification, goes straight to plan."""
    print("\n[direct] 'hey jarvis, play some jazz music' → intent clear → tool → done")
    proc = start_server()
    try:
        llm = _ScriptedLLM([
            # Intent node: clear
            json.dumps({
                "intent_clear": True,
                "parsed_intent": "play jazz music on Spotify",
                "clarifying_question": "",
            }),
            # Plan node turn 1: call play_music
            json.dumps({
                "next_step": "play jazz music on Spotify",
                "need_clarification": False, "question": "",
                "done": False, "final_response": "",
            }),
            # Plan node turn 2: done
            json.dumps({
                "next_step": "", "need_clarification": False, "question": "",
                "done": True, "final_response": "Playing jazz on Spotify.",
            }),
        ])

        final = await run_reactive_session(
            transcript="hey jarvis, play some jazz music",
            llm=llm, mcp_base=MCP_BASE,
        )

        ok = (
            final.get("outcome") == "completed"
            and len(final.get("tool_results", [])) >= 1
            and "jazz" in final.get("final_response", "").lower()
            or "Playing" in final.get("final_response", "")
        )
        print(f"   outcome={final.get('outcome')}  tools={len(final.get('tool_results',[]))}")
        print(f"   final={final.get('final_response')!r}")
        print(f"   no consent prompt: {'✓' if not final.get('consent_prompt') else '✗'}")
        print(f"   {'✓ PASS' if ok else '✗ FAIL'}")
        return ok
    finally:
        stop_server(proc)


async def scenario_clarify() -> bool:
    """Ambiguous intent — one clarifying question then plan."""
    print("\n[clarify] 'hey jarvis, book the usual' → ambiguous → question → clarify → tool → done")
    proc = start_server()
    try:
        # First run: intent is ambiguous → graph pauses at END
        llm_first = _ScriptedLLM([
            json.dumps({
                "intent_clear": False,
                "parsed_intent": "",
                "clarifying_question": "Book a cab or a restaurant?",
            }),
        ])
        state_after_intent = await run_reactive_session(
            transcript="hey jarvis, book the usual",
            llm=llm_first, mcp_base=MCP_BASE,
        )

        # Check that a clarifying question was emitted
        messages = state_after_intent.get("messages", [])
        questions = [
            m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            for m in messages
            if (isinstance(m, dict) and m.get("role") == "assistant")
            or (hasattr(m, "type") and m.type == "ai")
        ]
        question_found = any("cab" in q.lower() or "restaurant" in q.lower() for q in questions)

        print(f"   Clarifying question found: {'✓' if question_found else '✗'}")
        print(f"   Questions: {questions}")

        # Second run: user replies "a cab to the office"
        # Fresh LLM with responses for the resumed execution
        llm_second = _ScriptedLLM([
            # Intent node re-runs on resume — produces clear intent
            json.dumps({
                "intent_clear": True,
                "parsed_intent": "book a cab to the office",
                "clarifying_question": "",
            }),
            # Plan turn 1: call tool
            json.dumps({
                "next_step": "book a cab to the office",
                "need_clarification": False, "question": "",
                "done": False, "final_response": "",
            }),
            # Plan turn 2: done
            json.dumps({
                "next_step": "", "need_clarification": False, "question": "",
                "done": True, "final_response": "Cab booked to the office.",
            }),
        ])
        final = await run_reactive_session(
            transcript="hey jarvis, book the usual",
            llm=llm_second,
            mcp_base=MCP_BASE,
            clarification="A cab to the office",
        )

        ok = (
            question_found
            and final.get("outcome") == "completed"
            and len(final.get("tool_results", [])) >= 1
        )
        print(f"   outcome={final.get('outcome')}  tools={len(final.get('tool_results',[]))}")
        print(f"   final={final.get('final_response')!r}")
        print(f"   {'✓ PASS' if ok else '✗ FAIL'}")
        return ok
    finally:
        stop_server(proc)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(scenario: str):
    print(f"\n{'='*60}")
    print("  Block 13: Reactive LangGraph")
    print(f"{'='*60}")

    results = []

    if scenario in ("direct", "all"):
        ok = await scenario_direct()
        results.append(("Direct intent", ok))

    if scenario in ("clarify", "all"):
        ok = await scenario_clarify()
        results.append(("Clarify then execute", ok))

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
                        choices=["all", "direct", "clarify"])
    args = parser.parse_args()
    asyncio.run(main(args.scenario))
