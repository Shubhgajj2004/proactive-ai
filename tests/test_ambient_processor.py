"""
Block 7: Ambient Processor test.

Two modes:

  Mock (default) — LLM returns a hardcoded fixture, no API call:
      python tests/test_ambient_processor.py

  Real API — calls the configured LLM via OpenRouter:
      python tests/test_ambient_processor.py --real-api
      python tests/test_ambient_processor.py --real-api --transcript "your text here"
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from server.ambient.processor import AmbientAnalysis, AmbientProcessor, CONFIDENCE_THRESHOLD
from server.llm.client import LLMClient, LLMResponse


# ── Mock LLM client ───────────────────────────────────────────────────────────

MOCK_ANALYSIS = {
    "memory_operations": [
        {"op": "add", "fact": "User wants to book a table at Ristorante Roma tonight", "memory_id": None}
    ],
    "summary": "User is planning to dine at an Italian restaurant tonight.",
    "extracted_facts": ["User mentioned booking a restaurant for tonight"],
    "tags": ["dining", "booking", "restaurant"],
    "should_act": True,
    "confidence": 0.83,
    "proposed_action": "offer to book a table at Ristorante Roma",
    "consent_prompt": "Want me to book a table at Ristorante Roma for tonight?",
    "reasoning": "User explicitly said they should book the Italian place tonight, suggesting intent.",
}


class _MockLLMClient(LLMClient):
    def __init__(self, response_json: dict | None = None, fail_first: bool = False):
        self._response = json.dumps(response_json or MOCK_ANALYSIS)
        self._fail_first = fail_first
        self._calls = 0

    async def complete(self, messages, response_format=None, temperature=0.3) -> LLMResponse:
        self._calls += 1
        if self._fail_first and self._calls == 1:
            return LLMResponse(
                content="this is not json {{{",
                usage_input_tokens=10,
                usage_output_tokens=5,
            )
        return LLMResponse(
            content=self._response,
            usage_input_tokens=120,
            usage_output_tokens=80,
        )

    @property
    def last_usage_tokens(self) -> int:
        return 200


# ── Test helpers ──────────────────────────────────────────────────────────────

DEFAULT_TRANSCRIPT = (
    "Speaker A: I should book a table at that Italian place for tonight. "
    "We haven't been there in a while and it would be nice."
)
DEFAULT_MEMORIES = [
    "User likes Ristorante Roma",
    "User usually books for 2 people",
    "User prefers window seats",
]


def _print_analysis(analysis: AmbientAnalysis) -> None:
    print(f"\n  Summary      : {analysis.summary}")
    print(f"  Confidence   : {analysis.confidence:.2f}")
    print(f"  Should act   : {analysis.should_act}")
    if analysis.should_act:
        print(f"  Proposed     : {analysis.proposed_action}")
        print(f"  Consent      : {analysis.consent_prompt}")
    print(f"  Memory ops   : {len(analysis.memory_operations)}")
    for op in analysis.memory_operations:
        print(f"    [{op.op}] {op.fact}")
    print(f"  Facts        : {analysis.extracted_facts}")
    print(f"  Tags         : {analysis.tags}")


def _validate(analysis: AmbientAnalysis, label: str) -> bool:
    print(f"\n{'─'*60}")
    print(f"Validation: {label}")
    results = []

    def chk(name, ok):
        results.append(ok)
        print(f"  {'✓' if ok else '✗'} {name}")
        return ok

    chk("summary is non-empty",          bool(analysis.summary.strip()))
    chk("confidence in [0,1]",           0.0 <= analysis.confidence <= 1.0)
    chk("should_act is bool",            isinstance(analysis.should_act, bool))
    chk("memory_operations is list",     isinstance(analysis.memory_operations, list))
    chk("extracted_facts is list",       isinstance(analysis.extracted_facts, list))

    if analysis.should_act:
        chk("consent_prompt non-empty",  bool(analysis.consent_prompt.strip()))
        chk("proposed_action non-empty", bool(analysis.proposed_action.strip()))
        chk(f"confidence > {CONFIDENCE_THRESHOLD} when should_act",
            analysis.confidence >= CONFIDENCE_THRESHOLD)
    else:
        chk("confidence < threshold when not acting",
            analysis.confidence < CONFIDENCE_THRESHOLD)

    return all(results)


# ── Test modes ────────────────────────────────────────────────────────────────

async def test_mock() -> bool:
    print(f"\n{'='*60}")
    print("  Mock mode (no API call)")
    print(f"{'='*60}")

    processor = AmbientProcessor(client=_MockLLMClient())
    analysis  = await processor.analyse(
        transcript=DEFAULT_TRANSCRIPT,
        memories=DEFAULT_MEMORIES,
        capability_manifest="book_restaurant: Book a table at a restaurant",
    )

    _print_analysis(analysis)
    return _validate(analysis, "mock response")


async def test_mock_retry() -> bool:
    print(f"\n{'='*60}")
    print("  Mock retry (first call returns bad JSON)")
    print(f"{'='*60}")

    processor = AmbientProcessor(client=_MockLLMClient(fail_first=True))
    try:
        analysis = await processor.analyse(transcript=DEFAULT_TRANSCRIPT)
        _print_analysis(analysis)
        ok = _validate(analysis, "retry recovery")
        print(f"\n  ✓ Recovered from bad JSON on retry")
        return ok
    except ValueError as e:
        print(f"  ✗ Retry failed: {e}")
        return False


async def test_real_api(transcript: str) -> bool:
    print(f"\n{'='*60}")
    print("  Real API mode")
    print(f"{'='*60}")
    print(f"\nTranscript: {transcript[:120]}")

    from server.llm.factory import make_llm_client
    processor = AmbientProcessor(client=make_llm_client("ambient"))

    print("\nCalling LLM…")
    analysis = await processor.analyse(
        transcript=transcript,
        memories=DEFAULT_MEMORIES,
        capability_manifest="book_restaurant: Book a table at a restaurant\nbook_cab: Book a ride",
    )

    _print_analysis(analysis)
    return _validate(analysis, "real API response")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Block 7: Ambient Processor test")
    parser.add_argument("--real-api",   action="store_true")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    args = parser.parse_args()

    results = []

    if args.real_api:
        ok = await test_real_api(args.transcript)
        results.append(("Real API", ok))
    else:
        results.append(("Mock response",  await test_mock()))
        results.append(("Mock retry",     await test_mock_retry()))

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
