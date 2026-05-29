"""
Block 6: Session Router + Cost Governor test.

Pure logic — no external APIs, no DB, no audio.

Usage:
    python tests/test_router.py
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from server.ambient.cost_governor import CostGovernor, VAD_RATE_LIMIT, BACKOFF_SECONDS
from server.pipeline.session_router import route, contains_wake_word


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh_governor() -> CostGovernor:
    return CostGovernor()


def check(label: str, got, expected) -> bool:
    ok = got == expected
    print(f"  {'✓' if ok else '✗'} {label}")
    if not ok:
        print(f"      expected={expected!r}  got={got!r}")
    return ok


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_wake_word_detection() -> bool:
    print("\n[1] Wake word detection")
    results = [
        check("'hey jarvis' exact",            contains_wake_word("hey jarvis"), True),
        check("'Hey Jarvis' case-insensitive",  contains_wake_word("Hey Jarvis, play music"), True),
        check("'jarvis' standalone",            contains_wake_word("jarvis do this"), True),
        check("'ok jarvis'",                    contains_wake_word("ok jarvis set a timer"), True),
        check("no wake word",                   contains_wake_word("I should book a cab"), False),
        check("empty string",                   contains_wake_word(""), False),
    ]
    return all(results)


def test_routing_table() -> bool:
    print("\n[2] Routing table")
    gov = fresh_governor()
    results = []

    # AMBIENT → normal utterance, no backoff
    results.append(check(
        "normal utterance → AMBIENT",
        route("I should book the Italian place for tonight", "AMBIENT", gov),
        "AMBIENT",
    ))

    # REACTIVE → wake word
    results.append(check(
        "wake word → REACTIVE",
        route("hey jarvis book me a cab", "AMBIENT", gov),
        "REACTIVE",
    ))

    # ACTIVE → session already open
    results.append(check(
        "active session → ACTIVE",
        route("sure 2pm works for me", "ACTIVE", gov),
        "ACTIVE",
    ))

    # REACTIVE overrides ACTIVE
    results.append(check(
        "wake word during ACTIVE → REACTIVE",
        route("hey jarvis cancel that", "ACTIVE", gov),
        "REACTIVE",
    ))

    # SKIP → backoff active
    gov.set_budget_exceeded(True)
    results.append(check(
        "budget exceeded → SKIP",
        route("just some noise", "AMBIENT", gov),
        "SKIP",
    ))

    # REACTIVE still works during budget exceeded
    results.append(check(
        "wake word during budget exceeded → REACTIVE",
        route("hey jarvis help me", "AMBIENT", gov),
        "REACTIVE",
    ))

    return all(results)


def test_vad_rate_backoff() -> bool:
    print("\n[3] VAD rate backoff")
    gov = fresh_governor()
    results = []

    # Before limit — ambient allowed
    for _ in range(VAD_RATE_LIMIT):
        gov.on_vad_start()

    results.append(check(
        f"at {VAD_RATE_LIMIT} fires → still allowed",
        gov.ambient_allowed,
        True,
    ))

    # One more → triggers backoff
    gov.on_vad_start()
    results.append(check(
        f"at {VAD_RATE_LIMIT + 1} fires → backoff triggered",
        gov.ambient_allowed,
        False,
    ))

    results.append(check(
        "backoff_remaining > 0",
        gov.backoff_remaining > 0,
        True,
    ))

    # Wake word still works during backoff
    results.append(check(
        "wake word during VAD backoff → REACTIVE",
        route("hey jarvis", "AMBIENT", gov),
        "REACTIVE",
    ))

    # SKIP during backoff
    results.append(check(
        "normal utterance during VAD backoff → SKIP",
        route("just talking", "AMBIENT", gov),
        "SKIP",
    ))

    return all(results)


def test_backoff_expires() -> bool:
    print("\n[4] Backoff expires")
    # Use short backoff for test
    gov = CostGovernor(vad_rate_limit=1, backoff_seconds=0.1)
    gov.on_vad_start()
    gov.on_vad_start()  # triggers backoff

    results = []
    results.append(check("ambient blocked immediately after backoff", gov.ambient_allowed, False))

    time.sleep(0.15)   # let backoff expire
    results.append(check("ambient allowed after backoff expires", gov.ambient_allowed, True))

    return all(results)


def test_vad_rate_counter() -> bool:
    print("\n[5] VAD rate counter")
    gov = fresh_governor()
    results = []

    results.append(check("rate starts at 0", gov.vad_rate, 0))

    gov.on_vad_start()
    gov.on_vad_start()
    gov.on_vad_start()
    results.append(check("rate = 3 after 3 fires", gov.vad_rate, 3))

    return all(results)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  Block 6: Session Router + Cost Governor")
    print(f"{'='*60}")

    results = [
        ("Wake word detection",  test_wake_word_detection()),
        ("Routing table",        test_routing_table()),
        ("VAD rate backoff",     test_vad_rate_backoff()),
        ("Backoff expires",      test_backoff_expires()),
        ("VAD rate counter",     test_vad_rate_counter()),
    ]

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
