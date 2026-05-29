"""
Block 9: Context Writer test.

Writes a hardcoded AmbientAnalysis fixture to real local PostgreSQL,
then reads it back and verifies the data.

Requires:
  - docker compose up -d
  - python scripts/migrate.py

Usage:
    python tests/test_context_writer.py
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from server.ambient.context_writer import write_context, get_recent_summaries
from server.ambient.processor import AmbientAnalysis, MemoryOp

TEST_USER    = "test_context_writer_user"
TEST_SESSION = f"test-session-{uuid.uuid4().hex[:8]}"

FIXTURE_ANALYSIS = AmbientAnalysis(
    memory_operations=[
        MemoryOp(op="add", fact="User wants to book Ristorante Roma tonight"),
    ],
    summary="User is planning to dine at an Italian restaurant tonight.",
    extracted_facts=["User mentioned booking a restaurant for tonight"],
    tags=["dining", "booking", "restaurant"],
    should_act=True,
    confidence=0.83,
    proposed_action="offer to book a table at Ristorante Roma",
    consent_prompt="Want me to book a table at Ristorante Roma for tonight?",
    reasoning="User explicitly stated intent to book.",
)

FIXTURE_TRANSCRIPT = (
    "Speaker A: I should book a table at that Italian place for tonight. "
    "We haven't been there in a while."
)


async def run_tests() -> None:
    print(f"\n{'='*60}")
    print("  Block 9: Context Writer")
    print(f"{'='*60}")
    print(f"\nuser_id   : {TEST_USER}")
    print(f"session_id: {TEST_SESSION}")

    results = []

    # ── Test 1: Write ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[1] Write summary + transcript")

    write_result = await write_context(
        analysis=FIXTURE_ANALYSIS,
        raw_transcript=FIXTURE_TRANSCRIPT,
        user_id=TEST_USER,
        session_id=TEST_SESSION,
        speaker_labels=["Speaker A"],
    )

    ok1 = write_result.summary_id is not None and write_result.transcript_id is not None
    print(f"   summary_id    : {write_result.summary_id}")
    print(f"   transcript_id : {write_result.transcript_id}")
    print(f"   {'✓ PASS' if ok1 else '✗ FAIL'} — both rows inserted")
    results.append(("Write rows", ok1))

    # ── Test 2: Read back summary ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[2] Read back context_summaries")

    summaries = await get_recent_summaries(TEST_USER, limit=5)
    matching  = [s for s in summaries if s["session_id"] == TEST_SESSION]

    ok2 = len(matching) == 1
    if matching:
        s = matching[0]
        print(f"   id      : {s['id']}")
        print(f"   summary : {s['summary']}")
        print(f"   facts   : {s['extracted_facts']}")
        print(f"   tags    : {s['tags']}")
    print(f"   {'✓ PASS' if ok2 else '✗ FAIL'} — summary row found")
    results.append(("Read summary", ok2))

    # ── Test 3: Summary content matches ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[3] Summary content matches fixture")

    if matching:
        s = matching[0]
        ok3 = (
            s["summary"] == FIXTURE_ANALYSIS.summary
            and "dining" in (s["tags"] or "")
        )
        print(f"   summary match : {s['summary'] == FIXTURE_ANALYSIS.summary}")
        print(f"   tags present  : {'dining' in (s['tags'] or '')}")
    else:
        ok3 = False
    print(f"   {'✓ PASS' if ok3 else '✗ FAIL'}")
    results.append(("Content matches", ok3))

    # ── Test 4: Raw transcript ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[4] Raw transcript written correctly")

    import asyncpg
    from server.config import settings

    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        row = await conn.fetchrow(
            "SELECT raw_transcript FROM raw_transcripts WHERE session_id = $1",
            TEST_SESSION,
        )
    finally:
        await conn.close()

    ok4 = row is not None and row["raw_transcript"] == FIXTURE_TRANSCRIPT
    print(f"   transcript : {row['raw_transcript'][:80] if row else 'NOT FOUND'}")
    print(f"   {'✓ PASS' if ok4 else '✗ FAIL'} — transcript matches")
    results.append(("Raw transcript", ok4))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_tests())
