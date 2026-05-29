"""
Block 8: Memory Writer test.

Tests add → search → update → delete round-trip against real
local PostgreSQL + pgvector. No mock — this hits the actual DB.

Requires:
  - docker compose up -d  (PostgreSQL running)
  - DB migrated: python scripts/migrate.py
  - GEMINI_API_KEY and OPENROUTER_API_KEY set in .env

Usage:
    python tests/test_memory_writer.py
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from server.ambient.memory_writer import (
    apply_memory_ops, search_memories, get_all_memories,
)
from server.ambient.processor import MemoryOp

TEST_USER = "test_memory_writer_user"


async def cleanup(user_id: str) -> None:
    """Delete all test memories before and after the test."""
    from server.ambient.memory_writer import _get_memory, _get_all_sync
    import asyncio
    loop = asyncio.get_event_loop()
    memory = _get_memory()
    all_mems = await get_all_memories(user_id)
    for m in all_mems:
        try:
            await loop.run_in_executor(None, memory.delete, m["id"])
        except Exception:
            pass


async def run_tests() -> None:
    print(f"\n{'='*60}")
    print("  Block 8: Memory Writer (mem0 self-hosted)")
    print(f"{'='*60}")
    print("\nConnecting to local PostgreSQL + pgvector…")

    results = []

    # Clean slate
    await cleanup(TEST_USER)

    # ── Test 1: ADD ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[1] ADD — write a fact about the wearer")

    await apply_memory_ops(
        [MemoryOp(op="add", fact="User prefers window seats at restaurants")],
        user_id=TEST_USER,
    )

    all_mems = await get_all_memories(TEST_USER)
    ok1 = len(all_mems) >= 1
    print(f"   Memories after add: {len(all_mems)}")
    for m in all_mems:
        print(f"   id={m['id']}  fact={m['memory']!r}")
    print(f"   {'✓ PASS' if ok1 else '✗ FAIL'} — expected ≥1 memory")
    results.append(("ADD", ok1))

    # ── Test 2: SEARCH ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[2] SEARCH — retrieve relevant memories")

    facts = await search_memories("seating preference", user_id=TEST_USER, limit=5)
    ok2 = len(facts) >= 1
    print(f"   Results: {facts}")
    print(f"   {'✓ PASS' if ok2 else '✗ FAIL'} — expected ≥1 result")
    results.append(("SEARCH", ok2))

    # ── Test 3: UPDATE ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[3] UPDATE — modify an existing memory")

    memory_id = all_mems[0]["id"]
    await apply_memory_ops(
        [MemoryOp(op="update", fact="User strongly prefers window seats at restaurants", memory_id=memory_id)],
        user_id=TEST_USER,
    )

    updated = await get_all_memories(TEST_USER)
    updated_facts = [m["memory"] for m in updated]
    ok3 = any("strongly" in f for f in updated_facts)
    print(f"   Updated facts: {updated_facts}")
    print(f"   {'✓ PASS' if ok3 else '✗ FAIL'} — expected updated fact to contain 'strongly'")
    results.append(("UPDATE", ok3))

    # ── Test 4: DELETE ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[4] DELETE — remove a memory")

    await apply_memory_ops(
        [MemoryOp(op="delete", memory_id=memory_id)],
        user_id=TEST_USER,
    )

    after_delete = await get_all_memories(TEST_USER)
    remaining_ids = [m["id"] for m in after_delete]
    ok4 = memory_id not in remaining_ids
    print(f"   Memories after delete: {len(after_delete)}")
    print(f"   {'✓ PASS' if ok4 else '✗ FAIL'} — deleted memory should be gone")
    results.append(("DELETE", ok4))

    # ── Test 5: SEARCH returns empty after delete ─────────────────────────────
    print(f"\n{'─'*60}")
    print("[5] SEARCH after delete — should return empty")

    await cleanup(TEST_USER)   # clean everything
    facts_after = await search_memories("window seat", user_id=TEST_USER, limit=5)
    ok5 = len(facts_after) == 0
    print(f"   Results: {facts_after}")
    print(f"   {'✓ PASS' if ok5 else '✗ FAIL'} — expected empty")
    results.append(("SEARCH after delete", ok5))

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
