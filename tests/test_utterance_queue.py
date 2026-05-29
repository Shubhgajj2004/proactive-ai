"""
Utterance Queue test.

Verifies:
  1. Normal flow    — utterances processed in order, no drops
  2. Overflow       — queue full → newest dropped, oldest processed
  3. Handler error  — one bad utterance doesn't kill the worker
  4. Stats          — submitted/processed/dropped/errors counts are correct

Usage:
    python tests/test_utterance_queue.py
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from server.pipeline.utterance_queue import UtteranceQueue


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pcm(tag: int, size: int = 1024) -> bytes:
    """Fake PCM blob tagged with a byte value so we can identify it."""
    return bytes([tag % 256]) * size


async def run_worker(queue: UtteranceQueue, handler, timeout: float = 5.0):
    """Run queue worker until it drains or timeout."""
    task = asyncio.create_task(queue.start(handler))
    try:
        await asyncio.wait_for(queue.join(), timeout=timeout)
    finally:
        queue.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_normal_flow() -> bool:
    """3 utterances submitted to a size-5 queue — all processed in order."""
    print("\n[1] Normal flow (3 utterances, queue size 5)")

    received = []

    async def handler(pcm: bytes):
        await asyncio.sleep(0.01)   # simulate fast processing
        received.append(pcm[0])     # tag byte

    queue = UtteranceQueue(max_size=5)
    for i in range(3):
        queue.submit(make_pcm(i))

    await run_worker(queue, handler)

    ok = received == [0, 1, 2]
    print(f"   Received in order: {received}  {'✓' if ok else '✗'}")
    print(f"   Stats: {queue.stats}")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


async def test_overflow_drop() -> bool:
    """Submit 8 utterances to a size-5 queue — last 3 dropped."""
    print("\n[2] Overflow (8 utterances, queue size 5 → 3 dropped)")

    received = []
    processing_started = asyncio.Event()

    async def slow_handler(pcm: bytes):
        processing_started.set()
        await asyncio.sleep(0.05)   # slow — keeps queue occupied
        received.append(pcm[0])

    queue = UtteranceQueue(max_size=5)

    # Submit all 8 synchronously before worker starts
    results = [queue.submit(make_pcm(i)) for i in range(8)]

    await run_worker(queue, slow_handler)

    queued  = sum(results)
    dropped = results.count(False)
    ok = queued == 5 and dropped == 3 and queue.stats.dropped == 3
    print(f"   Queued: {queued}/8  Dropped: {dropped}/8")
    print(f"   Stats: {queue.stats}")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'} — expected 5 queued, 3 dropped")
    return ok


async def test_handler_error() -> bool:
    """Handler raises on utterance #2 — worker continues with #3."""
    print("\n[3] Handler error (utterance #2 raises — worker must survive)")

    received = []

    async def flaky_handler(pcm: bytes):
        tag = pcm[0]
        if tag == 1:
            raise ValueError("simulated STT failure")
        received.append(tag)

    queue = UtteranceQueue(max_size=5)
    for i in range(3):
        queue.submit(make_pcm(i))

    await run_worker(queue, flaky_handler)

    ok = received == [0, 2] and queue.stats.errors == 1
    print(f"   Received (skipping failed): {received}")
    print(f"   Stats: {queue.stats}")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'} — expected [0, 2], 1 error")
    return ok


async def test_stats() -> bool:
    """Stats counters are accurate across submit/process/drop/error."""
    print("\n[4] Stats accuracy")

    async def handler(pcm: bytes):
        if pcm[0] == 1:          # tag 1 IS in the queue (0,1,2 queued; 3,4 dropped)
            raise RuntimeError("boom")

    queue = UtteranceQueue(max_size=3)
    # Submit 5: first 3 queued (tags 0,1,2), last 2 dropped (tags 3,4)
    for i in range(5):
        queue.submit(make_pcm(i))

    await run_worker(queue, handler)

    s = queue.stats
    ok = (
        s.submitted == 5
        and s.dropped   == 2
        and s.processed == 2   # 3 queued, 1 errored → 2 processed
        and s.errors    == 1
    )
    print(f"   submitted={s.submitted} processed={s.processed} "
          f"dropped={s.dropped} errors={s.errors}")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    print(f"\n{'='*60}")
    print("  Utterance Queue tests")
    print(f"{'='*60}")

    results = [
        ("Normal flow",    await test_normal_flow()),
        ("Overflow drop",  await test_overflow_drop()),
        ("Handler error",  await test_handler_error()),
        ("Stats accuracy", await test_stats()),
    ]

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
