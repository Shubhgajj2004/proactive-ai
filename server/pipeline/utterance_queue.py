"""
Utterance Queue — decouples VAD (fast producer) from the downstream
pipeline (slow consumer: STT → segmenter → embedder → router → LLM).

Design:
  - Bounded asyncio.Queue (default maxsize=5)
  - VAD calls submit() — non-blocking, drops if full (never blocks VAD)
  - Single worker coroutine processes utterances strictly in order
    (conversation context must be sequential)
  - Dropped utterances are logged with a counter for observability

Why maxsize=5:
  Each utterance takes 1-3s to process downstream. At 5 slots that's
  5-15s of buffered backlog — enough to absorb short bursts without
  letting the queue grow unboundedly during sustained noise.
"""
import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE = 5


@dataclass
class QueueStats:
    submitted: int = 0
    processed: int = 0
    dropped:   int = 0
    errors:    int = 0


class UtteranceQueue:
    """
    Bounded async queue between VAD and the downstream pipeline.

    Usage:
        queue = UtteranceQueue(max_size=5)

        # In VAD callback (sync or async):
        queue.submit(pcm_bytes)

        # Start the consumer (runs until cancelled):
        await queue.start(handler=my_pipeline_fn)

    The handler receives raw int16 PCM bytes for one complete utterance
    and is awaited to completion before the next utterance is dequeued.
    """

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_size)
        self._max_size = max_size
        self.stats = QueueStats()
        self._running = False

    # ── Producer side (called from VAD) ──────────────────────────────────────

    def submit(self, pcm_bytes: bytes) -> bool:
        """
        Non-blocking enqueue. Returns True if queued, False if dropped.
        Safe to call from sync code and from asyncio callbacks.
        """
        self.stats.submitted += 1
        try:
            self._queue.put_nowait(pcm_bytes)
            qsize = self._queue.qsize()
            logger.info(
                "[QUEUE] enqueued utterance #%d  size=%dB  queue=%d/%d",
                self.stats.submitted, len(pcm_bytes), qsize, self._max_size,
            )
            return True
        except asyncio.QueueFull:
            self.stats.dropped += 1
            logger.warning(
                "[QUEUE] FULL (%d/%d) — dropping utterance #%d (%dB)  "
                "total dropped: %d",
                self._max_size, self._max_size,
                self.stats.submitted, len(pcm_bytes),
                self.stats.dropped,
            )
            return False

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    # ── Consumer side ─────────────────────────────────────────────────────────

    async def start(
        self,
        handler: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """
        Start the consumer loop. Runs until cancelled.

        handler(pcm_bytes) is awaited for each utterance.
        Exceptions in the handler are caught and logged — the queue
        keeps running so one bad utterance doesn't kill the pipeline.
        """
        self._running = True
        logger.info("[QUEUE] worker started  maxsize=%d", self._max_size)

        while self._running:
            try:
                pcm_bytes = await self._queue.get()
            except asyncio.CancelledError:
                logger.info("[QUEUE] worker cancelled")
                break

            try:
                logger.info(
                    "[QUEUE] processing utterance  size=%dB  remaining=%d",
                    len(pcm_bytes), self._queue.qsize(),
                )
                await handler(pcm_bytes)
                self.stats.processed += 1
            except Exception as exc:
                self.stats.errors += 1
                logger.error(
                    "[QUEUE] handler error (utterance skipped): %s", exc, exc_info=True,
                )
            finally:
                self._queue.task_done()

        logger.info(
            "[QUEUE] worker stopped — submitted=%d processed=%d dropped=%d errors=%d",
            self.stats.submitted, self.stats.processed,
            self.stats.dropped, self.stats.errors,
        )

    def stop(self) -> None:
        """Signal the worker to stop after finishing the current utterance."""
        self._running = False

    async def join(self) -> None:
        """Wait until all currently queued utterances are processed."""
        await self._queue.join()
