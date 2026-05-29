"""
Cost Governor — guards ambient LLM processing against runaway costs.

Two independent checks, both must pass for ambient to be allowed:

1. VAD rate limit
   Rolling 60s window of VAD start events. If > VAD_RATE_LIMIT fires/min,
   ambient is paused for BACKOFF_SECONDS (too much noise / continuous speech).
   Reactive ("hey jarvis") always bypasses this.

2. Daily token budget
   Tracked in Redis (incremented by token_counter.py after every LLM/STT call).
   If budget exceeded, ambient is paused until next UTC day.
   Reactive path still works regardless.

STT always runs — wake word detection needs the transcript.
Only the ambient LLM call is gated.
"""
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

VAD_RATE_LIMIT   = 60    # max VAD fires per 60s window before backoff
BACKOFF_SECONDS  = 30    # how long to pause ambient after rate limit hit
WINDOW_SECONDS   = 60    # rolling window size


class CostGovernor:
    """
    Stateful cost governor. One instance per session/pipeline.

    Thread-safety: not thread-safe — intended for a single asyncio task.
    """

    def __init__(
        self,
        vad_rate_limit: int = VAD_RATE_LIMIT,
        backoff_seconds: float = BACKOFF_SECONDS,
        window_seconds: float = WINDOW_SECONDS,
    ):
        self._rate_limit      = vad_rate_limit
        self._backoff_seconds = backoff_seconds
        self._window_seconds  = window_seconds

        # Rolling timestamp deque — maxlen caps memory usage
        self._vad_timestamps: deque[float] = deque(maxlen=200)
        self._paused_until: float = 0.0
        self._budget_exceeded: bool = False

    # ── VAD hook ──────────────────────────────────────────────────────────────

    def on_vad_start(self) -> None:
        """Call this every time VAD fires a START event."""
        now = time.time()
        self._vad_timestamps.append(now)

        recent = self._recent_count(now)
        if recent > self._rate_limit and now >= self._paused_until:
            self._paused_until = now + self._backoff_seconds
            logger.warning(
                "[COST] VAD rate %d/min > limit %d — ambient paused for %ds",
                recent, self._rate_limit, self._backoff_seconds,
            )

    # ── Budget hook ───────────────────────────────────────────────────────────

    def set_budget_exceeded(self, exceeded: bool) -> None:
        """Called by token_counter when daily budget is hit."""
        if exceeded and not self._budget_exceeded:
            logger.warning("[COST] daily token budget exceeded — ambient paused until midnight")
        self._budget_exceeded = exceeded

    # ── Decision ──────────────────────────────────────────────────────────────

    @property
    def ambient_allowed(self) -> bool:
        """True if ambient LLM processing is permitted right now."""
        now = time.time()

        if self._budget_exceeded:
            logger.debug("[COST] ambient blocked — budget exceeded")
            return False

        if now < self._paused_until:
            remaining = self._paused_until - now
            logger.debug("[COST] ambient blocked — backoff %.1fs remaining", remaining)
            return False

        return True

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def vad_rate(self) -> int:
        """VAD fires in the last 60s."""
        return self._recent_count(time.time())

    @property
    def backoff_remaining(self) -> float:
        """Seconds until ambient resumes (0 if not in backoff)."""
        return max(0.0, self._paused_until - time.time())

    def _recent_count(self, now: float) -> int:
        cutoff = now - self._window_seconds
        return sum(1 for t in self._vad_timestamps if t > cutoff)
