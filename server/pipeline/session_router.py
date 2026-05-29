"""
Session Router — decides what to do with each transcribed utterance.

Routes (checked in priority order):
  REACTIVE  — "hey jarvis" found anywhere in transcript
               (always fires, even during cost backoff or active session)
  ACTIVE    — an action session is already open
               (route new utterance to the running agent)
  AMBIENT   — normal conversation, cost governor permits ambient LLM call
  SKIP      — cost governor is in backoff / budget exceeded, no wake word

Input:
  transcript  — full text from STT (may contain multiple speaker turns)
  session_state — "AMBIENT" | "ACTIVE"
  cost_governor — CostGovernor instance

Output:
  Literal["REACTIVE", "ACTIVE", "AMBIENT", "SKIP"]
"""
import logging
import re
from typing import Literal

from server.ambient.cost_governor import CostGovernor

logger = logging.getLogger(__name__)

Route = Literal["REACTIVE", "ACTIVE", "AMBIENT", "SKIP"]

# Wake words — matched case-insensitively anywhere in the transcript.
# Add variants here; no code changes needed elsewhere.
WAKE_WORDS = [
    "hey jarvis",
    "ok jarvis",
    "jarvis",
]

# Pre-compiled for speed (called on every utterance)
_WAKE_PATTERN = re.compile(
    "|".join(re.escape(w) for w in WAKE_WORDS),
    re.IGNORECASE,
)


def contains_wake_word(transcript: str) -> bool:
    return bool(_WAKE_PATTERN.search(transcript))


def route(
    transcript: str,
    session_state: str,
    cost_governor: CostGovernor,
) -> Route:
    """
    Determine routing for this utterance.

    Priority:
      1. Wake word → REACTIVE  (always, overrides everything)
      2. ACTIVE session → ACTIVE
      3. Ambient allowed → AMBIENT
      4. Otherwise → SKIP
    """
    # 1. Wake word — highest priority, always reactive
    if contains_wake_word(transcript):
        logger.info("[ROUTER] wake word detected → REACTIVE")
        return "REACTIVE"

    # 2. Active session already open
    if session_state == "ACTIVE":
        logger.info("[ROUTER] session ACTIVE → routing to action agent")
        return "ACTIVE"

    # 3. Ambient allowed by cost governor
    if cost_governor.ambient_allowed:
        logger.info("[ROUTER] → AMBIENT")
        return "AMBIENT"

    # 4. Backoff / budget exceeded — skip ambient LLM
    logger.info(
        "[ROUTER] → SKIP  (backoff=%.1fs  budget_exceeded=%s)",
        cost_governor.backoff_remaining,
        cost_governor._budget_exceeded,
    )
    return "SKIP"
