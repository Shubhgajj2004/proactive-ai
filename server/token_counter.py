"""
Token accounting. Two paths, same Redis counter per user per UTC day.

  track_llm_tokens()  — after every LLMClient.complete() (OpenRouter)
  track_tokens()      — after Gemini STT and TTS calls
"""
from datetime import date

from server.db.postgres import fetchrow
from server.db.redis import get_redis
from server.llm.client import LLMResponse

_DEFAULT_BUDGET = 1_000_000


async def _increment(user_id: str, tokens: int) -> bool:
    """Increment daily token counter. Returns True if still within budget."""
    redis = get_redis()
    key = f"tokens:{user_id}:{date.today().isoformat()}"
    new_total = await redis.incrby(key, tokens)
    await redis.expire(key, 86400 * 2)  # 2-day TTL covers midnight boundary

    row = await fetchrow(
        "SELECT daily_token_budget FROM user_settings WHERE user_id = $1", user_id
    )
    budget = row["daily_token_budget"] if row else _DEFAULT_BUDGET
    return int(new_total) <= budget


async def track_llm_tokens(user_id: str, response: LLMResponse) -> bool:
    """Call after every LLMClient.complete() (OpenRouter path)."""
    tokens = response.usage_input_tokens + response.usage_output_tokens
    return await _increment(user_id, tokens)


async def track_tokens(user_id: str, usage_metadata) -> bool:
    """Call after Gemini STT / TTS calls (google-generativeai usage_metadata)."""
    tokens = getattr(usage_metadata, "total_token_count", 0) or 0
    return await _increment(user_id, tokens)
