"""All LLM callers use make_llm_client(). Never import openai directly outside openrouter.py."""
from server.config import settings
from server.llm.client import LLMClient
from server.llm.openrouter import OpenRouterClient


def make_llm_client(tier: str) -> LLMClient:
    """
    tier: 'ambient' | 'standard' | 'premium'
    To swap backend: change settings + optionally subclass LLMClient. No other code changes.
    """
    model = {
        "ambient": settings.LLM_AMBIENT_MODEL,
        "standard": settings.LLM_ACTION_STANDARD_MODEL,
        "premium": settings.LLM_ACTION_PREMIUM_MODEL,
    }[tier]

    return OpenRouterClient(
        model=model,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )
