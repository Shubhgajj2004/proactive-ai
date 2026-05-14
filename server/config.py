from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM via OpenRouter ───────────────────────────────────────────────────
    OPENROUTER_API_KEY: str = "sk-or-placeholder"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_AMBIENT_MODEL: str = "inclusionai/ring-2.6-1t:free"
    LLM_ACTION_STANDARD_MODEL: str = "inclusionai/ring-2.6-1t:free"
    LLM_ACTION_PREMIUM_MODEL: str = "inclusionai/ring-2.6-1t:free"
    # ── Example alternatives (swap in .env, no code change) ─────────────────
    # LLM_AMBIENT_MODEL      = "google/gemini-2.5-flash"
    # LLM_ACTION_PREMIUM_MODEL = "anthropic/claude-sonnet-4-6"
    # LLM_ACTION_PREMIUM_MODEL = "openai/gpt-4o"
    # To use direct Gemini API instead of OpenRouter:
    #   LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    #   OPENROUTER_API_KEY = <your-gemini-key>

    # ── STT (direct Gemini SDK) ──────────────────────────────────────────────
    GEMINI_API_KEY: str = "placeholder"
    STT_MODEL: str = "gemini-3-flash-preview"
    # Example alternative: STT_MODEL = "gemini-2.0-flash"

    # ── TTS (direct Gemini SDK) ──────────────────────────────────────────────
    TTS_MODEL: str = "gemini-2.5-flash-preview-tts"
    # Example alternative: TTS_MODEL = "gemini-2.0-flash-tts"

    # ── Embeddings (direct Google SDK) ──────────────────────────────────────
    EMBEDDING_MODEL: str = "text-embedding-004"  # 768-dim

    # ── Infrastructure ───────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://proactive:proactive@localhost:5432/proactive_ai"
    REDIS_URL: str = "redis://localhost:6379"
    DAILY_API_KEY: str = "placeholder"

    # ── Audio ────────────────────────────────────────────────────────────────
    AUDIO_GAIN_FACTOR: float = 5.0  # hardware mic volume boost (first pipeline stage)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
