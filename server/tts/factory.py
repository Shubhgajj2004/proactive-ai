"""
TTS factory — swap providers via TTS_PROVIDER in .env, zero code changes.

  TTS_PROVIDER=piper   → on-device Piper ONNX (default, fast, offline)
  TTS_PROVIDER=gemini  → Gemini cloud TTS (requires GEMINI_API_KEY)
"""
from server.config import settings
from server.tts.client import TTSClient


def make_tts_client() -> TTSClient:
    provider = getattr(settings, "TTS_PROVIDER", "piper").lower()

    if provider == "gemini":
        from server.tts.gemini import GeminiTTSClient
        return GeminiTTSClient(
            model=settings.TTS_MODEL,
            api_key=settings.GEMINI_API_KEY,
        )

    # Default: on-device Piper
    from server.tts.piper import PiperTTSClient
    return PiperTTSClient(model_path=settings.PIPER_MODEL_PATH)
