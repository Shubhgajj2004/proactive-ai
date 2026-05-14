"""All TTS callers use make_tts_client(). Never import google-generativeai outside tts/gemini.py."""
from server.config import settings
from server.tts.client import TTSClient
from server.tts.gemini import GeminiTTSClient


def make_tts_client() -> TTSClient:
    """To swap TTS: subclass TTSClient, add a new file here, update this factory."""
    return GeminiTTSClient(
        model=settings.TTS_MODEL,
        api_key=settings.GEMINI_API_KEY,
    )
