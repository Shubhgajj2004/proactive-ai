"""All STT callers use make_stt_client(). Never import google-generativeai outside stt/gemini.py."""
from server.config import settings
from server.stt.client import STTClient
from server.stt.gemini import GeminiSTTClient


def make_stt_client() -> STTClient:
    """To swap STT: subclass STTClient, add a new file here, update this factory."""
    return GeminiSTTClient(
        model=settings.STT_MODEL,
        api_key=settings.GEMINI_API_KEY,
    )
