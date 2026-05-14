"""Gemini TTS backend — the ONLY file that imports google-generativeai for TTS."""
from typing import AsyncIterator

import google.generativeai as genai

from server.tts.client import TTSClient


class GeminiTTSClient(TTSClient):
    def __init__(self, model: str, api_key: str):
        genai.configure(api_key=api_key)
        self._model_name = model

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        model = genai.GenerativeModel(self._model_name)

        # Gemini TTS streams audio chunks
        response = model.generate_content(
            text,
            stream=True,
            generation_config=genai.GenerationConfig(
                response_mime_type="audio/pcm",
                audio_encoding="LINEAR16",
                sample_rate_hertz=16000,
            ),
        )

        for chunk in response:
            if hasattr(chunk, "audio") and chunk.audio:
                yield chunk.audio
            elif chunk.text:
                # Some versions return base64-encoded audio as text
                import base64
                yield base64.b64decode(chunk.text)
