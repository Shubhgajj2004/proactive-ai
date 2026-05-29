"""Gemini TTS backend — the ONLY file that imports google-genai for TTS."""
from typing import AsyncIterator

import google.genai as genai
from google.genai import types

from server.tts.client import TTSClient

SAMPLE_RATE = 24000   # Gemini TTS outputs 24kHz PCM L16
VOICE_NAME  = "Aoede" # Natural-sounding voice


class GeminiTTSClient(TTSClient):
    def __init__(self, model: str, api_key: str, voice: str = VOICE_NAME):
        self._model  = model
        self._voice  = voice
        self._client = genai.Client(api_key=api_key)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Synthesise text → raw int16 PCM bytes at 24kHz.
        Yields the full blob as one chunk (Gemini TTS is non-streaming).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        raw  = await loop.run_in_executor(None, self._synthesize_sync, text)
        if raw:
            yield raw

    def _synthesize_sync(self, text: str) -> bytes | None:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self._voice
                            )
                        )
                    ),
                ),
            )
            return response.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("[TTS] synthesis failed: %s", e)
            return None
