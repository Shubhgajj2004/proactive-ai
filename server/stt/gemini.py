"""Gemini STT backend — the ONLY file that imports google-generativeai for STT."""
import json

import google.generativeai as genai

from server.prompts.stt import TRANSCRIBE
from server.stt.client import STTClient, STTSegment


class GeminiSTTClient(STTClient):
    def __init__(self, model: str, api_key: str):
        genai.configure(api_key=api_key)
        self._model_name = model
        self._last_usage_tokens = 0

    @property
    def last_usage_tokens(self) -> int:
        return self._last_usage_tokens

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> list[STTSegment]:
        model = genai.GenerativeModel(self._model_name)

        audio_part = {
            "mime_type": "audio/wav",
            "data": audio_bytes,
        }

        response = model.generate_content([audio_part, TRANSCRIBE])

        # Track token usage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self._last_usage_tokens = response.usage_metadata.total_token_count
        else:
            self._last_usage_tokens = 0

        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        return [STTSegment(**item) for item in data]
