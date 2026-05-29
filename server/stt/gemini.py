"""Gemini STT backend — the ONLY file that imports google-genai for STT."""
import json
import logging
import time

import google.genai as genai
from google.genai import types

from server.prompts.stt import TRANSCRIBE
from server.stt.client import STTClient, STTSegment

logger = logging.getLogger(__name__)


class GeminiSTTClient(STTClient):
    def __init__(self, model: str, api_key: str):
        self._model_name = model
        self._client = genai.Client(api_key=api_key)
        self._last_usage_tokens = 0

    @property
    def last_usage_tokens(self) -> int:
        return self._last_usage_tokens

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> list[STTSegment]:
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        text_part  = types.Part.from_text(text=TRANSCRIBE)

        t0 = time.perf_counter()
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=[audio_part, text_part],
        )
        elapsed = time.perf_counter() - t0
        if elapsed > 10:
            logger.warning("[STT] Gemini took %.1fs — likely rate-limited (free tier quota)", elapsed)
        else:
            logger.info("[STT] Gemini latency: %.2fs", elapsed)

        # Track token usage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self._last_usage_tokens = response.usage_metadata.total_token_count or 0
        else:
            self._last_usage_tokens = 0

        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            # Remove opening fence (```json or ```)
            lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        data = json.loads(raw)
        return [STTSegment(**item) for item in data]
