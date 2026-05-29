"""Google embedding backend — the ONLY file that imports google-genai for embeddings."""
import asyncio

import google.genai as genai
from google.genai import types

from server.embeddings.client import EmbeddingClient

_DIMENSIONS = {
    "gemini-embedding-001":       768,
    "gemini-embedding-2":         768,
    "gemini-embedding-2-preview": 768,
}


class GoogleEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str, api_key: str):
        self._model  = model
        self._dims   = _DIMENSIONS.get(model, 768)
        self._client = genai.Client(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return self._dims

    def _embed_sync(self, text: str) -> list[float]:
        response = self._client.models.embed_content(
            model=f"models/{self._model}",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self._dims),
        )
        return list(response.embeddings[0].values)

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            response = self._client.models.embed_content(
                model=f"models/{self._model}",
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self._dims),
            )
            results.append(list(response.embeddings[0].values))
        return results

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_batch_sync, texts)
