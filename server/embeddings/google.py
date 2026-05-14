"""Google embedding backend — the ONLY file that imports google-generativeai for embeddings."""
import asyncio

import google.generativeai as genai

from server.embeddings.client import EmbeddingClient

_DIMENSIONS = {
    "text-embedding-004": 768,
}


class GoogleEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str, api_key: str):
        genai.configure(api_key=api_key)
        self._model = model
        self._dims = _DIMENSIONS.get(model, 768)

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(model=self._model, content=text),
        )
        return result["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(model=self._model, content=texts),
        )
        return result["embedding"]
