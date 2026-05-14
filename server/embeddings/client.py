"""Embedding provider abstraction."""
from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output vector dimensionality (e.g. 768 for text-embedding-004)."""
        ...
