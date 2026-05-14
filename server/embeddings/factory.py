"""All embedding callers use make_embedding_client(). Never import google-generativeai outside embeddings/google.py."""
from server.config import settings
from server.embeddings.client import EmbeddingClient
from server.embeddings.google import GoogleEmbeddingClient


def make_embedding_client() -> EmbeddingClient:
    """To swap embeddings: subclass EmbeddingClient, add a new file here, update this factory."""
    return GoogleEmbeddingClient(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.GEMINI_API_KEY,
    )
