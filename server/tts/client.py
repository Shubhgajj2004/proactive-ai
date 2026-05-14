"""TTS provider abstraction."""
from abc import ABC, abstractmethod
from typing import AsyncIterator


class TTSClient(ABC):
    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Stream raw PCM audio chunks. Pipecat wraps each chunk in AudioRawFrame.
        Chunks are 16kHz mono int16 PCM bytes.
        """
        ...
