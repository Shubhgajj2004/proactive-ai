"""STT provider abstraction."""
from abc import ABC, abstractmethod

from pydantic import BaseModel


class STTSegment(BaseModel):
    """
    One contiguous speaker turn.

    No language field — the script itself carries that information.
    Hindi appears in Devanagari, English in Latin, Telugu in Telugu script, etc.
    Code-switching within a single turn is preserved exactly as spoken.
    """
    start_ms: int        # turn start in milliseconds
    end_ms: int          # turn end in milliseconds
    speaker_label: str   # e.g. "Speaker A", "Speaker B"
    text: str            # verbatim in original script; code-switching preserved


class STTClient(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> list[STTSegment]: ...

    @property
    @abstractmethod
    def last_usage_tokens(self) -> int:
        """Total tokens used in the most recent transcribe() call (for token_counter)."""
        ...
