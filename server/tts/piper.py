"""
Piper TTS backend — fully on-device, no API calls.

Piper is a fast neural TTS engine that runs on CPU in real-time.
Voice model (~60MB ONNX) lives at models/tts/ in the project root.

Outputs: raw int16 PCM at the model's native sample rate (22050 Hz for
lessac-medium). The WAV wrapper in test_server.py uses this rate.
"""
import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

from server.tts.client import TTSClient

logger = logging.getLogger(__name__)

# Lazy-loaded so the ONNX model is not parsed at import time
_voice_cache: dict = {}


def _load_voice(model_path: str):
    """Load (and cache) a PiperVoice from an ONNX model path."""
    if model_path not in _voice_cache:
        from piper.voice import PiperVoice  # type: ignore
        logger.info("[TTS/Piper] loading model: %s", model_path)
        _voice_cache[model_path] = PiperVoice.load(model_path)
        sr = _voice_cache[model_path].config.sample_rate
        logger.info("[TTS/Piper] model loaded — sample_rate=%d", sr)
    return _voice_cache[model_path]


def _synthesize_sync(model_path: str, text: str) -> bytes:
    """Blocking synthesis — called via run_in_executor."""
    voice = _load_voice(model_path)
    chunks: list[bytes] = []
    for chunk in voice.synthesize(text):
        chunks.append(chunk.audio_int16_bytes)
    return b"".join(chunks)


class PiperTTSClient(TTSClient):
    """
    On-device TTS using Piper (ONNX).

    Args:
        model_path: Absolute path to the .onnx voice model file.
    """

    def __init__(self, model_path: str):
        self._model_path = str(Path(model_path).expanduser().resolve())
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Piper model not found: {self._model_path}\n"
                "Run:  bash scripts/download_piper_voice.sh"
            )

    @property
    def sample_rate(self) -> int:
        """Return the model's native sample rate (loaded lazily)."""
        return _load_voice(self._model_path).config.sample_rate

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Synthesise text on-device and yield the PCM blob."""
        loop = asyncio.get_event_loop()
        pcm = await loop.run_in_executor(
            None, _synthesize_sync, self._model_path, text
        )
        if pcm:
            logger.info(
                "[TTS/Piper] synthesised %d chars → %d bytes PCM",
                len(text), len(pcm),
            )
            yield pcm
