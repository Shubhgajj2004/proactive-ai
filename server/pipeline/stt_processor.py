"""
STT Processor — third pipeline stage (after vad_processor.py).

Receives complete utterance blobs (int16 PCM bytes) emitted by VadProcessor,
wraps them as a valid WAV before sending to the STT provider, and returns
a list of STTSegment objects.

Never imports google-generativeai directly — always goes through stt/factory.py.
"""
import io
import logging
import wave

from server.stt.client import STTClient, STTSegment
from server.stt.factory import make_stt_client

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2   # int16 = 2 bytes
NUM_CHANNELS = 1


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw int16 PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class STTProcessor:
    """
    Stateless STT processor.

    Call transcribe(pcm_bytes) with a complete utterance blob.
    Returns a list of STTSegments (speaker turns with timestamps).

    Args:
        client: Optional injected STTClient (for testing). If None, uses factory.
    """

    def __init__(self, client: STTClient | None = None):
        self._client = client or make_stt_client()

    async def transcribe(self, pcm_bytes: bytes) -> list[STTSegment]:
        """
        Convert PCM utterance to WAV and send to STT provider.
        Returns list of speaker-turn segments with start_ms/end_ms.
        """
        duration_s = len(pcm_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH)
        logger.info("[STT] transcribing %.1fs utterance (%d bytes)", duration_s, len(pcm_bytes))

        wav_bytes = pcm_to_wav(pcm_bytes)
        segments = await self._client.transcribe(wav_bytes, sample_rate=SAMPLE_RATE)

        logger.info(
            "[STT] got %d segment(s) | tokens used: %d",
            len(segments),
            self._client.last_usage_tokens,
        )
        for seg in segments:
            logger.info(
                "[STT]   %s  %dms–%dms  %r",
                seg.speaker_label, seg.start_ms, seg.end_ms,
                seg.text[:60] + ("…" if len(seg.text) > 60 else ""),
            )

        return segments


# ── Pipecat frame processor wrapper ──────────────────────────────────────────

try:
    from pipecat.frames.frames import AudioRawFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class STTFrameProcessor(FrameProcessor):
        """
        Wraps STTProcessor for use inside the Pipecat pipeline.
        Listens for AudioRawFrame blobs (complete utterances from VAD),
        transcribes them, and pushes results downstream as a custom frame.
        """

        def __init__(self, client: STTClient | None = None, **kwargs):
            super().__init__(**kwargs)
            self._stt = STTProcessor(client=client)

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
                segments = await self._stt.transcribe(frame.audio)
                # Downstream stages receive the segments via a custom attribute on the frame.
                # Full custom frame type will be added when Pipecat integration is wired up.
                frame.stt_segments = segments
                await self.push_frame(frame, direction)
                return

            await self.push_frame(frame, direction)

except ImportError:
    pass
