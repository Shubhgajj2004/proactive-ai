"""
Audio gain amplifier — FIRST stage in the pipeline.

Hardware mics on wearables typically capture at very low volume.
This stage multiplies PCM int16 samples by a gain factor and hard-clips
at the int16 range to prevent wrap-around distortion.

Usage in Pipecat pipeline:
    Insert GainAmplifierProcessor before SileroVADAnalyzer in session_pipeline.py.
"""
import numpy as np


def amplify(pcm_bytes: bytes, gain: float) -> bytes:
    """
    Amplify int16 PCM audio in-place.

    Args:
        pcm_bytes: Raw PCM bytes (int16, any sample rate, mono or stereo).
        gain:      Multiplication factor (e.g. 5.0 for ×5).

    Returns:
        Amplified PCM bytes, clipped to [-32768, 32767]. No wrap-around.
    """
    if not pcm_bytes:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    samples *= gain
    samples = np.clip(samples, -32768, 32767)
    return samples.astype(np.int16).tobytes()


# ── Pipecat frame processor wrapper ──────────────────────────────────────────

try:
    from pipecat.frames.frames import AudioRawFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class GainAmplifierProcessor(FrameProcessor):
        """Pipecat processor: amplifies every AudioRawFrame before VAD."""

        def __init__(self, gain: float = 5.0):
            super().__init__()
            self._gain = gain

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
                amplified = amplify(frame.audio, self._gain)
                frame = AudioRawFrame(audio=amplified, sample_rate=frame.sample_rate, num_channels=frame.num_channels)
            await self.push_frame(frame, direction)

except ImportError:
    # Pipecat not installed — standalone amplify() still works for tests
    pass
