"""
VAD + Audio Accumulator — second pipeline stage (after audio_gain.py).

Uses silero-vad VADIterator for speech detection.
Layered FSM on top handles:
  - 60s hard force-emit (FORCE_STOP)
  - 1.5s minimum gate (drop utterances too short for STT)

Input:  512-sample int16 PCM chunks (32ms @ 16kHz) from the gain amplifier.
Output: utterance blobs (int16 PCM bytes) emitted when a complete utterance is ready.
        Returns None for every chunk that does not complete an utterance.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512                          # 32ms per chunk @ 16kHz
CHUNK_MS = CHUNK_SAMPLES * 1000 // SAMPLE_RATE  # 32

MIN_SPEECH_MS = 1500                         # drop utterances shorter than this
FORCE_EMIT_MS = 60_000                       # hard cap: force-emit at 60s
SILENCE_STOP_MS = 3000                       # passed to VADIterator directly


# ── Core processor ────────────────────────────────────────────────────────────

@dataclass
class _AccumulatorState:
    active: bool = False
    chunks: list[bytes] = field(default_factory=list)
    duration_ms: int = 0


class VadProcessor:
    """
    Stateful VAD + accumulator.

    Call process_chunk() with exactly CHUNK_SAMPLES int16 samples (1024 bytes).
    When an utterance is complete it returns the full PCM blob; otherwise None.

    Thread-safety: not thread-safe — intended for use in a single asyncio task.

    Args:
        threshold:   Silero speech probability threshold (0–1).
        vad_iterator: Optional injected VAD (for testing). If None, loads silero.
    """

    def __init__(self, threshold: float = 0.5, vad_iterator=None):
        if vad_iterator is not None:
            self._vad = vad_iterator
        else:
            model = load_silero_vad()
            self._vad = VADIterator(
                model,
                sampling_rate=SAMPLE_RATE,
                threshold=threshold,
                min_silence_duration_ms=SILENCE_STOP_MS,
            )
        self._state = _AccumulatorState()

    # ── Public API ────────────────────────────────────────────────────────────

    def process_chunk(self, pcm_int16: bytes) -> bytes | None:
        """
        Feed one 512-sample (1024-byte) int16 chunk.
        Returns a complete utterance blob when one is ready, else None.
        """
        audio_f32 = _to_float32(pcm_int16)
        event = self._vad(audio_f32)

        if event and "start" in event:
            self._on_speech_start(pcm_int16)

        elif self._state.active:
            self._accumulate(pcm_int16)

            # 60s hard cap — force emit regardless of VAD state
            if self._state.duration_ms >= FORCE_EMIT_MS:
                logger.info("[VAD] FORCE_EMIT at %ds — utterance too long", FORCE_EMIT_MS // 1000)
                return self._flush(forced=True)

        if event and "end" in event and self._state.active:
            return self._flush(forced=False)

        return None

    def reset(self) -> None:
        """Reset VAD and accumulator state (e.g. after pipeline reconnect)."""
        if hasattr(self._vad, "reset_states"):
            self._vad.reset_states()
        self._state = _AccumulatorState()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_speech_start(self, chunk: bytes) -> None:
        if self._state.active:
            return  # already accumulating; ignore duplicate start
        logger.info("[VAD] UTTERANCE_START")
        self._state = _AccumulatorState(active=True)
        self._accumulate(chunk)

    def _accumulate(self, chunk: bytes) -> None:
        self._state.chunks.append(chunk)
        self._state.duration_ms += CHUNK_MS

    def _flush(self, forced: bool) -> bytes | None:
        blob = b"".join(self._state.chunks)
        total_ms = self._state.duration_ms
        self._state = _AccumulatorState()
        if hasattr(self._vad, "reset_states"):
            self._vad.reset_states()

        # The accumulated window includes the trailing silence that triggered END.
        # Subtract it to get approximate speech-only duration for the gate check.
        speech_ms = max(0, total_ms - SILENCE_STOP_MS)

        if not forced and speech_ms < MIN_SPEECH_MS:
            logger.info(
                "[VAD] DROP: speech ~%dms < %dms minimum — skipping STT",
                speech_ms, MIN_SPEECH_MS,
            )
            return None

        logger.info("[VAD] UTTERANCE_STOP — emitting %.1fs chunk", total_ms / 1000)
        return blob


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_float32(pcm_int16: bytes) -> torch.Tensor:
    """Convert int16 PCM bytes to float32 tensor normalised to [-1, 1]."""
    samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return torch.from_numpy(samples)


# ── Pipecat frame processor wrapper ──────────────────────────────────────────

try:
    from pipecat.frames.frames import AudioRawFrame, UserStartedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class VADFrameProcessor(FrameProcessor):
        """
        Wraps VadProcessor for use inside the Pipecat pipeline.
        Emits UserStartedSpeakingFrame on speech start and passes
        complete utterance blobs downstream as AudioRawFrame.
        """

        def __init__(self, threshold: float = 0.5, **kwargs):
            super().__init__(**kwargs)
            self._vad = VadProcessor(threshold=threshold)

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
                utterance = self._vad.process_chunk(frame.audio)
                if utterance:
                    utterance_frame = AudioRawFrame(
                        audio=utterance,
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                    )
                    await self.push_frame(utterance_frame, direction)
                return  # don't forward raw chunks downstream

            await self.push_frame(frame, direction)

except ImportError:
    pass
