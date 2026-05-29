"""
Audio Segmenter — fourth pipeline stage (after stt_processor.py).

Slices a raw PCM utterance blob into per-speaker segments using the
start_ms/end_ms timestamps from STT diarization output.

Overlap rule: if two adjacent segments overlap in time, the overlapping
region is assigned to whichever segment has higher RMS energy.

Input:
  pcm_bytes   — int16 PCM of the full utterance (from VadProcessor)
  sample_rate — sample rate of pcm_bytes (typically 16000)
  segments    — list[STTSegment] from GeminiSTTClient

Output:
  list of AudioSegment dataclasses (one per speaker turn), each containing
  the sliced int16 PCM bytes resampled to 16kHz.
"""
import logging
from dataclasses import dataclass

import numpy as np

from server.pipeline.audio_resampler import resample_pcm, TARGET_RATE
from server.stt.client import STTSegment

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """One speaker turn: metadata + raw 16kHz int16 PCM."""
    speaker_label: str
    start_ms: int
    end_ms: int
    text: str
    pcm_bytes: bytes        # 16kHz int16 mono
    sample_rate: int = TARGET_RATE


def _rms(samples: np.ndarray) -> float:
    """Root-mean-square energy of a float32 array."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def _ms_to_samples(ms: int, sample_rate: int) -> int:
    return int(sample_rate * ms / 1000)


def segment_audio(
    pcm_bytes: bytes,
    sample_rate: int,
    segments: list[STTSegment],
) -> list[AudioSegment]:
    """
    Slice pcm_bytes by STT timestamps, resolve overlaps by RMS energy,
    resample each slice to 16kHz.

    Returns one AudioSegment per STTSegment (empty segments are kept with
    zero-length PCM so downstream stages can handle them gracefully).
    """
    if not segments:
        return []

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    total_samples = len(samples)
    results: list[AudioSegment] = []

    for i, seg in enumerate(segments):
        start_s = _ms_to_samples(seg.start_ms, sample_rate)
        end_s   = _ms_to_samples(seg.end_ms,   sample_rate)

        # Clamp to actual audio length
        start_s = max(0, min(start_s, total_samples))
        end_s   = max(start_s, min(end_s, total_samples))

        # ── Overlap resolution with previous segment ──────────────────────
        if results:
            prev        = results[-1]
            prev_end_s  = _ms_to_samples(prev.end_ms, sample_rate)
            overlap_end = min(end_s, prev_end_s)

            if start_s < overlap_end:
                overlap_region = samples[start_s:overlap_end]
                prev_samples   = np.frombuffer(prev.pcm_bytes, dtype=np.int16) \
                                 if prev.pcm_bytes else np.array([], dtype=np.int16)

                # Compare RMS of overlap region in each segment context
                rms_prev = _rms(prev_samples[-len(overlap_region):]) if len(overlap_region) else 0
                rms_curr = _rms(overlap_region)

                if rms_curr > rms_prev:
                    # Current segment wins: trim previous
                    logger.info(
                        "[SEGMENTER] OVERLAP %dms–%dms → %s (higher RMS %.0f > %.0f)",
                        seg.start_ms, prev.end_ms, seg.speaker_label, rms_curr, rms_prev,
                    )
                    trimmed_pcm = resample_pcm(
                        samples[:start_s].tobytes(), sample_rate
                    ) if start_s > 0 else b""
                    results[-1] = AudioSegment(
                        speaker_label=prev.speaker_label,
                        start_ms=prev.start_ms,
                        end_ms=seg.start_ms,
                        text=prev.text,
                        pcm_bytes=trimmed_pcm,
                    )
                else:
                    # Previous segment wins: push current start forward
                    logger.info(
                        "[SEGMENTER] OVERLAP %dms–%dms → %s (higher RMS %.0f > %.0f)",
                        seg.start_ms, prev.end_ms, prev.speaker_label, rms_prev, rms_curr,
                    )
                    start_s = overlap_end

        slice_pcm = samples[start_s:end_s].tobytes()
        resampled = resample_pcm(slice_pcm, sample_rate)

        duration_ms = seg.end_ms - seg.start_ms
        logger.info(
            "[SEGMENTER] %s  %dms–%dms  (%dms)  %d bytes → resampled %d bytes",
            seg.speaker_label, seg.start_ms, seg.end_ms,
            duration_ms, len(slice_pcm), len(resampled),
        )

        results.append(AudioSegment(
            speaker_label=seg.speaker_label,
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            text=seg.text,
            pcm_bytes=resampled,
        ))

    return results
