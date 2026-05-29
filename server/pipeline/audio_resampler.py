"""
Audio Resampler — normalises any PCM blob to 16kHz int16 mono.

Used after audio_segmenter.py so ECAPA-TDNN always receives 16kHz input.
Pure torchaudio — no external APIs.
"""
import logging

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

TARGET_RATE = 16000


def resample_pcm(pcm_bytes: bytes, src_rate: int) -> bytes:
    """
    Resample int16 PCM bytes from src_rate to 16kHz.
    Returns int16 PCM bytes at TARGET_RATE.
    No-op if src_rate == TARGET_RATE.
    """
    if src_rate == TARGET_RATE:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, N)

    resampled = torchaudio.functional.resample(waveform, src_rate, TARGET_RATE)

    out = (resampled.squeeze().numpy() * 32767).astype(np.int16)
    logger.debug(
        "[RESAMPLER] %dHz → %dHz  %d→%d samples",
        src_rate, TARGET_RATE, len(samples), len(out),
    )
    return out.tobytes()
