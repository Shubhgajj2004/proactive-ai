"""
Speaker Embedder — WeSpeaker ResNet34-LM via ONNX.

256-dim L2-normalized speaker embeddings.
Model: Wespeaker/wespeaker-voxceleb-resnet34-LM (ONNX, ~25MB)
Cached at: ~/.cache/wespeaker/resnet34_LM.onnx

CPU-bound inference is offloaded via run_in_executor so it never blocks
the asyncio event loop.

Key design:
  - Accepts raw int16 PCM bytes (from VadProcessor/AudioSegmenter)
  - Internally converts to float32, extracts 80-dim Kaldi Mel-filterbank
  - L2-normalizes output → cosine similarity = dot product
  - Energy gate: skips silent segments (returns None)
"""
import asyncio
import logging
import os
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH     = os.path.join(os.path.expanduser("~"), ".cache", "wespeaker", "resnet34_LM.onnx")
SAMPLE_RATE    = 16000
EMBEDDING_DIM  = 256
ENERGY_THRESH  = 0.01   # RMS threshold — skip silent/noise segments
MIN_DURATION_S = 1.5    # minimum segment length for reliable embedding


@lru_cache(maxsize=1)
def _load_model():
    """Load WeSpeaker ONNX model once and cache for the process lifetime."""
    import onnxruntime as ort

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"WeSpeaker ONNX model not found at {MODEL_PATH}.\n"
            "Run: python scripts/download_models.py"
        )

    logger.info("[EMBEDDER] loading WeSpeaker ResNet34-LM from %s", MODEL_PATH)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info("[EMBEDDER] model loaded — input: %s", sess.get_inputs()[0].name)
    return sess


def _extract_fbank(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Extract 80-dim log Mel-filterbank features (WeSpeaker training config).

    Args:
        wav: float32 mono audio array, normalised to [-1, 1]
        sample_rate: must be 16000

    Returns:
        [T, 80] float32 array where T ≈ duration_ms / 10
    """
    from kaldi_native_fbank import OnlineFbank, FbankOptions

    opts = FbankOptions()
    opts.frame_opts.samp_freq      = float(sample_rate)
    opts.frame_opts.dither         = 0.0
    opts.frame_opts.window_type    = "hamming"
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms  = 10.0
    opts.frame_opts.snip_edges      = True
    opts.mel_opts.num_bins          = 80

    bank = OnlineFbank(opts)
    bank.accept_waveform(float(sample_rate), wav.tolist())
    bank.input_finished()

    n_frames = bank.num_frames_ready
    if n_frames == 0:
        raise ValueError("Fbank extraction produced 0 frames — audio may be silent.")

    return np.array([bank.get_frame(i) for i in range(n_frames)], dtype=np.float32)


def _extract_sync(pcm_bytes: bytes, src_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """
    Synchronous embedding extraction. Runs on a thread pool.

    Args:
        pcm_bytes: int16 PCM bytes
        src_rate:  sample rate of pcm_bytes (resampled to 16kHz if needed)

    Returns:
        (256,) float32 L2-normalized array, or None if segment is too quiet/short.
    """
    # Convert int16 PCM → float32 [-1, 1]
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if src_rate != SAMPLE_RATE:
        import resampy
        samples = resampy.resample(samples, src_rate, SAMPLE_RATE, filter="kaiser_fast")

    # Duration check
    duration_s = len(samples) / SAMPLE_RATE
    if duration_s < MIN_DURATION_S:
        logger.info("[EMBEDDER] segment %.2fs < %.1fs minimum — skipping", duration_s, MIN_DURATION_S)
        return None

    # Energy gate
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < ENERGY_THRESH:
        logger.info("[EMBEDDER] segment RMS=%.4f below threshold — skipping (silence)", rms)
        return None

    # Extract Kaldi Mel-filterbank features
    feat = _extract_fbank(samples)   # [T, 80]
    feat = feat[np.newaxis, ...]     # [1, T, 80]

    # ONNX inference
    sess = _load_model()
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: feat})[0]

    # Pool over time if model returns frame-level output
    if out.ndim == 3:
        out = out.mean(axis=1)       # [1, D]
    emb = out[0]                     # [D]

    # L2 normalize
    norm = np.linalg.norm(emb)
    if norm < 1e-9:
        logger.warning("[EMBEDDER] near-zero norm — audio may be silent")
        return None
    emb = emb / norm

    logger.info(
        "[EMBEDDER] d-vector: shape=%s  norm=%.4f  rms=%.4f  dur=%.2fs",
        emb.shape, float(np.linalg.norm(emb)), rms, duration_s,
    )
    return emb


async def extract_embedding(pcm_bytes: bytes, src_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """
    Async wrapper — runs WeSpeaker inference on the default thread pool
    so it never blocks the asyncio event loop.

    Returns a (256,) L2-normalized float32 array, or None if the segment
    was too quiet or too short to embed reliably.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_sync, pcm_bytes, src_rate)
