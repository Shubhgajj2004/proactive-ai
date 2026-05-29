"""
Block 4: Audio Segmenter + Resampler test.

Reads tests/fixtures/input.wav + tests/fixtures/stt_output.json,
slices the audio by speaker timestamps, and saves one WAV per speaker
to tests/output/ so you can listen and verify.

Also runs an overlap test with synthetic timestamps.

Usage:
    python tests/test_segmenter.py
    python tests/test_segmenter.py --audio tests/fixtures/input.wav
"""
import argparse
import io
import json
import logging
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from server.pipeline.audio_segmenter import segment_audio, AudioSegment
from server.pipeline.audio_resampler import TARGET_RATE
from server.stt.client import STTSegment

SAMPLE_RATE   = 16000
SAMPLE_WIDTH  = 2
FIXTURE_PATH  = Path("tests/fixtures/stt_output.json")
OUTPUT_DIR    = Path("tests/output")


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_wav_as_pcm(path: Path) -> tuple[bytes, int]:
    """Load WAV → (int16 PCM bytes, sample_rate). Resamples to mono."""
    import torch, torchaudio
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    pcm = (waveform.squeeze().numpy() * 32767).astype("int16").tobytes()
    return pcm, sr


def save_wav(pcm_bytes: bytes, path: Path, sample_rate: int = TARGET_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    path.write_bytes(buf.getvalue())


def pcm_duration_s(pcm_bytes: bytes, sample_rate: int = TARGET_RATE) -> float:
    return len(pcm_bytes) / (sample_rate * SAMPLE_WIDTH)


# ── Main test ─────────────────────────────────────────────────────────────────

def test_file(audio_path: Path) -> bool:
    print(f"\n{'='*60}")
    print(f"  File mode: {audio_path.name}")
    print(f"{'='*60}")

    if not audio_path.exists():
        print(f"[SEGMENTER] ERROR: {audio_path} not found.")
        return False
    if not FIXTURE_PATH.exists():
        print(f"[SEGMENTER] ERROR: STT fixture not found at {FIXTURE_PATH}")
        print("  Run: python tests/test_stt.py --real-api --save-fixture")
        return False

    pcm, sr = load_wav_as_pcm(audio_path)
    total_s = len(pcm) / (sr * SAMPLE_WIDTH)
    print(f"\nAudio : {total_s:.1f}s @ {sr}Hz  ({len(pcm)} bytes PCM)")

    with open(FIXTURE_PATH) as f:
        stt_data = json.load(f)
    segments = [STTSegment(**s) for s in stt_data]
    print(f"STT segments: {len(segments)}")

    audio_segments = segment_audio(pcm, sr, segments)

    print(f"\nSliced segments: {len(audio_segments)}")
    ok = True
    for i, seg in enumerate(audio_segments):
        dur = pcm_duration_s(seg.pcm_bytes)
        expected_dur = (seg.end_ms - seg.start_ms) / 1000
        diff_ms = abs(dur - expected_dur) * 1000

        status = "✓" if diff_ms < 100 else "⚠"
        if diff_ms >= 100:
            ok = False

        print(f"  [{i+1}] {seg.speaker_label:<12}  {seg.start_ms}ms–{seg.end_ms}ms"
              f"  expected={expected_dur:.2f}s  actual={dur:.2f}s  diff={diff_ms:.0f}ms  {status}")
        print(f"       {seg.text[:80]}")

        out_path = OUTPUT_DIR / f"seg_{i+1}_{seg.speaker_label.replace(' ', '')}.wav"
        save_wav(seg.pcm_bytes, out_path)
        print(f"       Saved → {out_path}")

    return ok


def test_overlap() -> bool:
    """Synthetic overlap test: two segments that share a 200ms region."""
    print(f"\n{'='*60}")
    print("  Overlap resolution test")
    print(f"{'='*60}")

    # 5 seconds of audio: loud tone for first 3s, quieter for last 2s
    sr = SAMPLE_RATE
    t = np.linspace(0, 5, 5 * sr, endpoint=False)

    loud   = (np.sin(2 * np.pi * 440 * t[:3*sr]) * 0.8 * 32767).astype(np.int16)
    quiet  = (np.sin(2 * np.pi * 440 * t[:2*sr]) * 0.1 * 32767).astype(np.int16)
    pcm    = np.concatenate([loud, quiet]).tobytes()

    # Segments overlap: seg0 ends at 3100ms, seg1 starts at 2900ms (200ms overlap)
    segments = [
        STTSegment(start_ms=0,    end_ms=3100, speaker_label="Speaker A", text="loud part"),
        STTSegment(start_ms=2900, end_ms=5000, speaker_label="Speaker B", text="quiet part"),
    ]

    print("\nInput: Speaker A (0–3100ms, loud) overlaps Speaker B (2900–5000ms, quiet) by 200ms")
    audio_segs = segment_audio(pcm, sr, segments)

    # Speaker A had higher RMS in overlap → should keep it; Speaker B start pushed to 3100ms
    seg_a = audio_segs[0]
    seg_b = audio_segs[1]

    # After overlap resolution, Speaker B should start at or after 3100ms
    b_start_samples = len(np.frombuffer(seg_a.pcm_bytes, dtype=np.int16))
    expected_b_dur_s = (5000 - seg_a.end_ms) / 1000

    ok_a = seg_a.speaker_label == "Speaker A"
    ok_b = seg_b.speaker_label == "Speaker B"
    # B's duration should be shorter than original (overlap trimmed)
    ok_trim = pcm_duration_s(seg_b.pcm_bytes) < (5000 - 2900) / 1000

    print(f"\n  Speaker A: {seg_a.start_ms}ms–{seg_a.end_ms}ms  dur={pcm_duration_s(seg_a.pcm_bytes):.2f}s")
    print(f"  Speaker B: {seg_b.start_ms}ms–{seg_b.end_ms}ms  dur={pcm_duration_s(seg_b.pcm_bytes):.2f}s")
    print(f"  Speaker B trimmed (overlap given to louder A): {'✓' if ok_trim else '✗'}")

    ok = ok_a and ok_b and ok_trim
    print(f"\n  {'✓ PASS' if ok else '✗ FAIL'} — overlap assigned to higher-RMS segment")
    return ok


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Block 4: Audio Segmenter test")
    parser.add_argument("--audio", default="tests/fixtures/input.wav")
    args = parser.parse_args()

    results = []

    ok_file = test_file(Path(args.audio))
    results.append(("File segmentation", ok_file))

    ok_overlap = test_overlap()
    results.append(("Overlap resolution", ok_overlap))

    print(f"\n{'='*60}")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
