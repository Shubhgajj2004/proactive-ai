"""
Block 1: Audio Gain Amplifier test.

Usage:
    python tests/test_audio_gain.py \
        --input  tests/fixtures/input.wav \
        --output tests/output/gain_test.wav \
        --gain   5.0

Drop any WAV at tests/fixtures/input.wav. Output saved to tests/output/gain_test.wav.
Open the output in any audio player — it should sound the same but clearly louder.

Also runs a built-in clipping test on a synthetic signal.
"""
import argparse
import struct
import sys
import wave
from pathlib import Path

import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.pipeline.audio_gain import amplify


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_wav(path: Path) -> tuple[bytes, int, int]:
    """Returns (pcm_bytes, sample_rate, num_channels)."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
        if sample_width != 2:
            raise ValueError(f"Expected int16 (2-byte) WAV, got {sample_width}-byte samples. "
                             "Convert your file: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 input_16k.wav")
        return raw, sample_rate, num_channels


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int, num_channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def peak_amplitude(pcm_bytes: bytes) -> int:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int(np.max(np.abs(samples)))


def peak_pct(peak: int) -> str:
    return f"{peak / 32767 * 100:.1f}%"


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_file(input_path: Path, output_path: Path, gain: float) -> bool:
    if not input_path.exists():
        print(f"\n[GAIN] ERROR: Input file not found: {input_path}")
        print("       Drop any WAV file at tests/fixtures/input.wav and re-run.")
        return False

    pcm, sr, ch = read_wav(input_path)
    in_peak = peak_amplitude(pcm)
    amplified = amplify(pcm, gain)
    out_peak = peak_amplitude(amplified)

    expected_peak = min(int(in_peak * gain), 32767)
    clipped = (in_peak * gain) > 32767

    print(f"\n[GAIN] Input  → peak: {in_peak:6d} / 32767  ({peak_pct(in_peak)} of max)")
    print(f"[GAIN] Gain   → ×{gain}")
    print(f"[GAIN] Output → peak: {out_peak:6d} / 32767  ({peak_pct(out_peak)} of max)", end="")
    if clipped:
        print("  ← hard-clipped (input was near max)")
    else:
        print()

    write_wav(output_path, amplified, sr, ch)
    print(f"[GAIN] Saved: {output_path}")
    print(f"       Sample rate: {sr} Hz  Channels: {ch}")

    # Verify: output peak matches expectation within 1 sample
    ok = abs(out_peak - expected_peak) <= 1
    if ok:
        print(f"[GAIN] ✓ Peak check passed (expected {expected_peak}, got {out_peak})")
    else:
        print(f"[GAIN] ✗ Peak check FAILED (expected {expected_peak}, got {out_peak})")
    return ok


def test_clipping() -> bool:
    print("\n─── Clipping test (synthetic signal, peak=32000) ───")
    # Generate synthetic int16 samples near max (should clip after ×5)
    samples = np.full(512, 32000, dtype=np.int16)
    pcm_in = samples.tobytes()
    in_peak = peak_amplitude(pcm_in)

    pcm_out = amplify(pcm_in, gain=5.0)
    out_peak = peak_amplitude(pcm_out)

    print(f"[GAIN] Input peak: {in_peak}  →  Output peak: {out_peak}  (expected 32767)")

    # Verify no wrap-around: must not be negative or below max
    samples_out = np.frombuffer(pcm_out, dtype=np.int16)
    no_wraparound = bool(np.all(samples_out >= 0))
    capped = out_peak == 32767

    ok = capped and no_wraparound
    if ok:
        print("[GAIN] ✓ Hard-clipped correctly — no wrap-around distortion")
    else:
        print(f"[GAIN] ✗ FAILED — capped={capped}, no_wraparound={no_wraparound}")
        if not no_wraparound:
            print("         Wrap-around detected! Negative values in output.")
    return ok


def test_silence() -> bool:
    print("\n─── Silence test (all-zero input) ───")
    pcm_in = bytes(512)
    pcm_out = amplify(pcm_in, gain=5.0)
    ok = pcm_out == pcm_in
    print(f"[GAIN] {'✓ Silence stays silence' if ok else '✗ FAILED: silence modified'}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Block 1: Audio Gain Amplifier test")
    parser.add_argument("--input",  default="tests/fixtures/input.wav", help="Input WAV path")
    parser.add_argument("--output", default="tests/output/gain_test.wav", help="Output WAV path")
    parser.add_argument("--gain",   type=float, default=5.0, help="Gain factor (default: 5.0)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("  Block 1: Audio Gain Amplifier")
    print("=" * 60)

    results = []
    results.append(("File gain test", test_file(input_path, output_path, args.gain)))
    results.append(("Clipping test",  test_clipping()))
    results.append(("Silence test",   test_silence()))

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{total} passed")
    print("=" * 60)

    if passed == total:
        print("\nNext step: Open tests/output/gain_test.wav and verify it sounds louder.")
        print("Then move on to Block 2: VAD + Audio Accumulator")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
