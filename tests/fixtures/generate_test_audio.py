"""
Generate a synthetic test WAV at tests/fixtures/input.wav for use in all block tests.

Pattern: 1s silence → 3s 440Hz tone → 5s silence → 2s 880Hz tone → 4s silence
Total: 15 seconds, 16kHz mono int16.

Usage:
    python tests/fixtures/generate_test_audio.py
"""
import struct
import sys
import wave
import math
from pathlib import Path

OUTPUT = Path(__file__).parent / "input.wav"
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16


def silence(duration_s: float) -> bytes:
    n = int(SAMPLE_RATE * duration_s)
    return struct.pack(f"<{n}h", *([0] * n))


def tone(duration_s: float, freq_hz: float, amplitude: float = 0.2) -> bytes:
    """Generate a sine tone. amplitude=0.2 means 20% of max (6553 out of 32767)."""
    n = int(SAMPLE_RATE * duration_s)
    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * freq_hz * i / SAMPLE_RATE))
        for i in range(n)
    ]
    return struct.pack(f"<{n}h", *samples)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pcm = (
        silence(1.0)     # 1s silence
        + tone(3.0, 440) # 3s speech-like tone (passes 1.5s gate)
        + silence(5.0)   # 5s silence (triggers STOP)
        + tone(2.0, 880) # 2s second utterance (passes 1.5s gate)
        + silence(4.0)   # trailing silence
    )

    with wave.open(str(OUTPUT), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)

    total_s = len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    print(f"Generated: {OUTPUT}")
    print(f"Duration: {total_s:.1f}s  Sample rate: {SAMPLE_RATE}Hz  Channels: {CHANNELS}")
    print("Pattern: 1s silence | 3s 440Hz tone | 5s silence | 2s 880Hz tone | 4s silence")
    print("\nNote: This is a synthetic test file.")
    print("For STT/speaker tests, replace tests/fixtures/input.wav with a real voice recording.")


if __name__ == "__main__":
    main()
