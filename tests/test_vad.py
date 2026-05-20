"""
Block 2: VAD + Audio Accumulator test.

Three modes:

  File mode (default) — feed tests/fixtures/input.wav through VAD, print events:
      python tests/test_vad.py --audio tests/fixtures/input.wav

  Synthetic tests — built-in scenarios that verify gate and force-emit:
      python tests/test_vad.py --synthetic

  Real-time mic mode — speak into your microphone, see VAD fire live:
      python tests/test_vad.py --mic
      (requires: pip install sounddevice)
"""
import argparse
import logging
import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from server.pipeline.vad_processor import (
    CHUNK_SAMPLES, CHUNK_MS, MIN_SPEECH_MS, SILENCE_STOP_MS, VadProcessor,
)


# ── Energy-based mock VAD (for synthetic tests only) ─────────────────────────
# Silero is trained on real speech — pure sine tones won't trigger it.
# This mock uses RMS energy so the FSM logic can be tested with synthetic audio.

class _EnergyVAD:
    """
    Simple RMS energy detector that mimics VADIterator's event interface:
      {'start': sample}  when energy rises above threshold
      {'end': sample}    after N consecutive silent chunks
      None               otherwise
    """
    def __init__(self, rms_threshold: float = 500.0, silence_chunks: int | None = None):
        self._rms_threshold = rms_threshold
        self._silence_limit = silence_chunks or (SILENCE_STOP_MS // CHUNK_MS)
        self._triggered = False
        self._silent_count = 0
        self._sample = 0

    def __call__(self, audio):
        # Accepts torch.Tensor (float32, normalised [-1,1]) from VadProcessor
        import torch
        if isinstance(audio, torch.Tensor):
            samples = audio.numpy()
        else:
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2))) * 32768  # scale back to int16 range
        self._sample += CHUNK_SAMPLES

        if rms >= self._rms_threshold:
            self._silent_count = 0
            if not self._triggered:
                self._triggered = True
                return {"start": self._sample}
        else:
            if self._triggered:
                self._silent_count += 1
                if self._silent_count >= self._silence_limit:
                    self._triggered = False
                    self._silent_count = 0
                    return {"end": self._sample}
        return None

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE


# ── Audio helpers ─────────────────────────────────────────────────────────────

def read_wav_as_chunks(path: Path) -> list[bytes]:
    """Read a WAV, resample to 16kHz mono int16, return as list of 512-sample chunks."""
    import torchaudio
    import torch
    waveform, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    pcm = (waveform.squeeze().numpy() * 32767).astype("int16").tobytes()
    return [pcm[i:i+CHUNK_BYTES] for i in range(0, len(pcm) - CHUNK_BYTES, CHUNK_BYTES)]


def make_silence(duration_ms: int) -> bytes:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    return b"\x00\x00" * n


def make_tone(duration_ms: int, freq_hz: float = 440, amplitude: float = 0.3) -> bytes:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * freq_hz * i / SAMPLE_RATE))
        for i in range(n)
    ]
    return struct.pack(f"<{n}h", *samples)


def to_chunks(pcm: bytes) -> list[bytes]:
    return [pcm[i:i+CHUNK_BYTES] for i in range(0, len(pcm) - CHUNK_BYTES, CHUNK_BYTES)]


def run_vad(chunks: list[bytes], vad: VadProcessor, pad_silence_ms: int = 0) -> list[bytes]:
    """Feed chunks through VAD, return list of emitted utterance blobs.

    pad_silence_ms: append this many ms of silence after the last chunk so
    trailing speech gets a proper SILENCE_STOP end event (useful for file mode).
    """
    utterances = []
    if pad_silence_ms:
        chunks = list(chunks) + to_chunks(make_silence(pad_silence_ms))
    for chunk in chunks:
        if len(chunk) < CHUNK_BYTES:
            continue
        blob = vad.process_chunk(chunk)
        if blob:
            utterances.append(blob)
    return utterances


def make_mock_vad() -> "_EnergyVAD":
    return _EnergyVAD(rms_threshold=500.0)


# ── Test modes ────────────────────────────────────────────────────────────────

def test_file(audio_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  File mode: {audio_path.name}")
    print(f"{'='*60}")

    if not audio_path.exists():
        print(f"[VAD] ERROR: {audio_path} not found.")
        print("       Run: python tests/fixtures/generate_test_audio.py")
        sys.exit(1)

    chunks = read_wav_as_chunks(audio_path)
    vad = VadProcessor(threshold=0.5)
    # Pad 4s of silence so trailing speech gets a proper SILENCE_STOP end event
    utterances = run_vad(chunks, vad, pad_silence_ms=4000)

    total_s = len(chunks) * CHUNK_BYTES / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    print(f"\nAudio duration : {total_s:.1f}s")
    print(f"Chunks fed     : {len(chunks)}")
    print(f"Utterances out : {len(utterances)}")
    for i, u in enumerate(utterances):
        dur = len(u) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        print(f"  Utterance {i+1}: {dur:.2f}s  ({len(u)} bytes)")

    if utterances:
        print("\n✓ VAD detected and emitted utterances")
    else:
        print("\n⚠  No utterances detected — try a file with speech")


def test_synthetic() -> bool:
    print(f"\n{'='*60}")
    print("  Synthetic tests")
    print(f"{'='*60}")
    results = []

    # ── Test 1: Normal utterance (3s speech → emitted) ────────────────────────
    print("\n[1] Normal utterance (1s silence + 3s speech + 5s silence)")
    pcm = make_silence(1000) + make_tone(3000) + make_silence(5000)
    vad = VadProcessor(vad_iterator=make_mock_vad())
    utterances = run_vad(to_chunks(pcm), vad)
    ok = len(utterances) == 1 and len(utterances[0]) / (SAMPLE_RATE * 2) >= 1.5
    dur = len(utterances[0]) / (SAMPLE_RATE * 2) if utterances else 0
    print(f"   Utterances emitted: {len(utterances)}  duration: {dur:.2f}s")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'} — expected 1 utterance >= 1.5s")
    results.append(("Normal utterance", ok))

    # ── Test 2: Short utterance dropped (0.8s speech < 1.5s gate) ─────────────
    print("\n[2] Short utterance gate (1s silence + 0.8s speech + 5s silence)")
    pcm = make_silence(1000) + make_tone(800) + make_silence(5000)
    vad = VadProcessor(vad_iterator=make_mock_vad())
    utterances = run_vad(to_chunks(pcm), vad)
    ok = len(utterances) == 0
    print(f"   Utterances emitted: {len(utterances)}")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'} — expected 0 (dropped by 1.5s gate)")
    results.append(("Short utterance gate", ok))

    # ── Test 3: Two separate utterances ───────────────────────────────────────
    print("\n[3] Two utterances (speech, long silence, speech)")
    pcm = (make_silence(500) + make_tone(2000)
           + make_silence(5000) + make_tone(2000) + make_silence(5000))
    vad = VadProcessor(vad_iterator=make_mock_vad())
    utterances = run_vad(to_chunks(pcm), vad)
    ok = len(utterances) == 2
    print(f"   Utterances emitted: {len(utterances)}")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'} — expected 2 separate utterances")
    results.append(("Two utterances", ok))

    # ── Test 4: Force-emit at 60s ─────────────────────────────────────────────
    print("\n[4] Force-emit (65s continuous speech → force-emit at 60s)")
    pcm = make_tone(65000)   # 65 seconds of continuous tone
    vad = VadProcessor(vad_iterator=make_mock_vad())
    utterances = run_vad(to_chunks(pcm), vad)
    has_long = any(len(u) / (SAMPLE_RATE * 2) >= 55 for u in utterances)
    ok = len(utterances) >= 1 and has_long
    for i, u in enumerate(utterances):
        print(f"   Utterance {i+1}: {len(u)/(SAMPLE_RATE*2):.1f}s")
    print(f"   {'✓ PASS' if ok else '✗ FAIL'} — expected force-emit ~60s utterance")
    results.append(("Force-emit 60s", ok))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*60}")
    return passed == len(results)


def test_mic() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: uv pip install sounddevice")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Real-time mic mode")
    print("  Speak into your microphone. Ctrl+C to stop.")
    print(f"{'='*60}\n")

    vad = VadProcessor(threshold=0.5)
    buffer = bytearray()

    def callback(indata, frames, time, status):
        nonlocal buffer
        # sounddevice gives float32; convert to int16
        pcm = (indata[:, 0] * 32767).astype("int16").tobytes()
        buffer.extend(pcm)
        while len(buffer) >= CHUNK_BYTES:
            chunk = bytes(buffer[:CHUNK_BYTES])
            buffer = buffer[CHUNK_BYTES:]
            blob = vad.process_chunk(chunk)
            if blob:
                dur = len(blob) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                print(f"[VAD] → Utterance captured: {dur:.2f}s")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=callback,
        ):
            print("Listening... (Ctrl+C to stop)\n")
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Block 2: VAD + Accumulator test")
    parser.add_argument("--audio",     default="tests/fixtures/input.wav", help="WAV file for file mode")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic tests")
    parser.add_argument("--mic",       action="store_true", help="Real-time microphone mode")
    args = parser.parse_args()

    if args.mic:
        test_mic()
    elif args.synthetic:
        ok = test_synthetic()
        sys.exit(0 if ok else 1)
    else:
        test_file(Path(args.audio))


if __name__ == "__main__":
    main()
