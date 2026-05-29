"""
Block 3: STT Provider Abstraction + Gemini STT test.

Two modes:

  Real API — feed tests/fixtures/input.wav through Gemini STT, print segments,
  optionally save as fixture for future mock use:
      python tests/test_stt.py --audio tests/fixtures/input.wav --real-api
      python tests/test_stt.py --audio tests/fixtures/input.wav --real-api --save-fixture

  Mock — read saved fixture, return it without any API call:
      python tests/test_stt.py --mock

CRITICAL CHECK (real-api mode):
  Verify the output contains start_ms and end_ms on every segment.
  If timestamps are missing, stop — the audio segmentation design must be revised.
"""
import argparse
import asyncio
import json
import logging
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

FIXTURE_PATH = Path("tests/fixtures/stt_output.json")
SAMPLE_RATE  = 16000
SAMPLE_WIDTH = 2   # int16


# ── Audio helpers ─────────────────────────────────────────────────────────────

def read_wav_as_pcm(path: Path) -> bytes:
    """Load a WAV (any rate/channels), resample to 16kHz mono int16, return raw PCM bytes."""
    import torch, torchaudio
    waveform, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return (waveform.squeeze().numpy() * 32767).astype("int16").tobytes()


def pcm_to_wav(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


# ── Mock STT client ───────────────────────────────────────────────────────────

class _MockSTTClient:
    """Returns the saved fixture without any API call."""

    def __init__(self, fixture_path: Path):
        if not fixture_path.exists():
            print(f"[STT] ERROR: fixture not found at {fixture_path}")
            print("       Run with --real-api --save-fixture first.")
            sys.exit(1)
        with open(fixture_path) as f:
            self._data = json.load(f)
        self._last_usage_tokens = 0

    @property
    def last_usage_tokens(self) -> int:
        return self._last_usage_tokens

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000):
        from server.stt.client import STTSegment
        return [STTSegment(**item) for item in self._data]


# ── Test modes ────────────────────────────────────────────────────────────────

async def test_real_api(audio_path: Path, save_fixture: bool) -> None:
    print(f"\n{'='*60}")
    print(f"  Real API mode: {audio_path.name}")
    print(f"{'='*60}")

    if not audio_path.exists():
        print(f"[STT] ERROR: {audio_path} not found.")
        sys.exit(1)

    from server.pipeline.stt_processor import STTProcessor
    from server.stt.factory import make_stt_client

    pcm = read_wav_as_pcm(audio_path)
    dur_s = len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)
    print(f"\nAudio duration : {dur_s:.1f}s")
    print(f"Sending to Gemini STT...\n")

    processor = STTProcessor(client=make_stt_client())
    segments  = await processor.transcribe(pcm)

    _print_segments(segments)
    _check_timestamps(segments)

    if save_fixture:
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FIXTURE_PATH, "w") as f:
            json.dump([s.model_dump() for s in segments], f, indent=2, ensure_ascii=False)
        print(f"\nFixture saved → {FIXTURE_PATH}")
        print("(future blocks can use --mock to skip the API call)")


async def test_mock() -> None:
    print(f"\n{'='*60}")
    print("  Mock mode (no API call)")
    print(f"{'='*60}")

    from server.pipeline.stt_processor import STTProcessor

    processor = STTProcessor(client=_MockSTTClient(FIXTURE_PATH))
    # Feed dummy bytes — mock ignores them
    segments  = await processor.transcribe(b"\x00\x00" * SAMPLE_RATE)

    _print_segments(segments)
    _check_timestamps(segments)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_segments(segments) -> None:
    if not segments:
        print("\n⚠  No segments returned.")
        return
    print(f"Segments returned: {len(segments)}\n")
    for i, seg in enumerate(segments):
        print(f"  [{i+1}] {seg.speaker_label}  {seg.start_ms}ms – {seg.end_ms}ms")
        print(f"       {seg.text[:120]}")
        print()


def _check_timestamps(segments) -> None:
    print(f"{'='*60}")
    if not segments:
        print("  ✗ No segments to validate.")
        return

    missing_ts = [s for s in segments if s.start_ms == 0 and s.end_ms == 0]
    has_ts     = [s for s in segments if not (s.start_ms == 0 and s.end_ms == 0)]

    if missing_ts and not has_ts:
        print("  ✗ CRITICAL: No timestamps in any segment.")
        print("    Audio segmentation depends on start_ms/end_ms.")
        print("    Redesign required before proceeding to Block 4.")
    elif missing_ts:
        print(f"  ⚠  {len(missing_ts)}/{len(segments)} segments missing timestamps — check prompt.")
    else:
        print(f"  ✓ All {len(segments)} segments have start_ms + end_ms")
        print(f"  ✓ Schema: start_ms, end_ms, speaker_label, text")
    print(f"{'='*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Block 3: STT test")
    parser.add_argument("--audio",        default="tests/fixtures/input.wav", help="WAV file")
    parser.add_argument("--real-api",     action="store_true", help="Call Gemini STT")
    parser.add_argument("--save-fixture", action="store_true", help="Save output as JSON fixture")
    parser.add_argument("--mock",         action="store_true", help="Use saved fixture, no API call")
    args = parser.parse_args()

    if args.mock:
        asyncio.run(test_mock())
    elif args.real_api:
        asyncio.run(test_real_api(Path(args.audio), save_fixture=args.save_fixture))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
