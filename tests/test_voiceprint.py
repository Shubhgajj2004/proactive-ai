"""
Block 5: Speaker Embedder + Voiceprint Matcher test.

Enrollment:  one or more WAV clips of the wearer (averaged for stability)
Segments:    tests/output/seg_*.wav — per-speaker clips from Block 4

Usage:
    # Single clip
    python tests/test_voiceprint.py --wearer tests/fixtures/wearer.wav

    # Multiple clips (averaged enrollment — more stable)
    python tests/test_voiceprint.py \
        --wearer tests/fixtures/clip1.wav tests/fixtures/clip2.wav tests/fixtures/clip3.wav
"""
import argparse
import asyncio
import logging
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from server.pipeline.speaker_embedder import extract_embedding
from server.pipeline.voiceprint_matcher import identify_wearer, MIN_WEARER_SIM

OUTPUT_DIR   = Path("tests/output")
WEARER_WAV   = Path("tests/fixtures/wearer.wav")


# ── Audio helper ──────────────────────────────────────────────────────────────

def load_wav_pcm(path: Path) -> bytes:
    """Load a WAV file, return raw int16 PCM bytes resampled to 16kHz mono."""
    import torchaudio
    waveform, sr = torchaudio.load(str(path))
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return (waveform.squeeze().numpy() * 32767).astype("int16").tobytes()


# ── Main test ─────────────────────────────────────────────────────────────────

async def run_tests(wearer_wavs: list[Path]) -> None:
    print(f"\n{'='*60}")
    print("  Block 5: Speaker Embedder + Voiceprint Matcher")
    print(f"{'='*60}")

    # Validate inputs
    for p in wearer_wavs:
        if not p.exists():
            print(f"\nERROR: wearer audio not found at {p}")
            sys.exit(1)

    seg_files = sorted(OUTPUT_DIR.glob("seg_*.wav"))
    if not seg_files:
        print("\nERROR: No segment WAVs in tests/output/")
        print("  Run Block 4 first: python tests/test_segmenter.py")
        sys.exit(1)

    # ── Step 1: Extract enrolled wearer d-vector (average if multiple clips) ──
    print(f"\nEnrollment clips: {[p.name for p in wearer_wavs]}")
    print("Extracting wearer d-vector(s)…")

    raw_embeddings = []
    for p in wearer_wavs:
        pcm = load_wav_pcm(p)
        emb = await extract_embedding(pcm)
        if emb is None:
            print(f"  WARNING: {p.name} too quiet/short — skipped")
            continue
        dur_s = len(pcm) / (16000 * 2)
        print(f"  {p.name}: dur={dur_s:.1f}s  norm={np.linalg.norm(emb):.4f}")
        raw_embeddings.append(emb)

    if not raw_embeddings:
        print("ERROR: No valid wearer embeddings.")
        sys.exit(1)

    if len(raw_embeddings) > 1:
        mean = np.mean(raw_embeddings, axis=0)
        enrolled_dvec = mean / (np.linalg.norm(mean) + 1e-9)
        print(f"\nAveraged {len(raw_embeddings)} clips → enrolled d-vector  norm={np.linalg.norm(enrolled_dvec):.4f}")
    else:
        enrolled_dvec = raw_embeddings[0]
        print(f"\nEnrolled d-vector  norm={np.linalg.norm(enrolled_dvec):.4f}")

    # ── Step 2: Extract d-vectors for all segments ────────────────────────────
    print(f"\nSegments found  : {len(seg_files)}")
    print("Extracting segment d-vectors…\n")

    pairs: list[tuple[str, np.ndarray]] = []
    for seg_file in seg_files:
        pcm = load_wav_pcm(seg_file)
        emb = await extract_embedding(pcm)
        pairs.append((seg_file.stem, emb))
        print(f"  {seg_file.name}: shape={emb.shape}  norm={np.linalg.norm(emb):.4f}")

    # ── Step 3: Rank speakers — top scorer = wearer ───────────────────────────
    print(f"\n{'─'*60}")
    print(f"Ranking speakers (top scorer = wearer, MIN_SIM={MIN_WEARER_SIM})…\n")

    results = identify_wearer(pairs, enrolled_dvec)

    print(f"  {'Rank':<6} {'Segment':<25} {'cosine_sim':>10}  Verdict")
    print(f"  {'─'*6}  {'─'*25}  {'─'*10}  {'─'*20}")
    for i, r in enumerate(results):
        verdict = "✓ WEARER" if r.is_wearer == "True" else ("✗ bystander" if r.is_wearer == "False" else "? unknown")
        print(f"  #{i+1:<5} {r.speaker_label:<25} {r.cosine_sim:>10.4f}  {verdict}")

    # ── Step 4: Embedding shape check ─────────────────────────────────────────
    all_256 = all(emb.shape == (256,) for _, emb in pairs if emb is not None)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    wearers    = [r for r in results if r.is_wearer == "True"]
    bystanders = [r for r in results if r.is_wearer == "False"]

    checks = [
        ("Embedding shape (256,)",        all_256),
        ("Wearer identified (top rank)",   len(wearers) >= 1),
        ("Bystanders marked correctly",    len(bystanders) >= 1 if len(seg_files) >= 2 else None),
    ]

    passed = sum(1 for _, ok in checks if ok is True)
    skipped = sum(1 for _, ok in checks if ok is None)
    total = len(checks) - skipped

    for name, ok in checks:
        marker = "✓" if ok is True else ("–" if ok is None else "✗")
        print(f"  {marker} {name}")

    print(f"\n  {passed}/{total} passed" + (f"  ({skipped} skipped)" if skipped else ""))
    print(f"{'='*60}")

    if passed < total:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Block 5: Voiceprint matcher test")
    parser.add_argument(
        "--wearer", nargs="+",
        default=[str(WEARER_WAV)],
        help="One or more WAV files for enrollment (averaged if multiple)",
    )
    args = parser.parse_args()
    asyncio.run(run_tests([Path(p) for p in args.wearer]))


if __name__ == "__main__":
    main()
