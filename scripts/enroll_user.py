"""
Enroll a user's voiceprint into PostgreSQL.

Reads one or more WAV files, extracts 256-dim WeSpeaker d-vectors,
averages them (multiple clips = more stable enrollment), and upserts
into user_voiceprints.

Usage (single clip):
    python scripts/enroll_user.py --user-id alice --audio tests/fixtures/wearer.wav

Usage (multiple clips — recommended for production):
    python scripts/enroll_user.py --user-id alice \
        --audio clip1.wav clip2.wav clip3.wav
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


async def enroll(user_id: str, audio_paths: list[Path]) -> None:
    import torchaudio
    from server.pipeline.speaker_embedder import extract_embedding, EMBEDDING_DIM

    print(f"\nEnrolling user : {user_id}")
    print(f"Audio clips    : {[p.name for p in audio_paths]}")

    embeddings = []
    for path in audio_paths:
        if not path.exists():
            print(f"ERROR: {path} not found.")
            sys.exit(1)

        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        pcm = (waveform.squeeze().numpy() * 32767).astype("int16").tobytes()

        duration_s = len(pcm) / (16000 * 2) if sr == 16000 else waveform.shape[1] / sr
        print(f"\n  {path.name}  ({duration_s:.1f}s) — extracting d-vector…")

        t0 = time.perf_counter()
        emb = await extract_embedding(pcm, src_rate=sr)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if emb is None:
            print(f"  WARNING: {path.name} was too quiet/short — skipped")
            continue

        print(f"  shape={emb.shape}  norm={float(np.linalg.norm(emb)):.4f}  time={elapsed_ms:.0f}ms")
        embeddings.append(emb)

    if not embeddings:
        print("\nERROR: No valid embeddings extracted.")
        sys.exit(1)

    # Average + re-normalize (multi-clip enrollment)
    if len(embeddings) > 1:
        mean_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        d_vector = mean_emb / norm if norm > 1e-9 else mean_emb
        print(f"\nAveraged {len(embeddings)} clips → re-normalized d-vector")
    else:
        d_vector = embeddings[0]

    print(f"Final d-vector : shape={d_vector.shape}  norm={float(np.linalg.norm(d_vector)):.4f}")

    # Save to PostgreSQL
    import asyncpg
    from server.config import settings

    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        await conn.execute(
            """
            INSERT INTO user_voiceprints (user_id, d_vector)
            VALUES ($1, $2::vector)
            ON CONFLICT (user_id)
            DO UPDATE SET d_vector = EXCLUDED.d_vector, enrolled_at = now()
            """,
            user_id,
            "[" + ",".join(f"{x:.6f}" for x in d_vector.tolist()) + "]",
        )
        print(f"\n✓ Enrolled '{user_id}' → user_voiceprints ({EMBEDDING_DIM}-dim d-vector)")
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="Enroll user voiceprint")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--audio", nargs="+", required=True, help="One or more WAV files")
    args = parser.parse_args()
    asyncio.run(enroll(args.user_id, [Path(p) for p in args.audio]))


if __name__ == "__main__":
    main()
