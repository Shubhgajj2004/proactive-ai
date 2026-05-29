"""
Voiceprint Matcher — identifies which STT speaker is the wearer.

Strategy: relative ranking (not absolute threshold).
  - STT gives Speaker A, B, C ... (arbitrary labels, no identity info)
  - Extract one d-vector per speaker
  - Cosine-similarity each against enrolled wearer d-vector
  - Highest scorer = wearer; all others = bystander

A minimum confidence floor (MIN_WEARER_SIM) guards against utterances
where the wearer is not present at all — if even the top scorer is below
this floor, everyone is marked Unknown.
"""
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# If the top-ranked speaker scores below this, the wearer is probably
# not in this utterance at all (e.g. two bystanders talking).
MIN_WEARER_SIM = 0.35

IsWearer = Literal["True", "False", "Unknown"]


@dataclass
class MatchResult:
    speaker_label: str
    cosine_sim: float
    is_wearer: IsWearer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def identify_wearer(
    embeddings: list[tuple[str, np.ndarray | None]],
    enrolled_d_vector: np.ndarray,
) -> list[MatchResult]:
    """
    Rank all speakers by cosine similarity to enrolled wearer.
    The highest-scoring speaker is marked as the wearer.

    Args:
        embeddings:        [(speaker_label, d_vector | None), ...]
                           None means segment was too quiet to embed.
        enrolled_d_vector: wearer's d-vector from user_voiceprints table.

    Returns:
        list[MatchResult] sorted by cosine_sim descending.
        Exactly one speaker gets is_wearer="True" (the top scorer),
        unless the top score < MIN_WEARER_SIM — then all are Unknown
        (wearer likely absent from this utterance).
    """
    if not embeddings:
        return []

    # Score every speaker
    scored: list[MatchResult] = []
    for label, emb in embeddings:
        if emb is None:
            logger.info("[MATCHER] %s  embedding=None (low energy)", label)
            scored.append(MatchResult(speaker_label=label, cosine_sim=0.0, is_wearer="Unknown"))
        else:
            sim = cosine_similarity(emb, enrolled_d_vector)
            scored.append(MatchResult(speaker_label=label, cosine_sim=sim, is_wearer="False"))
            logger.info("[MATCHER] %s  cosine_sim=%.4f", label, sim)

    # Sort descending
    scored.sort(key=lambda r: r.cosine_sim, reverse=True)

    top = scored[0]
    if top.cosine_sim >= MIN_WEARER_SIM and top.is_wearer != "Unknown":
        # Top scorer = wearer
        scored[0] = MatchResult(
            speaker_label=top.speaker_label,
            cosine_sim=top.cosine_sim,
            is_wearer="True",
        )
        logger.info(
            "[MATCHER] wearer → %s  sim=%.4f  (next best: %.4f)",
            top.speaker_label,
            top.cosine_sim,
            scored[1].cosine_sim if len(scored) > 1 else 0.0,
        )
    else:
        # Nobody clears minimum floor — wearer not in this utterance
        logger.info(
            "[MATCHER] top score %.4f < MIN %.2f — wearer absent",
            top.cosine_sim, MIN_WEARER_SIM,
        )
        scored = [MatchResult(r.speaker_label, r.cosine_sim, "Unknown") for r in scored]

    return scored
