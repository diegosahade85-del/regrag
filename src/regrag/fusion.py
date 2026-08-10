"""Combining rankings from retrievers that score on incomparable scales."""

from collections import defaultdict
from collections.abc import Sequence

# The conventional damping constant from Cormack et al. (2009). Large relative
# to the ranks that matter, so the difference between rank 1 and rank 5 is small
# and appearing in both rankings outweighs placing first in one.
DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_K,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse rankings by position, ignoring each retriever's own scores.

    Cosine similarity and ts_rank are not on a common scale — 0.5 from one means
    nothing in terms of the other — so they cannot be averaged, normalised, or
    thresholded against each other in any principled way. Rank position is the
    one thing both produce that *is* comparable, which is what makes this work
    without tuning per-retriever weights.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[chunk_id] += 1.0 / (k + rank)

    # Sort by score, then by id: ties are common with two rankers and an
    # arbitrary order would make results jitter between identical runs.
    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return fused[:limit] if limit is not None else fused
