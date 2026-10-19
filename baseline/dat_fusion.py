"""
DAT's score fusion (paper §3-4), shared by every weight-based method so
the baselines, DAT and the Jev extension all fuse scores identically.
"""

import numpy as np


def minmax(scores):
    """Per-query min-max normalization to [0, 1] (paper Eq. 3-4).

    `scores` is (queries, documents). A row whose scores are all equal
    becomes all zeros, since it carries no ranking signal.
    """
    scores = np.asarray(scores, dtype=np.float64)
    low = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - low
    safe_span = np.where(span > 0, span, 1.0)
    return np.where(span > 0, (scores - low) / safe_span, 0.0)


def fuse(norm_dense, norm_bm25, alpha):
    """R(q, d) = alpha * S_dense + (1 - alpha) * S_bm25 (paper Eq. 5 / 7).

    `alpha` is a scalar (fixed hybrid) or one value per query (DAT).
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    return alpha * norm_dense + (1.0 - alpha) * norm_bm25


def dat_alpha(score_dense, score_bm25):
    """DAT's dynamic alpha from two 0-5 LLM scores (paper Eq. 6), rounded to
    one decimal as in §4.2. Python's round() is round-half-to-even, so
    (1, 3) gives 0.25 -> 0.2; the paper does not specify half rounding.
    """
    if not (0 <= score_dense <= 5 and 0 <= score_bm25 <= 5):
        raise ValueError(f"DAT scores must be in 0..5, got {score_dense}, {score_bm25}")
    if score_dense == 0 and score_bm25 == 0:
        alpha = 0.5
    elif score_dense == 5 and score_bm25 != 5:
        alpha = 1.0
    elif score_bm25 == 5 and score_dense != 5:
        alpha = 0.0
    else:
        alpha = score_dense / (score_dense + score_bm25)
    return round(alpha, 1)
