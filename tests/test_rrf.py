import numpy as np
import pytest

from extension.rrf import ranks_from_scores, rrf_scores


def test_ranks_from_scores_best_first_with_corpus_order_ties():
    scores = np.array([[0.2, 0.9, 0.9, 0.5]])
    assert ranks_from_scores(scores).tolist() == [[4, 1, 2, 3]]


def test_rrf_uses_ranks_not_score_scale():
    dense = np.array([[0.90, 0.80, 0.10]])  # ranks 1, 2, 3
    bm25 = np.array([[1.0, 30.0, 2.0]])  # ranks 3, 1, 2 (raw scale irrelevant)
    expected = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    np.testing.assert_allclose(rrf_scores(dense, bm25), [expected])
    assert int(rrf_scores(dense, bm25).argmax()) == 1  # ranked high by both


def test_rrf_unchanged_by_monotone_rescaling():
    rng = np.random.default_rng(0)
    dense, bm25 = rng.random((3, 10)), rng.random((3, 10)) * 25
    np.testing.assert_allclose(rrf_scores(dense, bm25), rrf_scores(dense * 7 + 1, np.log1p(bm25)))
