import numpy as np

from baseline.evaluate import (
    alpha_selection_accuracy,
    gold_ranks,
    hybrid_sensitive,
    mrr_at_k,
    precision_at_1,
)


def test_gold_ranks_break_ties_by_corpus_order():
    scores = np.array([
        [0.9, 0.5, 0.1],  # gold 0 is top
        [0.5, 0.5, 0.1],  # gold 1 tied with earlier doc 0 -> rank 2
        [0.5, 0.5, 0.1],  # gold 0 tied with later doc 1 -> rank 1
        [0.1, 0.2, 0.3],  # gold 0 is last
    ])
    ranks = gold_ranks(scores, np.array([0, 1, 0, 0]))
    assert ranks.tolist() == [1, 2, 1, 3]


def test_precision_and_mrr_with_cutoff():
    ranks = np.array([1, 2, 4, 21])
    assert precision_at_1(ranks) == 0.25
    assert mrr_at_k(ranks, k=20) == (1 + 0.5 + 0.25 + 0) / 4


def test_hybrid_sensitive_means_rank_changes_across_alphas():
    grid = np.array([
        [1, 1, 1],  # always top -> not sensitive
        [3, 3, 3],  # never top, never changes -> not sensitive
        [2, 1, 1],  # alpha matters
        [5, 3, 4],  # alpha matters even though never top
    ])
    assert hybrid_sensitive(grid).tolist() == [False, False, True, True]


def test_alpha_selection_accuracy_uses_best_rank_on_grid():
    alphas = np.array([0.0, 0.5, 1.0])
    grid = np.array([
        [1, 1, 1],  # every alpha optimal
        [2, 1, 3],  # only 0.5 optimal
        [4, 3, 2],  # only 1.0 optimal
    ])
    assert alpha_selection_accuracy(grid, 0.5, alphas) == 2 / 3
    assert alpha_selection_accuracy(grid, [0.0, 0.5, 1.0], alphas) == 1.0
