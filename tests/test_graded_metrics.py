"""Graded metrics on hand-built examples with values computed by hand."""

import math

import numpy as np
import pytest

from extension.graded_metrics import (
    mrr_at_k,
    ndcg_at_k,
    per_query_metrics,
    precision_at_1,
    recall_at_k,
    summarize,
)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([2, 1, 0, 0], [2, 1]) == pytest.approx(1.0)


def test_ndcg_hand_computed_with_linear_gain():
    # ranked grades 0, 2, 1; judged {2, 1}
    dcg = 2 / math.log2(3) + 1 / math.log2(4)  # 1.26186 + 0.5
    idcg = 2 / math.log2(2) + 1 / math.log2(3)  # 2 + 0.63093
    assert ndcg_at_k([0, 2, 1], [2, 1]) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([0, 2, 1], [2, 1]) == pytest.approx(0.669672, abs=1e-6)  # 1.761860 / 2.630930


def test_ndcg_gain_is_linear_not_exponential():
    # one grade-2 doc at rank 2 and a grade-1 doc at rank 1, judged {2, 1}
    linear = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    exponential = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert ndcg_at_k([1, 2], [2, 1]) == pytest.approx(linear)
    assert ndcg_at_k([1, 2], [2, 1]) != pytest.approx(exponential)


def test_ndcg_ideal_truncated_at_k_when_many_relevant():
    # 12 relevant (grade 1), top 10 all relevant -> nDCG@10 = 1
    assert ndcg_at_k([1] * 10, [1] * 12, k=10) == pytest.approx(1.0)


def test_ndcg_ignores_grade_zero_and_unjudged():
    assert ndcg_at_k([0, 0, 1], [1, 0, 0]) == pytest.approx(1 / math.log2(4))
    assert ndcg_at_k([0, 0], [0]) == 0.0  # nothing relevant judged


def test_recall_counts_all_relevant_in_denominator():
    assert recall_at_k([1] * 10, n_relevant=12) == pytest.approx(10 / 12)
    assert recall_at_k([0, 2, 0, 1], n_relevant=4) == pytest.approx(0.5)
    assert recall_at_k([1, 1], n_relevant=0) == 0.0


def test_mrr_first_relevant_within_k():
    assert mrr_at_k([0, 0, 1, 2]) == pytest.approx(1 / 3)
    assert mrr_at_k([0] * 10 + [1]) == 0.0  # first relevant at rank 11


def test_precision_at_1():
    assert precision_at_1([2, 0]) == 1.0
    assert precision_at_1([0, 2]) == 0.0
    assert precision_at_1([]) == 0.0


def test_per_query_metrics_end_to_end_with_corpus_order_ties():
    doc_ids = ["d1", "d2", "d3", "d4"]
    query_ids = ["q1", "q2"]
    scores = np.array([
        [0.9, 0.5, 0.5, 0.1],  # ranks d1, d2 (tie with d3, earlier in corpus), d3, d4
        [0.1, 0.2, 0.3, 0.4],  # ranks d4, d3, d2, d1
    ])
    qrels = {"q1": {"d3": 2, "d4": 1}, "q2": {"d1": 1}}
    per_query = per_query_metrics(scores, query_ids, doc_ids, qrels)
    # q1 ranked grades [0, 0, 2, 1]
    q1_ndcg = (2 / math.log2(4) + 1 / math.log2(5)) / (2 + 1 / math.log2(3))
    assert per_query["ndcg@10"][0] == pytest.approx(q1_ndcg)
    assert per_query["recall@10"][0] == 1.0
    assert per_query["mrr@10"][0] == pytest.approx(1 / 3)
    assert per_query["p@1"][0] == 0.0
    # q2 ranked grades [0, 0, 0, 1]
    assert per_query["ndcg@10"][1] == pytest.approx(1 / math.log2(5))
    assert per_query["mrr@10"][1] == pytest.approx(1 / 4)
    assert summarize(per_query)["recall@10"] == 1.0


def test_per_query_metrics_requires_judgments():
    with pytest.raises(ValueError):
        per_query_metrics(np.ones((1, 2)), ["q9"], ["d1", "d2"], {"q1": {"d1": 1}})
