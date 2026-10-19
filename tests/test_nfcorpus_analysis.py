import numpy as np
import pytest

from extension.nfcorpus_analysis import direction, rubric_alignment


def test_direction_counts_right_neutral_and_wrong():
    groups = {"only dense top-1 relevant": np.array([True, True, False, False]),
              "only BM25 top-1 relevant": np.array([False, False, True, True])}
    alphas = np.array([0.8, 0.5, 0.2, 0.9])  # right, neutral, right, wrong
    d = direction(alphas, groups)
    n, right, neutral, wrong = d["all one-relevant"]
    assert (n, right, neutral, wrong) == (4, 0.5, 0.25, 0.25)
    assert d["only dense top-1 relevant"][3] == pytest.approx(0.65)  # mean alpha


def test_rubric_alignment_spearman_with_and_without_ties():
    assert rubric_alignment([(0, 0), (4, 1), (5, 2)], "no ties") == pytest.approx(1.0)
    # grades tie at 0 -> average ranks 0.5, 0.5, 2, 3 vs score ranks 0, 1, 2, 3
    assert rubric_alignment([(0, 0), (1, 0), (4, 1), (5, 2)], "tie") == pytest.approx(0.948683, abs=1e-6)
