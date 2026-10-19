import pytest

from extension.jev_fusion import (
    choose_mapping,
    choose_option,
    j1_alphas,
    multi_alpha,
    right_direction,
    to_dat_score,
    verdict,
)


def j3(dense, bm25):
    """Build a k=3 judgment from (score, level) pairs per retriever."""
    out = {}
    for method, pairs in (("dense", dense), ("bm25", bm25)):
        for rank, (score, level) in enumerate(pairs, 1):
            out[f"{method}_{rank}"] = {"score": score, "top_level": level}
    return out


def test_option_a_mean_of_levels_then_dat_rule():
    judgment = j3([(4.9, 5), (2.1, 2), (1.0, 1)], [(3.0, 3), (3.2, 3), (2.8, 3)])
    # dense mean 8/3 = 2.67 -> 3, bm25 mean 3 -> 3 -> alpha 0.5
    assert multi_alpha(judgment, "A_mean_dat") == 0.5


def test_option_c_rank_weighting_favours_top1():
    judgment = j3([(4.9, 5), (2.1, 2), (1.0, 1)], [(3.0, 3), (3.2, 3), (2.8, 3)])
    # dense (5 + 2/2 + 1/3) / (11/6) = 3.45 -> 3 vs bm25 3 -> 0.5; now make top-1 dominate more
    assert multi_alpha(judgment, "C_rank_weighted_dat") == 0.5
    judgment = j3([(5.0, 5), (5.0, 5), (1.0, 1)], [(3.0, 3), (3.0, 3), (3.0, 3)])
    # dense (5 + 2.5 + 0.33) / 1.83 = 4.27 -> 4 vs 3 -> 4/7 -> 0.6; plain mean 3.67 -> 4 too
    assert multi_alpha(judgment, "C_rank_weighted_dat") == 0.6


def test_option_b_smooth_ratio_is_continuous_and_symmetric():
    judgment = j3([(5.0, 5)] * 3, [(0.0, 0)] * 3)
    assert multi_alpha(judgment, "B_smooth") == pytest.approx((1 + 0.05) / (1 + 0 + 0.1))
    tie = j3([(2.5, 2)] * 3, [(2.5, 3)] * 3)
    assert multi_alpha(tie, "B_smooth") == pytest.approx(0.5)


def test_options_a_and_c_reduce_to_j1_when_k_is_1():
    judgment = {"dense_1": {"score": 4.36, "top_level": 5}, "bm25_1": {"score": 3.23, "top_level": 4}}
    expected = j1_alphas([judgment], "top_level")[0]
    assert multi_alpha(judgment, "A_mean_dat") == expected == multi_alpha(judgment, "C_rank_weighted_dat")


def test_choose_option_rule():
    rows = {"A_mean_dat": {"p_at_1": 0.84, "mrr_at_20": 0.88},
            "B_smooth": {"p_at_1": 0.85, "mrr_at_20": 0.87},
            "C_rank_weighted_dat": {"p_at_1": 0.84, "mrr_at_20": 0.89}}
    assert choose_option(rows) == "B_smooth"
    rows["B_smooth"]["p_at_1"] = 0.84
    assert choose_option(rows) == "C_rank_weighted_dat"  # P@1 tie -> MRR@20
    rows["C_rank_weighted_dat"]["mrr_at_20"] = 0.88
    rows["B_smooth"]["mrr_at_20"] = 0.88
    assert choose_option(rows) == "A_mean_dat"  # full tie -> A


@pytest.mark.parametrize(
    "ci, expected",
    [
        ((0.010, 0.002, 0.018), "better"),
        ((0.000, -0.008, 0.008), "matches"),
        ((-0.004, -0.008, -0.001), "significantly lower"),
        ((-0.020, -0.030, -0.011), "worse"),
        ((-0.005, -0.015, 0.004), "inconclusive"),
    ],
)
def test_verdict_against_one_point_margin(ci, expected):
    assert expected in verdict(ci, margin=0.01)


def test_right_direction():
    assert right_direction(0.8, dense_correct=True) and right_direction(0.2, dense_correct=False)
    assert not right_direction(0.5, dense_correct=True) and not right_direction(0.5, dense_correct=False)


def judgment(score, top_level):
    return {"score": score, "top_level": top_level}


@pytest.mark.parametrize(
    "score, top_level, mapping, expected",
    [
        (4.36, 5, "top_level", 5),
        (4.36, 5, "round", 4),
        (4.5, 4, "round", 5),  # half rounds up
        (0.49, 0, "round", 0),
        (5.0, 5, "round", 5),
    ],
)
def test_to_dat_score(score, top_level, mapping, expected):
    assert to_dat_score(judgment(score, top_level), mapping) == expected


def test_j1_alphas_apply_dat_rule_to_mapped_scores():
    # The pilot's zinc example: dense 4.36 (level 5), BM25 3.23 (level 4).
    judgments = [{"dense_1": judgment(4.36, 5), "bm25_1": judgment(3.23, 4)}]
    assert j1_alphas(judgments, "top_level").tolist() == [1.0]  # 5 vs 4 -> direct-hit case
    assert j1_alphas(judgments, "round").tolist() == [0.6]  # 4 vs 3 -> 4/7 = 0.57 -> 0.6


def test_choose_mapping_rule():
    rows = {"top_level": {"p_at_1": 0.80, "mrr_at_20": 0.85}, "round": {"p_at_1": 0.81, "mrr_at_20": 0.84}}
    assert choose_mapping(rows) == "round"  # higher P@1 wins
    rows["round"]["p_at_1"] = 0.80
    assert choose_mapping(rows) == "top_level"  # P@1 tie -> higher MRR@20
    rows["round"]["mrr_at_20"] = 0.85
    assert choose_mapping(rows) == "top_level"  # full tie -> top_level
