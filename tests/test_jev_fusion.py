import pytest

from extension.jev_fusion import choose_mapping, j1_alphas, right_direction, to_dat_score, verdict


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
