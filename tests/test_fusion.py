import numpy as np
import pytest

from baseline.dat_fusion import dat_alpha, fuse, minmax


@pytest.mark.parametrize(
    "dense, bm25, expected",
    [
        (0, 0, 0.5),  # both fail -> equal weight
        (5, 3, 1.0),  # dense direct hit only
        (2, 5, 0.0),  # BM25 direct hit only
        (5, 5, 0.5),  # both direct hits -> proportional
        (3, 2, 0.6),  # paper §6.1 example
        (3, 4, 0.4),  # paper §6.2 example: 0.43 -> 0.4
        (3, 0, 1.0),
        (0, 3, 0.0),
        (1, 3, 0.2),  # 0.25, round-half-to-even
        (3, 1, 0.8),  # 0.75, round-half-to-even
    ],
)
def test_dat_alpha_cases(dense, bm25, expected):
    assert dat_alpha(dense, bm25) == expected


def test_dat_alpha_rejects_out_of_range():
    with pytest.raises(ValueError):
        dat_alpha(6, 0)


def test_minmax_scales_each_row_and_zeroes_constant_rows():
    out = minmax([[2.0, 4.0, 6.0], [3.0, 3.0, 3.0]])
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])


def test_fuse_scalar_and_per_query_alpha():
    dense = np.array([[1.0, 0.0], [1.0, 0.0]])
    bm25 = np.array([[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(fuse(dense, bm25, 0.6), [[0.6, 0.4], [0.6, 0.4]])
    np.testing.assert_allclose(fuse(dense, bm25, [1.0, 0.0]), [[1.0, 0.0], [0.0, 1.0]])
