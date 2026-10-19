"""
Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) as a
zero-judge baseline for RQ3: score(d) = sum over retrievers of
1 / (k + rank(d)), with the standard k = 60, not tuned.

RRF uses only rank positions, so it does not depend on min-max score
normalization or on any alpha. Both retrievers rank the full corpus, so no
paragraph is ever missing from a list. Ties are broken by corpus order,
as in the rest of the evaluation.

    python -m extension.rrf

Evaluates RRF on tune and test, and compares it on test with fixed alpha
0.6, DAT (GPT-4o-mini), Jev J1 and Jev J3 on the same questions.
Writes results/rrf_test.csv.
"""

from pathlib import Path

import numpy as np

from baseline.dat_fusion import fuse
from baseline.dat_judge import dat_alphas
from baseline.evaluate import (
    gold_ranks,
    hybrid_sensitive,
    load_normalized,
    mrr_at_k,
    paired_bootstrap_ci,
    precision_at_1,
    ranks_over_grid,
)
from baseline.retrieve import load_scores
from extension.jev_fusion import (
    CHOSEN_J3_OPTION,
    CHOSEN_MAPPING,
    j1_alphas,
    load_judgments,
    multi_alphas,
    write_csv,
)

ROOT = Path(__file__).resolve().parent.parent
RRF_K = 60


def ranks_from_scores(scores):
    """1-based rank of every document for every query, best first, ties
    broken by corpus order (stable sort)."""
    scores = np.asarray(scores)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1)
    return ranks


def rrf_scores(dense, bm25, k=RRF_K):
    return 1.0 / (k + ranks_from_scores(dense)) + 1.0 / (k + ranks_from_scores(bm25))


def gold_ranks_by_method(set_name, alphas):
    """Gold rank per question for RRF and for each alpha-based method."""
    norm_dense, norm_bm25, gold_index = load_normalized(set_name)
    bm25, dense = load_scores(set_name)
    ranks = {"RRF (k=60)": gold_ranks(rrf_scores(dense, bm25), gold_index)}
    for name, alpha in alphas.items():
        ranks[name] = gold_ranks(fuse(norm_dense, norm_bm25, alpha), gold_index)
    hybrid = hybrid_sensitive(ranks_over_grid(norm_dense, norm_bm25, gold_index))
    return ranks, hybrid


def summary_rows(set_name, ranks, hybrid):
    return [{
        "set": set_name, "method": name, "n_queries": len(r),
        "p_at_1": precision_at_1(r), "mrr_at_20": mrr_at_k(r),
        "hybrid_p_at_1": precision_at_1(r[hybrid]), "hybrid_mrr_at_20": mrr_at_k(r[hybrid]),
        "n_hybrid": int(hybrid.sum()),
    } for name, r in ranks.items()]


def print_rows(rows, title):
    print(f"\n== {title} ==")
    print(f"{'method':<34} {'P@1':>7} {'MRR@20':>7} {'hybrid P@1':>10}")
    for r in rows:
        print(f"{r['method']:<34} {r['p_at_1']:>7.4f} {r['mrr_at_20']:>7.4f} {r['hybrid_p_at_1']:>10.4f}")


def print_paired(name, ranks_a, ranks_b):
    def reciprocal(r):
        return np.where(r <= 20, 1.0 / r, 0.0)

    p1 = paired_bootstrap_ci(ranks_a == 1, ranks_b == 1)
    mrr = paired_bootstrap_ci(reciprocal(ranks_a), reciprocal(ranks_b))
    print(f"  {name:<32} P@1 {p1[0]:+.4f} [{p1[1]:+.4f}, {p1[2]:+.4f}]   "
          f"MRR@20 {mrr[0]:+.4f} [{mrr[1]:+.4f}, {mrr[2]:+.4f}]   "
          f"rank 1 gained {int(((ranks_a == 1) & (ranks_b != 1)).sum())}, lost {int(((ranks_a != 1) & (ranks_b == 1)).sum())}")


def main():
    tune_ranks, tune_hybrid = gold_ranks_by_method("tune", {"Fixed hybrid (alpha=0.6)": 0.6})
    print_rows(summary_rows("tune", tune_ranks, tune_hybrid), "SQuAD tune, 2865 questions (reference)")

    alphas = {
        "Fixed hybrid (alpha=0.6)": 0.6,
        "DAT (GPT-4o-mini)": np.array([row["alpha"] for row in dat_alphas("test")]),
        f"Jev J1 ({CHOSEN_MAPPING})": j1_alphas(load_judgments("test", 1), CHOSEN_MAPPING),
        f"Jev J3 ({CHOSEN_J3_OPTION})": multi_alphas(load_judgments("test", 3), CHOSEN_J3_OPTION),
    }
    ranks, hybrid = gold_ranks_by_method("test", alphas)
    rows = summary_rows("test", ranks, hybrid)
    print_rows(rows, "SQuAD test, 2996 questions")
    write_csv(rows, ROOT / "results" / "rrf_test.csv")

    print("\n== Each method minus RRF, paired on the same test questions (95% CI, 10,000 resamples) ==")
    for name in alphas:
        print_paired(name, ranks[name], ranks["RRF (k=60)"])


if __name__ == "__main__":
    main()
