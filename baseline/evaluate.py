"""
Steps 1.3 / 1.5 of the DAT reproduction: evaluate fusion methods on cached
retrieval scores and put our numbers next to the paper's.

Metrics (paper §5.1):
- Precision@1: gold paragraph ranked first.
- MRR@20: 1 / rank of the gold paragraph if within 20, else 0.
- Alpha selection accuracy (Tables 2, 4): the method's alpha is among the
  grid alphas (0.0..1.0, step 0.1) giving the gold paragraph its best rank.
- Hybrid-sensitive subset (Tables 4, 5): questions whose gold rank changes
  across the alpha grid. The paper's prose is loose, but this definition
  is the one consistent with its tables: under it, the queries outside
  the subset count identically for every method (1826 hits and 1865
  correct alphas on SQuAD, for BM25, dense, fixed and DAT alike).

Ties are broken by corpus order (a stable sort), so ranks are deterministic.

    python -m baseline.evaluate                   # free baselines
    python -m baseline.evaluate --tune-alpha      # also grid-search alpha* on the tune set
    python -m baseline.evaluate --dat             # add DAT from cached judge replies

Writes results/baseline_checkpoint.csv, or results/dat_reproduction.csv
with --dat (plus a paired DAT-vs-fixed bootstrap comparison).
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from baseline.dat_fusion import fuse, minmax
from baseline.dat_judge import dat_alphas
from baseline.prepare_squad import load_set
from baseline.retrieve import load_scores

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "baseline_checkpoint.csv"
OUT_DAT = ROOT / "results" / "dat_reproduction.csv"
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.1), 1)

# Transcribed from the DAT paper (arXiv:2503.23013v1), SQuAD columns.
# (P@1, MRR@20) from Table 3 (all) / Table 5 (hybrid-sensitive);
# alpha selection accuracy from Table 2 (all) / Table 4 (hybrid-sensitive).
PAPER_SQUAD = {
    "all": {
        "BM25 only (alpha=0.0)": (0.7594, 0.8223, 0.7981),
        "Dense only (alpha=1.0)": (0.7396, 0.8119, 0.7789),
        "Fixed hybrid (alpha=0.6)": (0.8461, 0.8997, 0.8975),
        "DAT (GPT-4o-mini)": (0.8676, 0.9093, 0.9194),
        "DAT (GPT-4o)": (0.8740, 0.9130, 0.9234),
    },
    "hybrid": {
        "BM25 only (alpha=0.0)": (0.3906, 0.5420, 0.4590),
        "Dense only (alpha=1.0)": (0.3375, 0.5143, 0.4077),
        "Fixed hybrid (alpha=0.6)": (0.6229, 0.7493, 0.7255),
        "DAT (GPT-4o-mini)": (0.6805, 0.7750, 0.7840),
        "DAT (GPT-4o)": (0.6976, 0.7849, 0.7948),
    },
}
PAPER_COUNTS = {"questions": 2976, "hybrid": 1111}


def gold_ranks(scores, gold_index):
    """1-based rank of each query's gold document, ties broken by corpus
    order (equivalent to a stable descending sort)."""
    scores = np.asarray(scores)
    rows = np.arange(scores.shape[0])
    gold_scores = scores[rows, gold_index][:, None]
    higher = (scores > gold_scores).sum(axis=1)
    tied_before = ((scores == gold_scores) & (np.arange(scores.shape[1])[None, :] < gold_index[:, None])).sum(axis=1)
    return 1 + higher + tied_before


def precision_at_1(ranks):
    return float(np.mean(ranks == 1))


def mrr_at_k(ranks, k=20):
    return float(np.mean(np.where(ranks <= k, 1.0 / ranks, 0.0)))


def ranks_over_grid(norm_dense, norm_bm25, gold_index, alphas=ALPHAS):
    """(queries, len(alphas)) gold ranks, one column per fixed alpha."""
    return np.stack([gold_ranks(fuse(norm_dense, norm_bm25, a), gold_index) for a in alphas], axis=1)


def hybrid_sensitive(grid_ranks):
    """Queries whose gold rank is not the same for every alpha in the grid."""
    return (grid_ranks != grid_ranks[:, :1]).any(axis=1)


def alpha_selection_accuracy(grid_ranks, chosen_alpha, alphas=ALPHAS):
    """Fraction of queries whose chosen alpha achieves the best gold rank
    available on the grid. `chosen_alpha` is a scalar or one per query."""
    chosen = np.broadcast_to(np.asarray(chosen_alpha, dtype=np.float64), (grid_ranks.shape[0],))
    column = np.array([int(np.argmin(np.abs(alphas - a))) for a in chosen])
    best = grid_ranks.min(axis=1)
    return float(np.mean(grid_ranks[np.arange(len(column)), column] == best))


def paired_bootstrap_ci(values_a, values_b, n_boot=10000, seed=0, chunk=500):
    """Mean of (a - b) over the same queries, with a 95% percentile
    bootstrap interval from resampling queries with replacement."""
    diff = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, len(diff), size=(min(chunk, n_boot - start), len(diff)))
        means.append(diff[idx].mean(axis=1))
    means = np.concatenate(means)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparison(set_name, alpha_a, alpha_b, name_a, name_b):
    """Per-query P@1 and RR@20 of two methods on the same questions."""
    norm_dense, norm_bm25, gold_index = load_normalized(set_name)
    ranks_a = gold_ranks(fuse(norm_dense, norm_bm25, alpha_a), gold_index)
    ranks_b = gold_ranks(fuse(norm_dense, norm_bm25, alpha_b), gold_index)
    hit_a, hit_b = ranks_a == 1, ranks_b == 1
    print(f"\n== Paired comparison on the same {len(gold_index)} questions: {name_a} minus {name_b} ==")
    for metric, a, b in (
        ("P@1", hit_a, hit_b),
        ("MRR@20", np.where(ranks_a <= 20, 1.0 / ranks_a, 0.0), np.where(ranks_b <= 20, 1.0 / ranks_b, 0.0)),
    ):
        mean, low, high = paired_bootstrap_ci(a, b)
        verdict = "significant" if low > 0 or high < 0 else "not significant"
        print(f"  {metric:<7} {mean:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]  ({verdict}, 10,000 bootstrap resamples)")
    print(f"  #1 gained by {name_a}: {int((hit_a & ~hit_b).sum())} questions, "
          f"lost: {int((~hit_a & hit_b).sum())}, unchanged: {int((hit_a == hit_b).sum())}")


def load_normalized(set_name):
    corpus, queries = load_set(set_name)
    position = {doc["id"]: i for i, doc in enumerate(corpus)}
    gold_index = np.array([position[q["gold"]] for q in queries])
    bm25, dense = load_scores(set_name)
    return minmax(dense), minmax(bm25), gold_index


def tune_alpha(set_name="tune"):
    """Grid-search one global alpha maximizing P@1 (DAT §5.1), MRR@20 as
    tie-break, on the tune set only."""
    norm_dense, norm_bm25, gold_index = load_normalized(set_name)
    grid = ranks_over_grid(norm_dense, norm_bm25, gold_index)
    table = [(precision_at_1(grid[:, j]), mrr_at_k(grid[:, j]), float(a)) for j, a in enumerate(ALPHAS)]
    print(f"\nAlpha grid search on '{set_name}' ({len(gold_index)} questions):")
    for p1, mrr, a in table:
        print(f"  alpha={a:.1f}  P@1={p1:.4f}  MRR@20={mrr:.4f}")
    best = max(table, key=lambda row: (row[0], row[1]))
    print(f"  -> alpha* = {best[2]:.1f} (paper found 0.6 on its validation set)")
    return best[2]


def evaluate(set_name, methods):
    """`methods` maps a display name to a scalar or per-query alpha."""
    norm_dense, norm_bm25, gold_index = load_normalized(set_name)
    grid = ranks_over_grid(norm_dense, norm_bm25, gold_index)
    subsets = {"all": np.ones(len(gold_index), dtype=bool), "hybrid": hybrid_sensitive(grid)}
    rows = []
    for name, alpha in methods.items():
        ranks = gold_ranks(fuse(norm_dense, norm_bm25, alpha), gold_index)
        alpha_all = np.broadcast_to(np.asarray(alpha, dtype=np.float64), ranks.shape)
        for subset, mask in subsets.items():
            paper = PAPER_SQUAD[subset].get(name, (None, None, None))
            rows.append({
                "set": set_name, "subset": subset, "method": name, "n_queries": int(mask.sum()),
                "p_at_1": precision_at_1(ranks[mask]),
                "mrr_at_20": mrr_at_k(ranks[mask]),
                "alpha_acc": alpha_selection_accuracy(grid[mask], alpha_all[mask]),
                "paper_p_at_1": paper[0], "paper_mrr_at_20": paper[1], "paper_alpha_acc": paper[2],
            })
    return rows


def print_table(rows):
    def fmt(value):
        return "  -   " if value is None else f"{value:.4f}"

    for subset in ("all", "hybrid"):
        subset_rows = [r for r in rows if r["subset"] == subset]
        n = subset_rows[0]["n_queries"]
        paper_n = PAPER_COUNTS["questions" if subset == "all" else "hybrid"]
        print(f"\n== SQuAD {subset} questions: ours n={n}, paper n={paper_n} ==")
        print(f"{'method':<26} {'P@1 ours':>9} {'paper':>7} {'MRR@20 ours':>12} {'paper':>7} "
              f"{'alpha-acc ours':>15} {'paper':>7}")
        for r in subset_rows:
            print(f"{r['method']:<26} {fmt(r['p_at_1']):>9} {fmt(r['paper_p_at_1']):>7} "
                  f"{fmt(r['mrr_at_20']):>12} {fmt(r['paper_mrr_at_20']):>7} "
                  f"{fmt(r['alpha_acc']):>15} {fmt(r['paper_alpha_acc']):>7}")


def write_csv(rows, path=OUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"\nWritten to {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tune-alpha", action="store_true", help="also grid-search alpha* on the tune set")
    parser.add_argument("--dat", action="store_true", help="add DAT (GPT-4o-mini) from cached judge replies")
    args = parser.parse_args()

    methods = {
        "BM25 only (alpha=0.0)": 0.0,
        "Dense only (alpha=1.0)": 1.0,
        "Fixed hybrid (alpha=0.6)": 0.6,
    }
    if args.tune_alpha:
        best = tune_alpha("tune")
        if best != 0.6:
            methods[f"Fixed hybrid (tuned alpha={best:.1f})"] = best
    if args.dat:
        methods["DAT (GPT-4o-mini)"] = np.array([row["alpha"] for row in dat_alphas("test")])
    rows = evaluate("test", methods)
    print_table(rows)
    write_csv(rows, OUT_DAT if args.dat else OUT)
    if args.dat:
        paired_comparison("test", methods["DAT (GPT-4o-mini)"], 0.6, "DAT (GPT-4o-mini)", "Fixed 0.6")


if __name__ == "__main__":
    main()
