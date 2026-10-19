"""
JevFusion alphas: turn cached Jev judgments into per-question fusion
weights and evaluate them with the baseline's own metrics and fusion.

J1 uses DAT's rule unchanged (baseline.dat_fusion.dat_alpha, paper Eq. 6),
which needs integer scores 0-5. Jev returns a decimal (a probability-
weighted mean of the levels), so a mapping is needed:

- "top_level": Jev's most likely level (ties -> lower level).
- "round": the decimal score rounded half up.

Step 2.5 chose the mapping on the tune set by a rule fixed before looking
(higher tune P@1, then MRR@20, then "top_level"): CHOSEN_MAPPING.
Step 2.6 judges J1 against DAT on the test set with a non-inferiority
margin also fixed before looking: NONINFERIORITY_MARGIN.

    python -m extension.jev_fusion --set tune --compare-mappings   # step 2.5
    python -m extension.jev_fusion --set test                      # step 2.6
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from baseline.dat_fusion import dat_alpha, fuse
from baseline.dat_judge import JudgeCache, dat_alphas, request_key
from baseline.evaluate import (
    alpha_selection_accuracy,
    gold_ranks,
    hybrid_sensitive,
    load_normalized,
    mrr_at_k,
    paired_bootstrap_ci,
    precision_at_1,
    ranks_over_grid,
)
from extension.jev_judge import build_request, cache_path, jev_config, topk_candidates

ROOT = Path(__file__).resolve().parent.parent
MAPPINGS = ("top_level", "round")
CHOSEN_MAPPING = "top_level"  # Step 2.5, tune set: P@1 0.8363 vs 0.8356 for "round"
NONINFERIORITY_MARGIN = 0.01  # Step 2.6: Jev "matches" DAT if the P@1 CI lies above -1 point


def to_dat_score(judgment, mapping):
    """One Jev judgment -> integer 0-5 for DAT's alpha rule."""
    if mapping == "top_level":
        return judgment["top_level"]
    if mapping == "round":
        return min(5, int(math.floor(judgment["score"] + 0.5)))
    raise ValueError(f"unknown mapping {mapping!r}")


def load_judgments(set_name, k):
    """Valid Jev judgments per question, in queries.jsonl order (the row
    order evaluate.py uses). Stops if any question lacks a valid reply,
    rather than silently defaulting it."""
    _, _, model = jev_config()
    cache = JudgeCache(cache_path(model, set_name, k))
    out = []
    for row in topk_candidates(set_name, k):
        body = build_request(model, row["question"], row["passages"])
        record = cache.get(request_key(model, json.dumps(body, sort_keys=True, ensure_ascii=False), {}))
        if record is None or record.get("judgments") is None:
            raise SystemExit(f"No valid Jev reply for {row['query_id']} (k={k}); run: "
                             f"python -m extension.jev_judge --set {set_name} --k {k}")
        out.append(record["judgments"])
    return out


def j1_alphas(judgments, mapping):
    """DAT's Eq. 6 on Jev's top-1 judgments, mapped to integers."""
    return np.array([
        dat_alpha(to_dat_score(j["dense_1"], mapping), to_dat_score(j["bm25_1"], mapping))
        for j in judgments
    ])


def evaluate_alphas(set_name, methods):
    """P@1, MRR@20 and alpha accuracy, on all and hybrid-sensitive
    questions, for a {name: scalar or per-query alpha} dict."""
    norm_dense, norm_bm25, gold_index = load_normalized(set_name)
    grid = ranks_over_grid(norm_dense, norm_bm25, gold_index)
    hybrid = hybrid_sensitive(grid)
    rows = []
    for name, alpha in methods.items():
        ranks = gold_ranks(fuse(norm_dense, norm_bm25, alpha), gold_index)
        alpha_all = np.broadcast_to(np.asarray(alpha, dtype=np.float64), ranks.shape)
        rows.append({
            "set": set_name, "method": name, "n_queries": len(gold_index),
            "p_at_1": precision_at_1(ranks), "mrr_at_20": mrr_at_k(ranks),
            "alpha_acc": alpha_selection_accuracy(grid, alpha_all),
            "hybrid_p_at_1": precision_at_1(ranks[hybrid]), "hybrid_mrr_at_20": mrr_at_k(ranks[hybrid]),
            "n_hybrid": int(hybrid.sum()),
        })
    return rows


def print_rows(rows, title):
    print(f"\n== {title} ==")
    print(f"{'method':<34} {'P@1':>7} {'MRR@20':>7} {'alpha-acc':>9} {'hybrid P@1':>10}")
    for r in rows:
        print(f"{r['method']:<34} {r['p_at_1']:>7.4f} {r['mrr_at_20']:>7.4f} "
              f"{r['alpha_acc']:>9.4f} {r['hybrid_p_at_1']:>10.4f}")


def choose_mapping(rows_by_mapping):
    """The pre-declared rule: P@1, then MRR@20, then 'top_level'."""
    order = {name: i for i, name in enumerate(MAPPINGS)}
    return max(rows_by_mapping, key=lambda m: (rows_by_mapping[m]["p_at_1"],
                                               rows_by_mapping[m]["mrr_at_20"], -order[m]))


def compare_pair(set_name, alpha_a, alpha_b):
    """Paired per-question comparison of two alphas: (mean, CI low, CI high)
    of a - b for P@1 and MRR@20, plus rank-1 questions gained and lost."""
    norm_dense, norm_bm25, gold_index = load_normalized(set_name)
    ranks_a = gold_ranks(fuse(norm_dense, norm_bm25, alpha_a), gold_index)
    ranks_b = gold_ranks(fuse(norm_dense, norm_bm25, alpha_b), gold_index)

    def reciprocal(ranks):
        return np.where(ranks <= 20, 1.0 / ranks, 0.0)

    return {
        "P@1": paired_bootstrap_ci(ranks_a == 1, ranks_b == 1),
        "MRR@20": paired_bootstrap_ci(reciprocal(ranks_a), reciprocal(ranks_b)),
        "gained": int(((ranks_a == 1) & (ranks_b != 1)).sum()),
        "lost": int(((ranks_a != 1) & (ranks_b == 1)).sum()),
    }


def verdict(ci, margin=NONINFERIORITY_MARGIN):
    """Read a (mean, low, high) difference interval against the margin."""
    _, low, high = ci
    if low > 0:
        return "better (CI entirely above 0)"
    if low > -margin:
        return ("non-inferior (CI above -margin) but significantly lower" if high < 0
                else "matches: non-inferior (CI above -margin), difference not significant")
    if high < 0:
        return "worse (CI entirely below 0)"
    return "inconclusive (CI reaches below -margin)"


def print_comparison(name_a, name_b, result, with_verdict=False):
    print(f"\n== {name_a} minus {name_b} (paired, 10,000 bootstrap resamples) ==")
    for metric in ("P@1", "MRR@20"):
        mean, low, high = result[metric]
        print(f"  {metric:<7} {mean:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]")
    print(f"  rank 1 gained: {result['gained']}, lost: {result['lost']}")
    if with_verdict:
        print(f"  Verdict on P@1 (margin {NONINFERIORITY_MARGIN:.2f}): {verdict(result['P@1'])}")


def right_direction(alpha, dense_correct):
    """Did alpha lean toward the retriever whose top-1 was correct?"""
    return alpha > 0.5 if dense_correct else alpha < 0.5


def judge_agreement(dat_rows, jev_alpha):
    """How the two judges' alphas relate, overall and on the questions where
    exactly one retriever had the gold paragraph at rank 1."""
    dat_alpha = np.array([row["alpha"] for row in dat_rows])
    print(f"\n== Judge agreement on {len(dat_rows)} questions ==")
    print(f"  Same alpha: {np.mean(dat_alpha == jev_alpha):.1%}")
    one_right = [i for i, row in enumerate(dat_rows) if row["dense_top1_correct"] != row["bm25_top1_correct"]]
    print(f"  Questions where exactly one retriever is right: {len(one_right)}")
    for name, alphas in (("DAT (GPT-4o-mini)", dat_alpha), ("Jev J1", jev_alpha)):
        good = sum(right_direction(alphas[i], dat_rows[i]["dense_top1_correct"]) for i in one_right)
        neutral = sum(alphas[i] == 0.5 for i in one_right)
        print(f"    {name:<18} toward the right retriever {good / len(one_right):.1%}, "
              f"neutral {neutral / len(one_right):.1%}, wrong way {(len(one_right) - good - neutral) / len(one_right):.1%}")
    only_jev = sum(right_direction(jev_alpha[i], dat_rows[i]["dense_top1_correct"])
                   and not right_direction(dat_alpha[i], dat_rows[i]["dense_top1_correct"]) for i in one_right)
    only_dat = sum(right_direction(dat_alpha[i], dat_rows[i]["dense_top1_correct"])
                   and not right_direction(jev_alpha[i], dat_rows[i]["dense_top1_correct"]) for i in one_right)
    print(f"    right direction for Jev only: {only_jev}, for DAT only: {only_dat}")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"\nWritten to {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="JevFusion alphas and evaluation")
    parser.add_argument("--set", choices=["test", "tune"], required=True)
    parser.add_argument("--compare-mappings", action="store_true", help="step 2.5: choose the mapping on tune")
    args = parser.parse_args()

    judgments = load_judgments(args.set, 1)
    if args.compare_mappings:
        methods = {"Fixed hybrid (alpha=0.6)": 0.6}
        for mapping in MAPPINGS:
            methods[f"Jev J1 ({mapping})"] = j1_alphas(judgments, mapping)
        rows = evaluate_alphas(args.set, methods)
        print_rows(rows, f"SQuAD {args.set}, {len(judgments)} questions")
        by_mapping = {m: next(r for r in rows if r["method"] == f"Jev J1 ({m})") for m in MAPPINGS}
        print(f"\nChosen mapping (tune P@1, then MRR@20, then top_level): {choose_mapping(by_mapping)}")
        return

    jev = j1_alphas(judgments, CHOSEN_MAPPING)
    methods = {
        "BM25 only (alpha=0.0)": 0.0,
        "Dense only (alpha=1.0)": 1.0,
        "Fixed hybrid (alpha=0.6)": 0.6,
    }
    dat_rows = dat_alphas(args.set) if args.set == "test" else None  # DAT was judged on test only
    if dat_rows:
        methods["DAT (GPT-4o-mini)"] = np.array([row["alpha"] for row in dat_rows])
    methods[f"Jev J1 ({CHOSEN_MAPPING})"] = jev
    rows = evaluate_alphas(args.set, methods)
    print_rows(rows, f"SQuAD {args.set}, {len(judgments)} questions")
    write_csv(rows, ROOT / "results" / f"jev_j1_{args.set}.csv")

    print_comparison("Jev J1", "Fixed 0.6", compare_pair(args.set, jev, 0.6))
    if dat_rows:
        print_comparison("Jev J1", "DAT (GPT-4o-mini)",
                         compare_pair(args.set, jev, methods["DAT (GPT-4o-mini)"]), with_verdict=True)
        judge_agreement(dat_rows, jev)
        write_csv([
            {"query_id": row["query_id"],
             "jev_dense_score": j["dense_1"]["score"], "jev_dense_level": j["dense_1"]["top_level"],
             "jev_bm25_score": j["bm25_1"]["score"], "jev_bm25_level": j["bm25_1"]["top_level"],
             "jev_alpha": float(a), "dat_dense_score": row["dense_score"], "dat_bm25_score": row["bm25_score"],
             "dat_alpha": row["alpha"], "dense_top1_correct": row["dense_top1_correct"],
             "bm25_top1_correct": row["bm25_top1_correct"], "same_top1": row["same_top1"]}
            for row, j, a in zip(dat_rows, judgments, jev)
        ], ROOT / "results" / f"jev_j1_alphas_{args.set}.csv")


if __name__ == "__main__":
    main()
