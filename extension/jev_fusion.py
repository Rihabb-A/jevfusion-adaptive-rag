"""
JevFusion alphas: turn cached Jev judgments into per-question fusion
weights and evaluate them with the baseline's own metrics and fusion.

J1 uses DAT's rule unchanged (baseline.dat_fusion.dat_alpha, paper Eq. 6),
which needs integer scores 0-5. Jev returns a decimal (a probability-
weighted mean of the levels), so a mapping is needed:

- "top_level": Jev's most likely level (ties -> lower level).
- "round": the decimal score rounded half up.

The mapping is chosen on the tune set only, by a rule fixed before
looking: higher tune P@1, then higher tune MRR@20, then "top_level".

    python -m extension.jev_fusion --set tune --k 1 --compare-mappings
"""

import argparse
import json
import math

import numpy as np

from baseline.dat_fusion import dat_alpha, fuse
from baseline.dat_judge import JudgeCache, request_key
from baseline.evaluate import (
    alpha_selection_accuracy,
    gold_ranks,
    hybrid_sensitive,
    load_normalized,
    mrr_at_k,
    precision_at_1,
    ranks_over_grid,
)
from extension.jev_judge import build_request, cache_path, jev_config, topk_candidates

MAPPINGS = ("top_level", "round")


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
            "method": name,
            "p_at_1": precision_at_1(ranks), "mrr_at_20": mrr_at_k(ranks),
            "alpha_acc": alpha_selection_accuracy(grid, alpha_all),
            "hybrid_p_at_1": precision_at_1(ranks[hybrid]), "n_hybrid": int(hybrid.sum()),
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


def main():
    parser = argparse.ArgumentParser(description="JevFusion alphas and evaluation")
    parser.add_argument("--set", choices=["test", "tune"], required=True)
    parser.add_argument("--k", type=int, choices=[1], default=1)
    parser.add_argument("--compare-mappings", action="store_true")
    args = parser.parse_args()

    judgments = load_judgments(args.set, args.k)
    methods = {"Fixed hybrid (alpha=0.6)": 0.6}
    for mapping in MAPPINGS:
        methods[f"Jev J1 ({mapping})"] = j1_alphas(judgments, mapping)
    rows = evaluate_alphas(args.set, methods)
    print_rows(rows, f"SQuAD {args.set}, {len(judgments)} questions")
    if args.compare_mappings:
        by_mapping = {m: next(r for r in rows if r["method"] == f"Jev J1 ({m})") for m in MAPPINGS}
        print(f"\nChosen mapping (tune P@1, then MRR@20, then top_level): {choose_mapping(by_mapping)}")


if __name__ == "__main__":
    main()
