"""
Step 2.10: the final comparison table on the test set, recomputed from the
cached scores and judge replies (no API calls), so it cannot drift from
the per-step results.

For every method: P@1, MRR@20, hybrid-sensitive P@1, the paired P@1
difference vs DAT and vs fixed alpha 0.6 with 95% bootstrap CIs, judge
cost per 1,000 questions (mean billed cost of the full test run) and
sequential judge latency p50 / p95 (from results/latency_summary.csv,
written by extension.latency).

    python -m extension.final_table

Writes results/final_comparison.csv.
"""

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from baseline.dat_judge import cache_path as dat_cache_path
from baseline.dat_judge import dat_alphas
from baseline.evaluate import mrr_at_k, paired_bootstrap_ci, precision_at_1
from extension.jev_fusion import CHOSEN_J3_OPTION, CHOSEN_MAPPING, j1_alphas, load_judgments, multi_alphas, write_csv
from extension.jev_judge import cache_path as jev_cache_path
from extension.jev_judge import jev_config
from extension.rrf import gold_ranks_by_method

ROOT = Path(__file__).resolve().parent.parent
LATENCY = ROOT / "results" / "latency_summary.csv"


def mean_cost_per_1000(path):
    with path.open(encoding="utf-8") as handle:
        return 1000 * statistics.mean(json.loads(line)["cost_usd"] for line in handle)


def latency_by_judge():
    if not LATENCY.exists():
        return {}
    with LATENCY.open(encoding="utf-8") as handle:
        return {row["judge"]: row for row in csv.DictReader(handle)}


def main():
    load_dotenv(ROOT / ".env")
    _, _, jev_model = jev_config()
    dat_model = "openai/gpt-4o-mini"
    names = {"dat": "DAT (GPT-4o-mini)", "j1": "Jev J1", "j3": "Jev J3"}
    alphas = {
        "BM25 only (alpha=0.0)": 0.0,
        "Dense only (alpha=1.0)": 1.0,
        "Fixed hybrid (alpha=0.6)": 0.6,
        names["dat"]: np.array([row["alpha"] for row in dat_alphas("test")]),
        names["j1"]: j1_alphas(load_judgments("test", 1), CHOSEN_MAPPING),
        names["j3"]: multi_alphas(load_judgments("test", 3), CHOSEN_J3_OPTION),
    }
    ranks, hybrid = gold_ranks_by_method("test", alphas)
    order = ["BM25 only (alpha=0.0)", "Dense only (alpha=1.0)", "RRF (k=60)", "Fixed hybrid (alpha=0.6)",
             names["dat"], names["j1"], names["j3"]]
    cost = {
        names["dat"]: mean_cost_per_1000(dat_cache_path(dat_model, "test")),
        names["j1"]: mean_cost_per_1000(jev_cache_path(jev_model, "test", 1)),
        names["j3"]: mean_cost_per_1000(jev_cache_path(jev_model, "test", 3)),
    }
    latency = latency_by_judge()

    rows = []
    for name in order:
        r = ranks[name]
        row = {"method": name, "p_at_1": precision_at_1(r), "mrr_at_20": mrr_at_k(r),
               "hybrid_p_at_1": precision_at_1(r[hybrid])}
        for ref_key, ref in (("vs_dat", names["dat"]), ("vs_fixed", "Fixed hybrid (alpha=0.6)")):
            if name == ref:
                row.update({f"p1_{ref_key}": None, f"p1_{ref_key}_low": None, f"p1_{ref_key}_high": None})
            else:
                mean, low, high = paired_bootstrap_ci(r == 1, ranks[ref] == 1)
                row.update({f"p1_{ref_key}": mean, f"p1_{ref_key}_low": low, f"p1_{ref_key}_high": high})
        row["judge_usd_per_1000"] = cost.get(name, 0.0)
        lat = latency.get(name)
        row["judge_p50_ms"] = float(lat["p50_ms"]) if lat else (0.0 if name not in cost else None)
        row["judge_p95_ms"] = float(lat["p95_ms"]) if lat else (0.0 if name not in cost else None)
        rows.append(row)

    print(f"== Final comparison, SQuAD test, {len(ranks[order[0]])} questions ==")
    print(f"{'method':<26} {'P@1':>7} {'MRR@20':>7} {'hyb P@1':>8} {'P@1 vs DAT [95% CI]':>28} "
          f"{'$/1k q':>8} {'p50 ms':>7} {'p95 ms':>7}")
    for row in rows:
        vs = ("reference" if row["p1_vs_dat"] is None else
              f"{row['p1_vs_dat']:+.4f} [{row['p1_vs_dat_low']:+.4f}, {row['p1_vs_dat_high']:+.4f}]")
        p50 = "-" if row["judge_p50_ms"] is None else f"{row['judge_p50_ms']:.0f}"
        p95 = "-" if row["judge_p95_ms"] is None else f"{row['judge_p95_ms']:.0f}"
        print(f"{row['method']:<26} {row['p_at_1']:>7.4f} {row['mrr_at_20']:>7.4f} {row['hybrid_p_at_1']:>8.4f} "
              f"{vs:>28} {row['judge_usd_per_1000']:>8.4f} {p50:>7} {p95:>7}")
    write_csv([{k: ("" if v is None else v) for k, v in row.items()} for row in rows],
              ROOT / "results" / "final_comparison.csv")


if __name__ == "__main__":
    main()
