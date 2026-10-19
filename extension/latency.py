"""
Step 2.10: fair judge latency. The judge runs used 8 parallel requests
under load, so their logged latencies are not comparable between judges.
Here every judge is timed the same way:

- the same random sample of test questions (seed 0) for every judge;
- one request at a time, never parallel;
- per question the three judges take turns in a rotating order, so a slow
  network patch hits all of them alike;
- 2 warm-up calls per judge first, discarded;
- fresh, uncached calls, timed end to end (including any retry), with the
  same request format, model and settings as the real runs.

Each fresh reply is also compared with the cached reply for the same
request, which measures how repeatable each judge is.

    python -m extension.latency --n 100

Writes results/latency_test.csv (one row per timed call) and
results/latency_summary.csv; spend goes to the budget ledger.
"""

import argparse
import csv
import json
import os
import random
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from baseline import budget
from baseline.dat_judge import JUDGE_SETTINGS, JudgeCache, build_prompt, call_judge, request_key, top1_candidates
from baseline.dat_judge import cache_path as dat_cache_path
from extension.jev_judge import (
    USD_PER_INPUT_TOKEN,
    VERSIONED_MODEL,
    build_request,
    call_jev,
    jev_config,
    parse_jev,
    topk_candidates,
)
from extension.jev_judge import cache_path as jev_cache_path

ROOT = Path(__file__).resolve().parent.parent
JUDGES = ("DAT (GPT-4o-mini)", "Jev J1", "Jev J3")
WARMUP = 2


def percentile(values, q):
    """Linear-interpolated percentile, q in [0, 100]."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Judges:
    """Builds and sends one request per judge, in the exact format of the
    real runs, and reports latency, cost and agreement with the cache."""

    def __init__(self, set_name):
        load_dotenv(ROOT / ".env")
        self.or_key = os.getenv("OPENROUTER_API_KEY")
        self.or_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.dat_model = os.getenv("DAT_JUDGE_MODEL", "openai/gpt-4o-mini")
        self.jev_key, self.jev_base, self.jev_model = jev_config()
        self.expected_jev = self.jev_model if VERSIONED_MODEL.fullmatch(self.jev_model) else None
        self.top1 = top1_candidates(set_name)
        self.topk = {1: topk_candidates(set_name, 1), 3: topk_candidates(set_name, 3)}
        self.dat_cache = JudgeCache(dat_cache_path(self.dat_model, set_name))
        self.jev_cache = {k: JudgeCache(jev_cache_path(self.jev_model, set_name, k)) for k in (1, 3)}

    def call(self, judge, index):
        start = time.perf_counter()
        if judge == "DAT (GPT-4o-mini)":
            row = self.top1[index]
            prompt = build_prompt(row["question"], row["dense_text"], row["bm25_text"])
            result = call_judge(prompt, self.dat_model, self.or_base, self.or_key)
            wall_ms = (time.perf_counter() - start) * 1000
            cached = self.dat_cache.get(request_key(self.dat_model, prompt, JUDGE_SETTINGS))
            fresh = list(result["parsed"]) if result.get("parsed") else None
            # The cache stores JSON, so its scores come back as a list: compare lists, not tuple vs list.
            same = bool(fresh and cached and cached.get("parsed") and fresh == list(cached["parsed"]))
            return {
                "query_id": row["query_id"], "http_status": result["http_status"], "latency_ms": round(wall_ms, 1),
                "input_tokens": result.get("prompt_tokens", 0), "cost_usd": result.get("cost_usd", 0.0),
                "fresh_reply": " ".join(map(str, fresh)) if fresh else "", "same_as_cached": same,
            }
        k = 1 if judge == "Jev J1" else 3
        row = self.topk[k][index]
        body = build_request(self.jev_model, row["question"], row["passages"])
        result = call_jev(body, self.jev_base, self.jev_key)
        wall_ms = (time.perf_counter() - start) * 1000
        same, tokens, fresh = False, 0, ""
        if result["http_status"] == 200:
            payload = result["payload"]
            tokens = int((payload.get("usage") or {}).get("input_tokens", 0))
            judgments, _ = parse_jev(payload, [p["slot"] for p in row["passages"]], self.expected_jev)
            cached = self.jev_cache[k].get(request_key(self.jev_model, json.dumps(body, sort_keys=True, ensure_ascii=False), {}))
            if judgments:
                fresh = " ".join(str(judgments[s]["top_level"]) for s in judgments)
            if judgments and cached and cached.get("judgments"):
                same = all(judgments[s]["top_level"] == cached["judgments"][s]["top_level"] for s in judgments)
        return {
            "query_id": row["query_id"], "http_status": result["http_status"], "latency_ms": round(wall_ms, 1),
            "input_tokens": tokens, "cost_usd": tokens * USD_PER_INPUT_TOKEN,
            "fresh_reply": fresh, "same_as_cached": same,
        }


def main():
    parser = argparse.ArgumentParser(description="Fair sequential judge latency (step 2.10)")
    parser.add_argument("--n", type=int, default=100, help="questions to time")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    judges = Judges("test")
    sample = random.Random(args.seed).sample(range(len(judges.top1)), args.n + WARMUP)
    warmup, timed = sample[:WARMUP], sample[WARMUP:]
    # Upper-bound estimate from the real runs' mean cost per question, +50% margin.
    budget.ensure_affordable(len(sample) * (0.000110 + 0.000047 + 0.000118) * 1.5)

    spend = {judge: [0, 0, 0.0] for judge in JUDGES}  # calls, tokens, cost
    rows = []
    try:
        for index in warmup:
            for judge in JUDGES:
                result = judges.call(judge, index)
                spend[judge][0] += 1
                spend[judge][1] += result["input_tokens"]
                spend[judge][2] += result["cost_usd"]
        print(f"Warm-up done ({WARMUP} calls per judge, discarded). Timing {len(timed)} questions...")
        for position, index in enumerate(timed):
            shift = position % len(JUDGES)
            for slot, judge in enumerate(JUDGES[shift:] + JUDGES[:shift]):
                result = judges.call(judge, index)
                spend[judge][0] += 1
                spend[judge][1] += result["input_tokens"]
                spend[judge][2] += result["cost_usd"]
                rows.append({"judge": judge, "order_slot": slot, **result})
            if (position + 1) % 25 == 0:
                print(f"  {position + 1}/{len(timed)} questions timed")
    finally:
        for judge, (calls, tokens, cost) in spend.items():
            if calls:
                budget.record("latency_" + judge.split(" (")[0].lower().replace(" ", "_"),
                              judges.dat_model if judge.startswith("DAT") else judges.jev_model,
                              calls, tokens, cost, note=f"step 2.10 sequential latency, {calls} calls incl. warm-up")

    out = ROOT / "results" / "latency_test.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    print(f"\n== Sequential judge latency, {len(timed)} test questions (ms) ==")
    print(f"{'judge':<20} {'ok':>4} {'p50':>7} {'p95':>7} {'mean':>7} {'max':>7} {'$/1k q':>8} {'same as cached':>15}")
    for judge in JUDGES:
        mine = [r for r in rows if r["judge"] == judge and r["http_status"] == 200]
        latency = [r["latency_ms"] for r in mine]
        cost_per_1k = 1000 * statistics.mean(r["cost_usd"] for r in mine)
        entry = {
            "judge": judge, "n_ok": len(mine), "n_failed": sum(r["judge"] == judge for r in rows) - len(mine),
            "p50_ms": percentile(latency, 50), "p95_ms": percentile(latency, 95),
            "mean_ms": statistics.mean(latency), "max_ms": max(latency),
            "usd_per_1000_questions": cost_per_1k,
            "same_as_cached": statistics.mean(r["same_as_cached"] for r in mine),
        }
        summary.append(entry)
        print(f"{judge:<20} {entry['n_ok']:>4} {entry['p50_ms']:>7.0f} {entry['p95_ms']:>7.0f} {entry['mean_ms']:>7.0f} "
              f"{entry['max_ms']:>7.0f} {cost_per_1k:>8.4f} {entry['same_as_cached']:>15.1%}")
    with (ROOT / "results" / "latency_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in s.items()} for s in summary)
    total = sum(cost for _, _, cost in spend.values())
    print(f"\nTotal spend for this step: ${total:.4f}. Written to results/latency_test.csv and results/latency_summary.csv")


if __name__ == "__main__":
    main()
