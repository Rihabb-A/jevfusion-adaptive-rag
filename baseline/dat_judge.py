"""
Step 1.4 of the DAT reproduction: the LLM effectiveness judge (paper §4.1).

For each question the judge sees only two paragraphs: the top-1 from BM25
and the top-1 from dense retrieval (paper §4). It returns two integers
0-5, "dense bm25", which baseline.dat_fusion.dat_alpha turns into alpha.

Every reply is stored in an append-only JSONL cache keyed by the exact
request, so re-runs never pay twice and an interrupted run resumes.

    python -m baseline.dat_judge --set test --limit 20   # pilot
    python -m baseline.dat_judge --set test              # all questions
    python -m baseline.dat_judge --set test --alphas     # alpha per question (no API calls)

Replies go to data/cache/judge/<model>/<set>.jsonl; spend goes to the
budget ledger; --alphas writes results/dat_alphas_<set>.csv.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from baseline import budget
from baseline.dat_fusion import dat_alpha
from baseline.prepare_squad import load_set
from baseline.retrieve import load_scores

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "judge"

# OpenRouter list prices for openai/gpt-4o-mini on the OpenAI endpoint,
# checked 2026-10-19. Only used for the pre-run estimate; the ledger
# records the cost OpenRouter actually bills.
USD_PER_INPUT_TOKEN = 0.15 / 1_000_000
USD_PER_OUTPUT_TOKEN = 0.60 / 1_000_000

# DAT paper, Appendix A (arXiv:2503.23013v1), word for word. The PDF prints
# it in a code font that adds spaces around punctuation ("( Cosine
# Distance )", "it ' s"); normal spacing is restored, no words changed.
DAT_PROMPT = """You are an evaluator assessing the retrieval effectiveness of dense retrieval (Cosine Distance) and BM25 retrieval for finding the correct answer.

## Task:
Given a question and two top1 search results (one from dense retrieval, one from BM25 retrieval), score each retrieval method from **0 to 5** based on whether the correct answer is likely to appear in top2, top3, etc.

### **Scoring Criteria:**
1. **Direct hit --> 5 points**
- If the retrieved document directly answers the question, assign **5 points**.
2. **Good wrong result (High likelihood correct answer is nearby) --> 3-4 points**
- If the top1 result is **conceptually close** to the correct answer (e.g., mentions relevant entities, related events, partial answer), it indicates the search method is in the right direction.
- Give **4** if it's very close, **3** if somewhat close.
3. **Bad wrong result (Low likelihood correct answer is nearby) --> 1-2 points**
- If the top1 result is **loosely related but misleading** (e.g., shares keywords but changes context), correct answers might not be in top2, top3.
- Give **2** if there's a small chance correct answers are nearby, **1** if unlikely.
4. **Completely off-track --> 0 points**
- If the result is **totally unrelated**, it means the retrieval method is failing.
---
### **Given Data:**
- **Question:** "{question}"
- **dense retrieval Top1 Result:** "{vector_reference}"
- **BM25 retrieval Top1 Result:** "{bm25_reference}"
---
### **Output Format:**
Return two integers separated by a space:
- **First number:** dense retrieval score.
- **Second number:** BM25 retrieval score.
- Example output: 3 4
(Vector: 3, BM25: 4)
**Do not output any other text.**"""

# Generation settings for every judge call. DAT does not state any; these
# are chosen for reproducibility. They are part of the cache key, so
# changing one re-judges instead of silently reusing old replies.
JUDGE_SETTINGS = {
    "temperature": 0,
    "seed": 42,  # best-effort determinism; OpenAI does not guarantee it
    "max_tokens": 10,  # the reply is "d b"; anything longer is cut, first line parsed
    # Pin OpenRouter to OpenAI's own endpoint (the host DAT cites) with no
    # fallback, so a run never mixes hosts; an outage errors instead.
    "provider": {"order": ["openai"], "allow_fallbacks": False},
}

# Exactly the requested format: "3 4".
_STRICT = re.compile(r"\s*([0-5])\s+([0-5])\s*")
# Tolerated: the same two scores on the first line, followed by anything the
# model echoed from the prompt's example, e.g. "3 4\n(Vector: 3, BM25: 4)".
_FIRST_LINE = re.compile(r"\s*([0-5])[ \t]*[ ,\t][ \t]*([0-5])\s*")


def build_prompt(question, dense_text, bm25_text):
    return DAT_PROMPT.format(question=question, vector_reference=dense_text, bm25_reference=bm25_text)


def parse_scores(text):
    """Return (dense_score, bm25_score) from the judge's reply, or None if it
    is not two integers 0-5 on the first line."""
    if text is None:
        return None
    match = _STRICT.fullmatch(text)
    if not match:
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        match = _FIRST_LINE.fullmatch(first_line)
    return (int(match.group(1)), int(match.group(2))) if match else None


def request_key(model, prompt, settings):
    """Stable id for one judge request: same model, prompt and generation
    settings -> same key -> cached reply reused."""
    blob = json.dumps({"model": model, "prompt": prompt, **settings}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class JudgeCache:
    """Append-only JSONL store of judge replies, one record per request key.

    Appending (rather than rewriting) means a crash loses at most the line
    being written; a truncated last line is skipped on reload.
    """

    def __init__(self, path):
        self.path = path
        self.records = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.records[record["key"]] = record

    def get(self, key):
        return self.records.get(key)

    def put(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records[record["key"]] = record

    def __len__(self):
        return len(self.records)


def top1_indices(scores):
    """Index of each query's highest-scoring document. np.argmax returns the
    first maximum, so ties go to corpus order, as in evaluate.gold_ranks."""
    return np.asarray(scores).argmax(axis=1)


def top1_candidates(set_name):
    """One record per question with its BM25 and dense top-1 paragraphs."""
    corpus, queries = load_set(set_name)
    bm25, dense = load_scores(set_name)
    bm25_top, dense_top = top1_indices(bm25), top1_indices(dense)
    return [
        {
            "query_id": q["id"],
            "question": q["question"],
            "gold": q["gold"],
            "dense_top1": corpus[d]["id"],
            "dense_text": corpus[d]["text"],
            "bm25_top1": corpus[b]["id"],
            "bm25_text": corpus[b]["text"],
        }
        for q, d, b in zip(queries, dense_top, bm25_top)
    ]


def call_judge(prompt, model, base_url, api_key, settings=JUDGE_SETTINGS, retries=3, timeout=60):
    """One chat completion through OpenRouter. Transient HTTP errors (429,
    5xx) are retried with backoff; the returned dict always has `http_status`
    and either the reply fields or `error`."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], **settings}
    for attempt in range(retries):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=timeout,
            )
            status = response.status_code
        except requests.RequestException as exc:
            status, response = None, exc
        latency_ms = (time.perf_counter() - start) * 1000
        if status == 200:
            payload = response.json()
            if "choices" in payload:
                usage = payload.get("usage", {})
                raw = payload["choices"][0]["message"].get("content")
                return {
                    "http_status": 200,
                    "raw": raw,
                    "parsed": parse_scores(raw),
                    "served_model": payload.get("model"),
                    "provider": payload.get("provider"),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "cost_usd": float(usage.get("cost", 0.0)),
                    "latency_ms": round(latency_ms, 1),
                }
            error = json.dumps(payload)[:500]
        else:
            error = response.text[:500] if status else repr(response)
        if (status is None or status in (429, 500, 502, 503, 529)) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        return {"http_status": status, "error": error, "latency_ms": round(latency_ms, 1)}


def cache_path(model, set_name):
    return CACHE_DIR / re.sub(r"[^\w.-]", "_", model) / f"{set_name}.jsonl"


def select_questions(candidates, limit=None, sample_seed=0):
    """All questions, or a reproducible random `limit` of them (spread across
    articles, unlike the first rows of the file)."""
    if limit is None or limit >= len(candidates):
        return candidates
    return random.Random(sample_seed).sample(candidates, limit)


def judge_set(set_name, limit=None, sample_seed=0, workers=8, candidates=None):
    """Judge the selected questions, skipping any already in the cache.
    Successful HTTP replies are cached (even unparseable ones, for 1.4.6);
    HTTP errors are not, so a re-run retries them.

    `workers` requests run in parallel; only this thread writes the cache
    and tallies cost, and the run stops early if billed spend would pass
    the budget left under BUDGET_CAP_USD.

    `candidates` (optional): rows with query_id, question, dense_text and
    bm25_text for another dataset (e.g. NFCorpus); by default the SQuAD
    candidates of `set_name`, exactly as before. `set_name` names the cache."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set in .env")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.getenv("DAT_JUDGE_MODEL", "openai/gpt-4o-mini")

    cache = JudgeCache(cache_path(model, set_name))
    selected = select_questions(candidates if candidates is not None else top1_candidates(set_name),
                                limit, sample_seed)
    jobs = []
    for row in selected:
        prompt = build_prompt(row["question"], row["dense_text"], row["bm25_text"])
        key = request_key(model, prompt, JUDGE_SETTINGS)
        jobs.append((row, prompt, key))
    todo = [job for job in jobs if cache.get(job[2]) is None]

    estimate = sum(len(p) / 4 * USD_PER_INPUT_TOKEN + 5 * USD_PER_OUTPUT_TOKEN for _, p, _ in todo) * 1.5
    print(f"{len(selected)} questions selected, {len(jobs) - len(todo)} cached, {len(todo)} to judge")
    if todo:
        budget.ensure_affordable(estimate)

    remaining = float(os.getenv("BUDGET_CAP_USD", "20")) - budget.spent()
    calls, tokens, cost, errors, done = 0, 0, 0.0, 0, 0
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(call_judge, prompt, model, base_url, api_key): (row, key)
                   for row, prompt, key in todo}
        for future in as_completed(futures):
            row, key = futures[future]
            result = future.result()
            done += 1
            if result["http_status"] == 200:
                calls += 1
                tokens += result["prompt_tokens"]
                cost += result["cost_usd"]
                cache.put({
                    "key": key, "set": set_name, "query_id": row["query_id"], "model": model,
                    "settings": JUDGE_SETTINGS,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **result,
                })
            else:
                errors += 1
                print(f"  [{row['query_id']}] HTTP {result['http_status']}: {result['error'][:200]}")
            if cost > remaining:
                print(f"  Stopping: spend ${cost:.4f} reached the ${remaining:.4f} left under the cap")
                break
            if done % 250 == 0:
                print(f"  {done}/{len(todo)} done, ${cost:.4f} so far, "
                      f"{(time.perf_counter() - started) / 60:.1f} min elapsed")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if calls:
            budget.record("dat_judge", model, calls, tokens, cost,
                          note=f"{set_name}, {calls} questions, {workers} workers")
    print(f"Judged {calls} new questions in {(time.perf_counter() - started) / 60:.1f} min: "
          f"{tokens} input tokens, ${cost:.6f}, {errors} HTTP errors (not cached, re-run to retry)")
    return [(row, cache.get(key)) for row, _, key in jobs]


def print_pilot(results):
    """Per-question view for a small run: judge scores next to which
    retriever actually had the gold paragraph at #1."""
    print(f"\n{'question':<52} {'dense#1':>8} {'bm25#1':>7} {'reply':>6} {'d':>2} {'b':>2} {'alpha':>6} {'ms':>6}")
    for row, record in results:
        dense_ok = "right" if row["dense_top1"] == row["gold"] else "wrong"
        bm25_ok = "right" if row["bm25_top1"] == row["gold"] else "wrong"
        if row["dense_top1"] == row["bm25_top1"]:
            bm25_ok = "same"
        if record is None:
            print(f"{row['question'][:52]:<52} {dense_ok:>8} {bm25_ok:>7}  (HTTP error, not cached)")
            continue
        parsed = record.get("parsed")
        d, b = parsed if parsed else ("?", "?")
        alpha = f"{dat_alpha(d, b):.1f}" if parsed else "-"
        print(f"{row['question'][:52]:<52} {dense_ok:>8} {bm25_ok:>7} {repr(record['raw'])[:6]:>6} "
              f"{d:>2} {b:>2} {alpha:>6} {record['latency_ms']:>6.0f}")


def summarize(results, total_questions):
    records = [r for _, r in results if r is not None]
    parsed = sum(r.get("parsed") is not None for r in records)
    providers = sorted({str(r.get("provider")) for r in records})
    served = sorted({str(r.get("served_model")) for r in records})
    cost = [r["cost_usd"] for r in records]
    latency = sorted(r["latency_ms"] for r in records)
    print(f"\nReplies: {len(records)}/{len(results)}, parsed: {parsed}/{len(records)}")
    print(f"Served by provider(s): {providers}, model(s): {served}")
    if records:
        mean_cost = sum(cost) / len(cost)
        print(f"Cost per call: mean ${mean_cost:.6f} -> {total_questions} questions ~ ${mean_cost * total_questions:.4f}")
        print(f"Latency per call: median {latency[len(latency) // 2]:.0f} ms, max {latency[-1]:.0f} ms "
              f"-> sequential run of {total_questions} ~ {sum(latency) / len(latency) * total_questions / 60000:.0f} min")


def dat_alphas(set_name, model=None):
    """Per-question DAT alpha from cached judge replies, in queries.jsonl
    order (the row order evaluate.py uses). Makes no API calls; stops if any
    question lacks a parsed reply, rather than silently defaulting it."""
    load_dotenv(ROOT / ".env")
    model = model or os.getenv("DAT_JUDGE_MODEL", "openai/gpt-4o-mini")
    cache = JudgeCache(cache_path(model, set_name))
    rows = []
    for row in top1_candidates(set_name):
        prompt = build_prompt(row["question"], row["dense_text"], row["bm25_text"])
        record = cache.get(request_key(model, prompt, JUDGE_SETTINGS))
        if record is None or record.get("parsed") is None:
            raise SystemExit(f"No parsed judge reply for {row['query_id']}; run: "
                             f"python -m baseline.dat_judge --set {set_name}")
        score_dense, score_bm25 = record["parsed"]
        rows.append({
            "query_id": row["query_id"],
            "dense_score": score_dense,
            "bm25_score": score_bm25,
            "alpha": dat_alpha(score_dense, score_bm25),
            "dense_top1_correct": row["dense_top1"] == row["gold"],
            "bm25_top1_correct": row["bm25_top1"] == row["gold"],
            "same_top1": row["dense_top1"] == row["bm25_top1"],
        })
    return rows


def write_alphas(rows, set_name):
    path = ROOT / "results" / f"dat_alphas_{set_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-question alphas written to {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="DAT LLM judge (paper §4.1)")
    parser.add_argument("--set", choices=["test", "tune"], required=True)
    parser.add_argument("--limit", type=int, help="judge a reproducible random sample of this many questions")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="parallel requests")
    parser.add_argument("--alphas", action="store_true", help="compute alphas from cached replies only")
    args = parser.parse_args()

    if args.alphas:
        write_alphas(dat_alphas(args.set), args.set)
        return

    results = judge_set(args.set, args.limit, args.sample_seed, args.workers)
    if args.limit and args.limit <= 50:
        print_pilot(results)
    total = len(load_set(args.set)[1])
    summarize(results, total)


if __name__ == "__main__":
    main()
