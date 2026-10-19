"""
JevFusion judge: DAT's 0-5 retrieval-effectiveness rubric asked of Jev
(TypeSafe System One, https://docs.typesafe.ai) instead of GPT-4o-mini.

- k = 1 (J1): Jev scores the top-1 paragraph from each retriever, the same
  two paragraphs DAT's judge saw. Isolates the judge swap.
- k = 3 (J3): Jev scores the top-3 from each retriever (our extension).

One request per question: the question and all 2k paragraphs go in the
`state` once (billed once), with one 6-level `score` question per
paragraph. If both retrievers return the same paragraph it is sent under
both labels, as DAT's prompt shows it twice.

Each reply keeps the full answer per paragraph (decimal score, level
probabilities, most likely level, confidence); how a decimal score becomes
an alpha is decided later on the tune set, not here.

Reuses the baseline's retrieval cache, reply cache, question sampler and
budget ledger, so Jev judges exactly the paragraphs DAT judged.

    python -m extension.jev_judge --set tune --k 1 --limit 20   # pilot
    python -m extension.jev_judge --set test --k 1              # all questions

Replies go to data/cache/jev/<model>/<set>_k<k>.jsonl; spend goes to the
budget ledger.
"""

import argparse
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from baseline import budget
from baseline.dat_judge import JudgeCache, request_key, select_questions
from baseline.prepare_squad import load_set
from baseline.retrieve import load_scores

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "jev"

# Jev 1.13 list price (docs.typesafe.ai/models, checked 2026-10-19): input
# tokens only, output free. Jev reports tokens, not dollars, so recorded
# cost = input_tokens x this price.
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000

# DAT's rubric (paper §4.1 and Appendix A) as Jev score levels, low to high.
DAT_RUBRIC = [
    "0 - Completely off-track: the passage is totally unrelated to the question; the retrieval method is failing.",
    "1 - Bad wrong result: loosely related but misleading (e.g. shares keywords but changes context); the correct answer is unlikely to be nearby.",
    "2 - Bad wrong result: loosely related but misleading; there is a small chance the correct answer is nearby.",
    "3 - Good wrong result: somewhat conceptually close to the correct answer (e.g. mentions relevant entities or related events); the search is in the right direction.",
    "4 - Good wrong result: very conceptually close to the correct answer, or a partial answer; the correct answer is likely nearby.",
    "5 - Direct hit: the passage directly answers the question.",
]
INSTRUCTIONS = (
    "Score the retrieval effectiveness of passage '{slot}', the top{rank} result of {method} retrieval, "
    "for finding the correct answer to the question: is the correct answer likely to appear in this "
    "result or in the results ranked right after it?"
)
METHOD_NAMES = {"dense": "dense", "bm25": "BM25"}
RETRYABLE = {429, 500, 502, 503, 529}  # 401 and 422 are not retried (Jev docs)
VERSIONED_MODEL = re.compile(r"jev-\d+\.\d+\.\d+")


def topk_indices(scores, k):
    """Top-k document indices per query, best first, ties broken by corpus
    order (stable sort), matching baseline argmax and gold_ranks."""
    return np.argsort(-np.asarray(scores), axis=1, kind="stable")[:, :k]


def topk_candidates(set_name, k):
    """One record per question with its top-k dense and top-k BM25
    paragraphs, labelled dense_1..k then bm25_1..k."""
    corpus, queries = load_set(set_name)
    bm25, dense = load_scores(set_name)
    tops = {"dense": topk_indices(dense, k), "bm25": topk_indices(bm25, k)}
    rows = []
    for qi, query in enumerate(queries):
        passages = []
        for method in ("dense", "bm25"):
            for rank, doc in enumerate(tops[method][qi], 1):
                passages.append({
                    "slot": f"{method}_{rank}", "method": method, "rank": rank,
                    "id": corpus[doc]["id"], "text": corpus[doc]["text"],
                })
        rows.append({"query_id": query["id"], "question": query["question"],
                     "gold": query["gold"], "passages": passages})
    return rows


def build_request(model, question, passages):
    return {
        "model": model,
        "state": {"question": question, "passages": {p["slot"]: p["text"] for p in passages}},
        "questions": {
            p["slot"]: {
                "type": "score",
                "instructions": INSTRUCTIONS.format(slot=p["slot"], rank=p["rank"],
                                                    method=METHOD_NAMES[p["method"]]),
                "criteria": DAT_RUBRIC,
            }
            for p in passages
        },
    }


def parse_jev(payload, slots, expected_model=None):
    """Return ({slot: judgment}, None) for a valid reply, else (None, reason).
    Valid means: served by the expected version, one score answer per slot,
    score within 0-5, probabilities for exactly levels 0-5."""
    served = payload.get("model")
    if expected_model and served != expected_model:
        return None, f"served by {served}, expected {expected_model}"
    answers = payload.get("answers") or {}
    judgments = {}
    for slot in slots:
        answer = answers.get(slot)
        if not isinstance(answer, dict) or answer.get("type") != "score":
            return None, f"missing score answer for {slot}"
        score = answer.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 5:
            return None, f"{slot} score {score!r} outside 0-5"
        try:
            levels = {int(level): float(p) for level, p in (answer.get("probabilities") or {}).items()}
        except (TypeError, ValueError):
            return None, f"{slot} has malformed probabilities"
        if sorted(levels) != list(range(6)):
            return None, f"{slot} probabilities cover levels {sorted(levels)}, expected 0-5"
        top_level = max(range(6), key=lambda level: (levels[level], -level))  # ties -> lower level
        judgments[slot] = {
            "score": float(score),
            "top_level": top_level,
            "probabilities": [levels[level] for level in range(6)],
            "confidence": answer.get("confidence"),
        }
    return judgments, None


def retry_delay(response, attempt):
    """Honor Retry-After when the server sends it, else exponential backoff."""
    try:
        return float(response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return 2 ** attempt


def call_jev(body, base_url, api_key, retries=4, timeout=60):
    """POST one System One request. Returns {"http_status", "payload" or
    "error", "latency_ms"}; retries only overload/rate-limit/network errors."""
    for attempt in range(retries):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/systemone",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=timeout,
            )
            status = response.status_code
        except requests.RequestException as exc:
            status, response = None, exc
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        if status == 200:
            return {"http_status": 200, "payload": response.json(), "latency_ms": latency_ms}
        if (status is None or status in RETRYABLE) and attempt < retries - 1:
            time.sleep(retry_delay(response, attempt))
            continue
        error = response.text[:500] if status is not None else repr(response)
        return {"http_status": status, "error": error, "latency_ms": latency_ms}


def cache_path(model, set_name, k):
    return CACHE_DIR / re.sub(r"[^\w.-]", "_", model) / f"{set_name}_k{k}.jsonl"


def jev_config():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("JEV_API_KEY")
    if not api_key:
        raise SystemExit("JEV_API_KEY is not set in .env")
    base_url = os.getenv("JEV_API_BASE", "https://api.typesafe.ai/v1").rstrip("/")
    model = os.getenv("JEV_MODEL", "jev-1.13.0")
    return api_key, base_url, model


def judge_set(set_name, k=1, limit=None, sample_seed=0, workers=8):
    """Judge the selected questions with Jev, skipping cached ones. Valid and
    invalid 200 replies are cached (invalid ones carry `error`); HTTP errors
    are not, so a re-run retries them. Stops early if spend would pass the
    budget left under BUDGET_CAP_USD."""
    api_key, base_url, model = jev_config()
    expected_model = model if VERSIONED_MODEL.fullmatch(model) else None
    cache = JudgeCache(cache_path(model, set_name, k))
    selected = select_questions(topk_candidates(set_name, k), limit, sample_seed)
    jobs = []
    for row in selected:
        body = build_request(model, row["question"], row["passages"])
        key = request_key(model, json.dumps(body, sort_keys=True, ensure_ascii=False), {})
        jobs.append((row, body, key))
    todo = [job for job in jobs if cache.get(job[2]) is None]

    estimate = sum(len(json.dumps(body)) / 4 for _, body, _ in todo) * USD_PER_INPUT_TOKEN * 1.5
    print(f"{len(selected)} questions selected (k={k}), {len(jobs) - len(todo)} cached, {len(todo)} to judge")
    if todo:
        budget.ensure_affordable(estimate)

    remaining = float(os.getenv("BUDGET_CAP_USD", "20")) - budget.spent()
    calls, tokens, cost, http_errors, invalid, done = 0, 0, 0.0, 0, 0, 0
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(call_jev, body, base_url, api_key): (row, key) for row, body, key in todo}
        for future in as_completed(futures):
            row, key = futures[future]
            result = future.result()
            done += 1
            if result["http_status"] == 200:
                payload = result["payload"]
                judgments, error = parse_jev(payload, [p["slot"] for p in row["passages"]], expected_model)
                input_tokens = int((payload.get("usage") or {}).get("input_tokens", 0))
                calls += 1
                tokens += input_tokens
                cost += input_tokens * USD_PER_INPUT_TOKEN
                invalid += error is not None
                cache.put({
                    "key": key, "set": set_name, "query_id": row["query_id"], "k": k, "model": model,
                    "served_model": payload.get("model"), "judgments": judgments, "error": error,
                    "input_tokens": input_tokens, "cost_usd": input_tokens * USD_PER_INPUT_TOKEN,
                    "latency_ms": result["latency_ms"],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                if error:
                    print(f"  [{row['query_id']}] invalid reply: {error}")
            else:
                http_errors += 1
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
            budget.record(f"jev_judge_k{k}", model, calls, tokens, cost,
                          note=f"{set_name}, {calls} questions, {workers} workers; "
                               f"cost = input tokens x $0.042/M list price")
    print(f"Judged {calls} new questions in {(time.perf_counter() - started) / 60:.1f} min: "
          f"{tokens} input tokens, ${cost:.6f}, {invalid} invalid replies, "
          f"{http_errors} HTTP errors (not cached, re-run to retry)")
    # Re-read the cache from disk: appended lines can be lost (e.g. a sync
    # client replacing the file mid-run), and the in-memory copy would hide it.
    on_disk = JudgeCache(cache.path)
    lost = sum(on_disk.get(key) is None for _, _, key in jobs if cache.get(key) is not None)
    if lost:
        print(f"WARNING: {lost} replies are missing from {cache.path.name} on disk; re-run this command "
              f"to judge them again (only the missing ones are sent)")
    return [(row, on_disk.get(key)) for row, _, key in jobs]


def print_pilot(results):
    """Per-question view for a small run: Jev's score (most likely level)
    per paragraph, next to which retriever had the gold paragraph at #1."""
    for row, record in results:
        passages = {p["slot"]: p for p in row["passages"]}
        dense_ok = "right" if passages["dense_1"]["id"] == row["gold"] else "wrong"
        bm25_ok = "right" if passages["bm25_1"]["id"] == row["gold"] else "wrong"
        same = " (same paragraph)" if passages["dense_1"]["id"] == passages["bm25_1"]["id"] else ""
        print(f"\n{row['question'][:70]}")
        print(f"  #1 correct? dense {dense_ok}, BM25 {bm25_ok}{same}")
        if record is None or record.get("judgments") is None:
            print(f"  no valid reply: {record.get('error') if record else 'HTTP error'}")
            continue
        for method in ("dense", "bm25"):
            cells = [f"{j['score']:.2f} ({j['top_level']})"
                     for slot, j in record["judgments"].items() if slot.startswith(method)]
            print(f"  {METHOD_NAMES[method]:<5} score (level): {'  '.join(cells)}")


def summarize(results):
    records = [r for _, r in results if r is not None]
    valid = [r for r in records if r.get("judgments") is not None]
    print(f"\nReplies: {len(records)}/{len(results)}, valid: {len(valid)}/{len(records)}")
    print(f"Served model(s): {sorted({str(r.get('served_model')) for r in records})}")
    if not records:
        return
    tokens = [r["input_tokens"] for r in records]
    latency = sorted(r["latency_ms"] for r in records)
    per_question = statistics.mean(r["cost_usd"] for r in records)
    print(f"Input tokens per request: mean {statistics.mean(tokens):.0f}, max {max(tokens)}")
    print(f"Cost per question: ${per_question:.6f} -> test (2,996) ~ ${per_question * 2996:.4f}, "
          f"tune (2,865) ~ ${per_question * 2865:.4f}")
    print(f"Latency per request: median {latency[len(latency) // 2]:.0f} ms, max {latency[-1]:.0f} ms")


def main():
    parser = argparse.ArgumentParser(description="JevFusion judge (DAT rubric via Jev)")
    parser.add_argument("--set", choices=["test", "tune"], required=True)
    parser.add_argument("--k", type=int, choices=[1, 3], default=1, help="paragraphs judged per retriever")
    parser.add_argument("--limit", type=int, help="judge a reproducible random sample of this many questions")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="parallel requests")
    args = parser.parse_args()

    results = judge_set(args.set, args.k, args.limit, args.sample_seed, args.workers)
    if args.limit and args.limit <= 50:
        print_pilot(results)
    summarize(results)


if __name__ == "__main__":
    main()
