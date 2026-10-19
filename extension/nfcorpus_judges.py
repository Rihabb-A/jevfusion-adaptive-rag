"""
Phase 4, N5-N8: the DAT (GPT-4o-mini) and Jev judges on NFCorpus, reusing
the SQuAD judge code unchanged: same prompt and settings
(`baseline.dat_judge`), same Jev request format, validation and caching
(`extension.jev_judge`), same alpha rules (DAT Eq. 6; Jev `top_level`
mapping for J1 and option C for J3, both chosen on SQuAD's tune set and
not re-tuned here).

Only the candidates are new: the top-k documents of each NFCorpus
retriever (title + text, the indexed text), from the N3 score matrices,
with the same stable tie-breaking as SQuAD. Judge caches are named by the
NFCorpus set (`nfcorpus-dev`, `nfcorpus-test`), so they can never mix with
the SQuAD caches.

    python -m extension.nfcorpus_judges check-sizes
    python -m extension.nfcorpus_judges dat --set nfcorpus-dev --limit 20   # pilot
    python -m extension.nfcorpus_judges dat --set nfcorpus-test
    python -m extension.nfcorpus_judges jev --k 1 --set nfcorpus-test
"""

import argparse
import json
import os

import numpy as np
from dotenv import load_dotenv

from baseline import dat_judge
from baseline.dat_fusion import dat_alpha
from extension import jev_judge
from extension.jev_fusion import CHOSEN_J3_OPTION, CHOSEN_MAPPING, j1_alphas, load_judgments, multi_alphas
from extension.jev_judge import topk_indices
from extension.nfcorpus import ROOT, SETS, ensure_nfcorpus_budget, load_scores, load_split

# Context limits checked before any paid run.
GPT4O_MINI_CONTEXT = 128_000  # tokens
JEV_REQUEST_LIMIT, JEV_QUESTION_LIMIT = 64_000, 32_000  # tokens (docs.typesafe.ai/models)
# Pre-run cost estimates use ~4 characters per token; Jev counted ~25% more
# tokens than that on SQuAD, so its estimate gets a larger margin.
DAT_USD_PER_TOKEN, JEV_USD_PER_TOKEN = 0.15 / 1e6, 0.042 / 1e6


def build_dat_candidates(docs, queries, bm25, dense):
    """DAT rows (top-1 per retriever) from score matrices; ties -> corpus order."""
    dense_top, bm25_top = topk_indices(dense, 1)[:, 0], topk_indices(bm25, 1)[:, 0]
    return [{
        "query_id": q["id"], "question": q["question"],
        "dense_top1": docs[d]["id"], "dense_text": docs[d]["text"],
        "bm25_top1": docs[b]["id"], "bm25_text": docs[b]["text"],
        "same_top1": bool(d == b),
    } for q, d, b in zip(queries, dense_top, bm25_top)]


def build_jev_candidates(docs, queries, bm25, dense, k):
    """Jev rows: dense_1..k then bm25_1..k, each with document id and text.
    A document in both lists is kept under both labels (as on SQuAD)."""
    tops = {"dense": topk_indices(dense, k), "bm25": topk_indices(bm25, k)}
    rows = []
    for i, q in enumerate(queries):
        passages = [{"slot": f"{m}_{r}", "method": m, "rank": r, "id": docs[j]["id"], "text": docs[j]["text"]}
                    for m in ("dense", "bm25") for r, j in enumerate(tops[m][i], 1)]
        rows.append({"query_id": q["id"], "question": q["question"], "passages": passages})
    return rows


def dat_candidates(set_name):
    docs, qs, _ = load_split(set_name)
    bm25, dense = load_scores(set_name)
    return build_dat_candidates(docs, qs, bm25, dense)


def jev_candidates(set_name, k):
    docs, qs, _ = load_split(set_name)
    bm25, dense = load_scores(set_name)
    return build_jev_candidates(docs, qs, bm25, dense, k)


def request_sizes(set_name):
    """Approximate token counts (characters / 4) of every request."""
    model = os.getenv("JEV_MODEL", "jev-1.13.0")
    dat = [len(dat_judge.build_prompt(r["question"], r["dense_text"], r["bm25_text"])) / 4
           for r in dat_candidates(set_name)]
    jev = {}
    for k in (1, 3):
        sizes = []
        for row in jev_candidates(set_name, k):
            body = jev_judge.build_request(model, row["question"], row["passages"])
            state = len(json.dumps(body["state"])) / 4
            longest_q = max(len(json.dumps(q)) for q in body["questions"].values()) / 4
            sizes.append((len(json.dumps(body)) / 4, state + longest_q))
        jev[k] = sizes
    return dat, jev


def check_sizes():
    load_dotenv(ROOT / ".env")
    for set_name in SETS:
        dat, jev = request_sizes(set_name)
        print(f"{set_name}: DAT prompt ~tokens mean {np.mean(dat):.0f}, max {max(dat):.0f} (limit {GPT4O_MINI_CONTEXT:,})")
        for k, sizes in jev.items():
            total, per_q = zip(*sizes)
            print(f"  Jev k={k}: request mean {np.mean(total):.0f}, max {max(total):.0f} (limit {JEV_REQUEST_LIMIT:,}); "
                  f"state + longest question max {max(per_q):.0f} (limit {JEV_QUESTION_LIMIT:,})")
        assert max(dat) < GPT4O_MINI_CONTEXT * 0.8, "DAT prompt too long"
        for sizes in jev.values():
            assert max(t for t, _ in sizes) < JEV_REQUEST_LIMIT * 0.8 and max(q for _, q in sizes) < JEV_QUESTION_LIMIT * 0.8
    print("All requests are well within the context limits (no truncation needed).")


def run_dat(set_name, limit=None, workers=8):
    candidates = dat_candidates(set_name)
    selected = dat_judge.select_questions(candidates, limit, 0)
    tokens = sum(len(dat_judge.build_prompt(r["question"], r["dense_text"], r["bm25_text"])) / 4 for r in selected)
    ensure_nfcorpus_budget(tokens * DAT_USD_PER_TOKEN * 1.5 + len(selected) * 5 * 0.6e-6)
    results = dat_judge.judge_set(set_name, limit=limit, sample_seed=0, workers=workers, candidates=candidates)
    return verify_on_disk(dat_judge.JudgeCache(dat_cache_path(set_name)), results, "DAT")


def dat_cache_path(set_name):
    load_dotenv(ROOT / ".env")
    return dat_judge.cache_path(os.getenv("DAT_JUDGE_MODEL", "openai/gpt-4o-mini"), set_name)


def run_jev(set_name, k, limit=None, workers=8):
    candidates = jev_candidates(set_name, k)
    _, _, model = jev_judge.jev_config()
    selected = dat_judge.select_questions(candidates, limit, 0)
    tokens = sum(len(json.dumps(jev_judge.build_request(model, r["question"], r["passages"]))) / 4 for r in selected)
    ensure_nfcorpus_budget(tokens * JEV_USD_PER_TOKEN * 2.0)
    results = jev_judge.judge_set(set_name, k=k, limit=limit, sample_seed=0, workers=workers, candidates=candidates)
    return verify_on_disk(dat_judge.JudgeCache(jev_judge.cache_path(model, set_name, k)), results, f"Jev k={k}")


def verify_on_disk(on_disk, results, label):
    """Every selected question must have a reply in the cache file itself."""
    keys = [r["key"] for _, r in results if r is not None]
    missing_in_run = sum(r is None for _, r in results)
    missing_on_disk = sum(on_disk.get(key) is None for key in keys)
    print(f"{label}: {len(results)} selected, {len(keys)} replies, {missing_in_run} without reply, "
          f"{missing_on_disk} missing on disk")
    if missing_in_run or missing_on_disk:
        raise SystemExit("Incomplete judge cache; re-run the same command (only missing questions are sent)")
    return results


def dat_alphas_nf(set_name):
    """Per-query DAT alpha (Eq. 6) from cached GPT-4o-mini replies, in the
    evaluation order; stops if any reply is missing or unparseable."""
    load_dotenv(ROOT / ".env")
    model = os.getenv("DAT_JUDGE_MODEL", "openai/gpt-4o-mini")
    cache = dat_judge.JudgeCache(dat_judge.cache_path(model, set_name))
    alphas, scores = [], []
    for row in dat_candidates(set_name):
        prompt = dat_judge.build_prompt(row["question"], row["dense_text"], row["bm25_text"])
        record = cache.get(dat_judge.request_key(model, prompt, dat_judge.JUDGE_SETTINGS))
        if record is None or record.get("parsed") is None:
            raise SystemExit(f"No parsed DAT reply for {row['query_id']} in {set_name}")
        scores.append(tuple(record["parsed"]))
        alphas.append(dat_alpha(*record["parsed"]))
    return np.array(alphas), scores


def jev_alphas_nf(set_name, k):
    """Per-query Jev alpha with the SQuAD-chosen rule: `top_level` + DAT Eq. 6
    for k = 1, option C for k = 3. Returns (alphas, judgments)."""
    judgments = load_judgments(set_name, k, candidates=jev_candidates(set_name, k))
    alphas = j1_alphas(judgments, CHOSEN_MAPPING) if k == 1 else multi_alphas(judgments, CHOSEN_J3_OPTION)
    return alphas, judgments


def main():
    parser = argparse.ArgumentParser(description="DAT and Jev judges on NFCorpus")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-sizes")
    for name in ("dat", "jev"):
        p = sub.add_parser(name)
        p.add_argument("--set", choices=sorted(SETS), required=True)
        p.add_argument("--limit", type=int)
        p.add_argument("--workers", type=int, default=8)
        if name == "jev":
            p.add_argument("--k", type=int, choices=[1, 3], required=True)
    args = parser.parse_args()
    if args.command == "check-sizes":
        check_sizes()
    elif args.command == "dat":
        run_dat(args.set, args.limit, args.workers)
    else:
        run_jev(args.set, args.k, args.limit, args.workers)


if __name__ == "__main__":
    main()
