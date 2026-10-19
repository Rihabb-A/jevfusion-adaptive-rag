"""
Phase 4: the BEIR NFCorpus experiment (data, retrieval, fusion, evaluation).

Datasets are named `nfcorpus-dev` and `nfcorpus-test` everywhere (cache
files, judge caches, ledger notes, result files), so nothing can collide
with the SQuAD `tune` / `test` runs. All score matrices and the NFCorpus
embedding cache live under data/cache/nfcorpus/; the shared SQuAD
embedding cache file is never opened for writing.

Approved setup (PROGRESS.md §12): index title + text for both retrievers
(the same BM25 implementation and settings as SQuAD, and
text-embedding-3-large), score the full 3,633-document corpus, dev for
any tuning, test for final results, nDCG@10 primary.

    python -m extension.nfcorpus retrieve --set nfcorpus-dev --bm25-only
    python -m extension.nfcorpus retrieve --set nfcorpus-test            # + dense (paid)
    python -m extension.nfcorpus baselines                               # N4
"""

import argparse
import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from baseline import budget
from baseline.dat_fusion import fuse, minmax
from baseline.retrieve import EmbeddingCache, bm25_scores, embed_batch
from extension.beir_data import doc_text, load_beir
from extension.graded_metrics import per_query_metrics, summarize
from extension.rrf import rrf_scores

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "nfcorpus"
RESULTS = ROOT / "results" / "nfcorpus"
SETS = {"nfcorpus-dev": "dev", "nfcorpus-test": "test"}
NFCORPUS_CAP_USD = 2.0  # approved ceiling for additional NFCorpus spending
EMBED_BATCH = 64
# OpenRouter list price for openai/text-embedding-3-large (checked 2026-10-15).
EMBED_USD_PER_TOKEN = 0.13 / 1_000_000
FIXED_ALPHA = 0.6  # transferred from the SQuAD tune set, not re-tuned
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.1), 1)


@lru_cache(maxsize=None)
def _beir():
    return load_beir("nfcorpus")


def load_split(set_name):
    """(docs, queries, qrels) for `nfcorpus-dev` / `nfcorpus-test`.

    docs: [{"id", "title", "text"}] in corpus order, `text` = title + text
    (the indexed text). queries: [{"id", "question"}] sorted by query id,
    restricted to queries judged in this split. qrels: {qid: {doc_id: grade}}."""
    if set_name not in SETS:
        raise ValueError(f"unknown NFCorpus set {set_name!r}; expected one of {sorted(SETS)}")
    corpus, queries, splits = _beir()
    qrels = splits[SETS[set_name]]
    docs = [{"id": d["_id"], "title": d.get("title", ""), "text": doc_text(d)} for d in corpus]
    qs = [{"id": q, "question": queries[q]} for q in sorted(qrels)]
    return docs, qs, qrels


def fingerprint(items):
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]


def score_path(set_name, retriever):
    return CACHE / f"{set_name}_{retriever}.npy"


def meta_path(set_name):
    return CACHE / f"{set_name}_meta.json"


class NFCorpusEmbeddingCache(EmbeddingCache):
    """The baseline's text-hash embedding cache, stored in its own file under
    data/cache/nfcorpus/ so the shared SQuAD cache file is never rewritten."""

    def __init__(self, model):  # noqa: super().__init__ would open the SQuAD file
        slug = re.sub(r"[^\w.-]", "_", model)
        self.path = CACHE / f"embeddings_{slug}.npz"
        self.vectors = {}
        if self.path.exists():
            data = np.load(self.path)
            self.vectors = dict(zip(data["keys"].tolist(), data["vectors"]))


def nfcorpus_spent():
    """USD spent so far on NFCorpus (ledger rows whose note names it)."""
    import csv
    if not budget.LEDGER.exists():
        return 0.0
    with budget.LEDGER.open(encoding="utf-8", newline="") as handle:
        return sum(float(r["cost_usd"]) for r in csv.DictReader(handle) if "nfcorpus" in r["note"].lower())


def ensure_nfcorpus_budget(estimate_usd):
    """Stop before a run that could take NFCorpus spending past its cap,
    then apply the project-wide $20 check."""
    spent = nfcorpus_spent()
    if spent + estimate_usd > NFCORPUS_CAP_USD:
        raise SystemExit(f"Refusing: NFCorpus spend ${spent:.4f} + estimate ${estimate_usd:.4f} "
                         f"exceeds the ${NFCORPUS_CAP_USD:.2f} NFCorpus cap")
    budget.ensure_affordable(estimate_usd)
    print(f"NFCorpus budget: spent ${spent:.4f}, this run est. ${estimate_usd:.4f}, cap ${NFCORPUS_CAP_USD:.2f}")


def embed_texts(texts, model, base_url, api_key, cache, label):
    """Embed texts not yet cached, saving after every few batches; the
    ledger note names the NFCorpus set. Returns (calls, tokens, cost)."""
    todo = cache.missing(texts)
    if not todo:
        return 0, 0, 0.0
    estimate = sum(len(t) for t in todo) / 4 * EMBED_USD_PER_TOKEN * 1.5  # ~4 chars/token, +50%
    print(f"{len(todo)} texts to embed, ~{sum(len(t) for t in todo) / 4:,.0f} tokens")
    ensure_nfcorpus_budget(estimate)
    calls, tokens, cost = 0, 0, 0.0
    try:
        for start in range(0, len(todo), EMBED_BATCH):
            batch = todo[start:start + EMBED_BATCH]
            vectors, usage = embed_batch(batch, model, base_url, api_key)
            cache.add(batch, vectors)
            calls += 1
            tokens += int(usage.get("prompt_tokens", 0))
            cost += float(usage.get("cost", 0.0))
            if calls % 10 == 0:
                cache.save()
                print(f"  {start + len(batch)}/{len(todo)} embedded, ${cost:.4f}")
    finally:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.save()
        if calls:
            budget.record("dense_embeddings", model, calls, tokens, cost, note=f"{label}: {calls} batches")
    reloaded = NFCorpusEmbeddingCache(model)  # verify what reached the disk
    if cache.vectors.keys() - reloaded.vectors.keys():
        raise SystemExit(f"{len(cache.vectors.keys() - reloaded.vectors.keys())} embeddings missing on disk; re-run")
    print(f"Embedded {len(todo)} texts in {calls} calls: {tokens} tokens, ${cost:.6f} (verified on disk)")
    return calls, tokens, cost


def retrieve(set_name, bm25_only=False):
    """Full-corpus BM25 (and dense) score matrices for one NFCorpus set."""
    docs, qs, _ = load_split(set_name)
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bm25 = bm25_scores(docs, qs)
    np.save(score_path(set_name, "bm25"), bm25)
    print(f"BM25 {set_name}: {bm25.shape} in {time.perf_counter() - started:.1f}s -> {score_path(set_name, 'bm25').relative_to(ROOT)}")
    meta = {"set": set_name, "n_queries": len(qs), "n_docs": len(docs),
            "query_ids_sha": fingerprint([q["id"] for q in qs]), "doc_ids_sha": fingerprint([d["id"] for d in docs]),
            "indexed_text": "title + ' ' + text", "bm25": "rank_bm25.BM25Okapi k1=1.5 b=0.75, lowercase \\w+ tokens (as SQuAD)"}
    if not bm25_only:
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set in .env")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        model = os.getenv("DAT_EMBED_MODEL", "openai/text-embedding-3-large")
        cache = NFCorpusEmbeddingCache(model)
        doc_texts, query_texts = [d["text"] for d in docs], [q["question"] for q in qs]
        embed_texts(doc_texts + query_texts, model, base_url, api_key, cache, label=f"nfcorpus {set_name} corpus+queries")
        d_vec, q_vec = cache.matrix(doc_texts), cache.matrix(query_texts)
        d_vec = d_vec / np.linalg.norm(d_vec, axis=1, keepdims=True)
        q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
        dense = q_vec @ d_vec.T
        np.save(score_path(set_name, "dense"), dense)
        meta["dense"] = f"{model} cosine"
        print(f"Dense {set_name}: {dense.shape} -> {score_path(set_name, 'dense').relative_to(ROOT)}")
    meta_path(set_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_scores(set_name):
    """(bm25, dense) matrices, checked against the current query/doc order."""
    docs, qs, _ = load_split(set_name)
    meta = json.loads(meta_path(set_name).read_text(encoding="utf-8"))
    if (meta["query_ids_sha"], meta["doc_ids_sha"]) != (fingerprint([q["id"] for q in qs]), fingerprint([d["id"] for d in docs])):
        raise SystemExit(f"{set_name}: cached scores do not match the current query/document order")
    return np.load(score_path(set_name, "bm25")), np.load(score_path(set_name, "dense"))


def method_scores(set_name, alphas=None):
    """Score matrices for the zero-judge methods plus any {name: alpha}."""
    bm25, dense = load_scores(set_name)
    norm_dense, norm_bm25 = minmax(dense), minmax(bm25)
    out = {"BM25 only": bm25, "Dense only": dense, "RRF (k=60)": rrf_scores(dense, bm25)}
    for name, alpha in (alphas or {}).items():
        out[name] = fuse(norm_dense, norm_bm25, alpha)
    return out


def evaluate(set_name, scores_by_method):
    """{method: per-query metric arrays} with the N2 graded metrics."""
    docs, qs, qrels = load_split(set_name)
    doc_ids, query_ids = [d["id"] for d in docs], [q["id"] for q in qs]
    return {name: per_query_metrics(s, query_ids, doc_ids, qrels) for name, s in scores_by_method.items()}


def tune_alpha(set_name="nfcorpus-dev"):
    """Dev-only grid search: highest nDCG@10, then Recall@10, then the alpha
    closest to the transferred 0.6 (rule fixed before running)."""
    per = evaluate(set_name, method_scores(set_name, {f"alpha={a:.1f}": a for a in ALPHAS}))
    grid = [(a, summarize(per[f"alpha={a:.1f}"])) for a in ALPHAS]
    best = max(grid, key=lambda item: (item[1]["ndcg@10"], item[1]["recall@10"], -abs(item[0] - FIXED_ALPHA)))
    return float(best[0]), grid


def write_rows(rows, path):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"Written to {path.relative_to(ROOT)}")


def baselines():
    """N4: zero-judge baselines on dev and test; alpha tuned on dev only."""
    alpha_dev, grid = tune_alpha("nfcorpus-dev")
    print("Dev alpha grid (selection by dev nDCG@10):")
    for a, m in grid:
        print(f"  alpha={a:.1f}  " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
    print(f"-> dev-tuned alpha = {alpha_dev:.1f} (transferred SQuAD alpha = {FIXED_ALPHA})")
    rows = []
    for set_name in SETS:
        methods = {f"Fixed alpha={FIXED_ALPHA} (SQuAD)": FIXED_ALPHA, f"Fixed alpha={alpha_dev:.1f} (dev-tuned)": alpha_dev}
        per = evaluate(set_name, method_scores(set_name, methods))
        for name, metrics in per.items():
            rows.append({"set": set_name, "method": name, "n_queries": len(metrics["ndcg@10"]), **summarize(metrics)})
    write_rows(rows, RESULTS / "baselines.csv")
    write_rows([{"alpha": a, **m} for a, m in grid], RESULTS / "dev_alpha_grid.csv")
    return alpha_dev, rows


def main():
    parser = argparse.ArgumentParser(description="NFCorpus experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--set", choices=sorted(SETS), required=True)
    r.add_argument("--bm25-only", action="store_true")
    sub.add_parser("baselines")
    args = parser.parse_args()
    if args.command == "retrieve":
        retrieve(args.set, args.bm25_only)
    elif args.command == "baselines":
        baselines()


if __name__ == "__main__":
    main()
