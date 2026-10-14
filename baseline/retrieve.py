"""
Step 1.2 of the DAT reproduction: score every paragraph for every question
with BM25 and with dense retrieval, and cache the raw score matrices.

- BM25 (paper Eq. 1): rank_bm25's BM25Okapi with its defaults (k1=1.5,
  b=0.75) over lowercase word tokens. DAT only says "standard word
  tokenization", so the exact tokenizer is a documented deviation.
- Dense (paper Eq. 2): cosine similarity of text-embedding-3-large
  vectors (DAT §5.1), called through OpenRouter.

Each sampled corpus has only ~600 paragraphs, so both retrievers score the
whole corpus rather than a top-N list. Min-max normalization (Eq. 3-4)
then runs over the whole corpus too, and no document is ever "missing"
from one retriever's list.

    python -m baseline.retrieve --set test
    python -m baseline.retrieve --set tune

Writes data/cache/<set>_{bm25,dense}.npy, float arrays of shape
(questions, paragraphs) in the row/column order of queries.jsonl and
corpus.jsonl. Embeddings are cached by text hash in
data/cache/embeddings_<model>.npz, so nothing is ever paid for twice.
"""

import argparse
import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from baseline import budget
from baseline.prepare_squad import load_set

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"

EMBED_BATCH = 128
# OpenRouter list price for openai/text-embedding-3-large, checked 2026-10-15.
# Only used for the pre-run estimate; the ledger records the billed cost.
EMBED_USD_PER_TOKEN = 0.13 / 1_000_000


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def bm25_scores(corpus, queries):
    bm25 = BM25Okapi([tokenize(doc["text"]) for doc in corpus])
    return np.vstack([bm25.get_scores(tokenize(q["question"])) for q in queries])


def text_key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Text-hash -> vector store, persisted in one .npz per model."""

    def __init__(self, model):
        slug = re.sub(r"[^\w.-]", "_", model)
        self.path = CACHE_DIR / f"embeddings_{slug}.npz"
        self.vectors = {}
        if self.path.exists():
            data = np.load(self.path)
            self.vectors = dict(zip(data["keys"].tolist(), data["vectors"]))

    def missing(self, texts):
        seen, out = set(), []
        for text in texts:
            key = text_key(text)
            if key not in self.vectors and key not in seen:
                seen.add(key)
                out.append(text)
        return out

    def add(self, texts, vectors):
        for text, vector in zip(texts, vectors):
            self.vectors[text_key(text)] = np.asarray(vector, dtype=np.float32)

    def save(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        keys = list(self.vectors)
        np.savez(self.path, keys=np.array(keys), vectors=np.vstack([self.vectors[k] for k in keys]))

    def matrix(self, texts):
        return np.vstack([self.vectors[text_key(t)] for t in texts])


def embed_batch(texts, model, base_url, api_key, retries=4):
    """One OpenRouter embeddings call; returns (vectors, usage dict)."""
    for attempt in range(retries):
        response = requests.post(
            f"{base_url}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": texts},
            timeout=120,
        )
        if response.status_code == 200:
            payload = response.json()
            data = sorted(payload["data"], key=lambda item: item["index"])
            if len(data) != len(texts):
                raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(data)}")
            return [item["embedding"] for item in data], payload.get("usage", {})
        if response.status_code in (429, 500, 502, 503, 529) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"embeddings HTTP {response.status_code}: {response.text[:500]}")


def ensure_embeddings(texts, model, base_url, api_key, cache):
    todo = cache.missing(texts)
    if not todo:
        return
    estimate = sum(len(t) for t in todo) / 4 * EMBED_USD_PER_TOKEN * 1.5  # ~4 chars/token, +50% margin
    budget.ensure_affordable(estimate)
    calls, tokens, cost = 0, 0, 0.0
    try:
        for start in tqdm(range(0, len(todo), EMBED_BATCH), desc=f"embedding {len(todo)} texts"):
            batch = todo[start:start + EMBED_BATCH]
            vectors, usage = embed_batch(batch, model, base_url, api_key)
            cache.add(batch, vectors)
            calls += 1
            tokens += int(usage.get("prompt_tokens", 0))
            cost += float(usage.get("cost", 0.0))
    finally:
        cache.save()
        if calls:
            budget.record("dense_embeddings", model, calls, tokens, cost, note=f"{len(todo)} new texts")
    print(f"Embedded {len(todo)} texts in {calls} calls: {tokens} tokens, ${cost:.6f}")


def dense_scores(corpus, queries, model, base_url, api_key):
    cache = EmbeddingCache(model)
    doc_texts = [doc["text"] for doc in corpus]
    query_texts = [q["question"] for q in queries]
    ensure_embeddings(doc_texts + query_texts, model, base_url, api_key, cache)
    docs, qs = cache.matrix(doc_texts), cache.matrix(query_texts)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    qs = qs / np.linalg.norm(qs, axis=1, keepdims=True)
    return qs @ docs.T


def score_path(set_name, retriever):
    return CACHE_DIR / f"{set_name}_{retriever}.npy"


def load_scores(set_name):
    """Return (bm25, dense) score matrices for a set, as cached by main()."""
    paths = [score_path(set_name, r) for r in ("bm25", "dense")]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{path} missing, run: python -m baseline.retrieve --set {set_name}")
    return tuple(np.load(path) for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", choices=["test", "tune"], required=True)
    parser.add_argument("--bm25-only", action="store_true", help="skip the paid dense step")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    corpus, queries = load_set(args.set)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    bm25 = bm25_scores(corpus, queries)
    np.save(score_path(args.set, "bm25"), bm25)
    print(f"BM25 scores: {bm25.shape} -> {score_path(args.set, 'bm25').relative_to(ROOT)}")
    if args.bm25_only:
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set in .env")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.getenv("DAT_EMBED_MODEL", "openai/text-embedding-3-large")
    dense = dense_scores(corpus, queries, model, base_url, api_key)
    np.save(score_path(args.set, "dense"), dense)
    print(f"Dense scores: {dense.shape} -> {score_path(args.set, 'dense').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
