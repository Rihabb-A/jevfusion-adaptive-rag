"""
Phase 4 analyses for the NFCorpus experiment, one subcommand per step.
Every number is recomputed from the cached score matrices, judge caches
and the official relevance labels; nothing here calls an API.

    python -m extension.nfcorpus_analysis retrieval      # N3
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from baseline.evaluate import paired_bootstrap_ci
from extension.graded_metrics import per_query_metrics, recall_at_k, summarize
from extension.jev_judge import topk_indices
from extension.nfcorpus import RESULTS, SETS, load_scores, load_split, write_rows

ROOT = Path(__file__).resolve().parent.parent

# Anserini BM25 "flat" (title + text in one field) on BEIR v1.0.0 NFCorpus
# test, from docs/reproduce/from-document-collection/beir-v1.0.0-nfcorpus.flat.md
# (castorini/anserini, master, fetched 2026-10-20); multifield: 0.3254.
ANSERINI_FLAT = {"ndcg@10": 0.3218, "recall@100": 0.2457, "recall@1000": 0.3704}
LUCENE_STOPWORDS = set("a an and are as at be but by for if in into is it no not of on or such "
                       "that the their then there these they this to was will with".split())


def recall_at(scores, qids, dids, qrels, k, positive_only=False):
    top = topk_indices(scores, k)
    out = []
    for i, q in enumerate(qids):
        cut = [j for j in top[i] if not positive_only or scores[i, j] > 0]
        out.append(recall_at_k([qrels[q].get(dids[j], 0) for j in cut], sum(g >= 1 for g in qrels[q].values()), k))
    return float(np.mean(out))


def positive_only_metrics(scores, qids, dids, qrels):
    """Metrics when documents with score 0 count as not retrieved (as in an
    inverted-index run, which only returns matching documents)."""
    masked = np.where(scores > 0, scores, -np.inf)
    per = per_query_metrics(masked, qids, dids, qrels)
    for i in np.where((scores > 0).sum(axis=1) < 10)[0]:
        positive = [j for j in topk_indices(scores[i:i + 1], 10)[0] if scores[i, j] > 0]
        ranked = [qrels[qids[i]].get(dids[j], 0) for j in positive]
        judged = qrels[qids[i]].values()
        from extension.graded_metrics import mrr_at_k, ndcg_at_k, precision_at_1
        per["ndcg@10"][i] = ndcg_at_k(ranked, judged)
        per["recall@10"][i] = recall_at_k(ranked, sum(g >= 1 for g in judged))
        per["mrr@10"][i] = mrr_at_k(ranked)
        per["p@1"][i] = precision_at_1(ranked)
    return per


def bm25_variant(docs, qs, k1, b, stop=False, stem=False, lucene_idf=False):
    """BM25 diagnostics only (the experiment keeps SQuAD's BM25)."""
    stemmer = None
    if stem:
        import snowballstemmer  # optional; installed only for diagnostics
        stemmer = snowballstemmer.stemmer("porter")

    def tok(text):
        words = re.findall(r"\w+", text.lower())
        if stop:
            words = [w for w in words if w not in LUCENE_STOPWORDS]
        return stemmer.stemWords(words) if stemmer else words

    doc_toks = [tok(d["text"]) for d in docs]
    if not lucene_idf:
        from rank_bm25 import BM25Okapi
        bm = BM25Okapi(doc_toks, k1=k1, b=b)
        return np.vstack([bm.get_scores(tok(q["question"])) for q in qs])
    # Lucene BM25: idf = ln(1 + (N - n + 0.5) / (n + 0.5)), tf / (tf + k1 (1 - b + b dl / avgdl))
    n_docs = len(doc_toks)
    lengths = np.array([len(t) for t in doc_toks], dtype=float)
    norm = k1 * (1 - b + b * lengths / lengths.mean())
    postings = {}
    for j, toks in enumerate(doc_toks):
        for term, tf in Counter(toks).items():
            postings.setdefault(term, []).append((j, tf))
    out = np.zeros((len(qs), n_docs))
    for i, q in enumerate(qs):
        for term in tok(q["question"]):
            plist = postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            idx = np.array([j for j, _ in plist])
            tf = np.array([t for _, t in plist], dtype=float)
            out[i, idx] += idf * tf / (tf + norm[idx])
    return out


def retrieval():
    """N3: BM25 and dense quality, reference comparison, BM25 diagnostics,
    query-type analysis and examples."""
    rows = []
    per_test = {}
    for set_name in SETS:
        docs, qs, qrels = load_split(set_name)
        dids, qids = [d["id"] for d in docs], [q["id"] for q in qs]
        bm25, dense = load_scores(set_name)
        for name, s in (("BM25", bm25), ("Dense", dense)):
            per = per_query_metrics(s, qids, dids, qrels)
            if set_name == "nfcorpus-test":
                per_test[name] = per
            rows.append({"set": set_name, "method": name, **summarize(per),
                         "recall@100": recall_at(s, qids, dids, qrels, 100),
                         "recall@1000": recall_at(s, qids, dids, qrels, 1000)})
    print("== N3 retrieval quality ==")
    for r in rows:
        print(f"  {r['set']:<14} {r['method']:<6} nDCG@10 {r['ndcg@10']:.4f}  R@10 {r['recall@10']:.4f}  "
              f"MRR@10 {r['mrr@10']:.4f}  P@1 {r['p@1']:.4f}  R@100 {r['recall@100']:.4f}  R@1000 {r['recall@1000']:.4f}")
    write_rows(rows, RESULTS / "retrieval.csv")

    docs, qs, qrels = load_split("nfcorpus-test")
    dids, qids = [d["id"] for d in docs], [q["id"] for q in qs]
    bm25, dense = load_scores("nfcorpus-test")
    print(f"\n== BM25 vs Anserini flat (test): reference {ANSERINI_FLAT} ==")
    pos = positive_only_metrics(bm25, qids, dids, qrels)
    print(f"  ours, zero-score docs counted as not retrieved: nDCG@10 {summarize(pos)['ndcg@10']:.4f}  "
          f"R@100 {recall_at(bm25, qids, dids, qrels, 100, True):.4f}  R@1000 {recall_at(bm25, qids, dids, qrels, 1000, True):.4f}")
    matched = (bm25 > 0).sum(axis=1)
    print(f"  queries matching < 10 docs: {(matched < 10).sum()}, matching none: {(matched == 0).sum()}")
    diag = []
    variants = [
        ("ours: k1=1.5 b=0.75, no stop/stem (SQuAD setting)", dict(k1=1.5, b=0.75)),
        ("k1=0.9 b=0.4 (Anserini defaults)", dict(k1=0.9, b=0.4)),
        ("+ Lucene stopwords", dict(k1=0.9, b=0.4, stop=True)),
        ("+ Porter stemming", dict(k1=0.9, b=0.4, stop=True, stem=True)),
        ("+ Lucene idf/tf formula", dict(k1=0.9, b=0.4, stop=True, stem=True, lucene_idf=True)),
    ]
    for set_name in SETS:
        d2, q2, r2 = load_split(set_name)
        dd, qq = [d["id"] for d in d2], [q["id"] for q in q2]
        for label, kw in variants:
            try:
                s = bm25_variant(d2, q2, **kw)
            except ImportError:
                print(f"  (skipped {label}: snowballstemmer not installed)")
                continue
            m = summarize(positive_only_metrics(s, qq, dd, r2))
            diag.append({"set": set_name, "variant": label, "ndcg@10_matched_only": m["ndcg@10"],
                         "ndcg@10_all_ranked": summarize(per_query_metrics(s, qq, dd, r2))["ndcg@10"],
                         "recall@100_matched_only": recall_at(s, qq, dd, r2, 100, True)})
    print("\n== BM25 diagnostics (nDCG@10; 'matched only' = zero-score docs not retrieved) ==")
    for d in diag:
        print(f"  {d['set']:<14} {d['variant']:<52} all-ranked {d['ndcg@10_all_ranked']:.4f}  "
              f"matched-only {d['ndcg@10_matched_only']:.4f}  R@100 {d['recall@100_matched_only']:.4f}")
    write_rows(diag, RESULTS / "bm25_diagnostics.csv")

    print("\n== Dense vs BM25 on test (paired, 10,000 resamples) ==")
    for metric in ("ndcg@10", "recall@10", "mrr@10", "p@1"):
        mean, low, high = paired_bootstrap_ci(per_test["Dense"][metric], per_test["BM25"][metric])
        print(f"  {metric:<10} dense - BM25 {mean:+.4f} [{low:+.4f}, {high:+.4f}]")
    diff = per_test["Dense"]["ndcg@10"] - per_test["BM25"]["ndcg@10"]
    print(f"  per query nDCG@10: dense better {np.sum(diff > 0)}, BM25 better {np.sum(diff < 0)}, tie {np.sum(diff == 0)}")

    words = np.array([len(q["question"].split()) for q in qs])
    n_rel = np.array([sum(g >= 1 for g in qrels[q].values()) for q in qids])
    print("\n== By query type (test, mean nDCG@10) ==")
    for label, mask in (("1 word", words == 1), ("2-3 words", (words >= 2) & (words <= 3)), ("4+ words", words >= 4),
                        ("BM25 matches < 10 docs", matched < 10), ("1-5 relevant", n_rel <= 5),
                        ("6-30 relevant", (n_rel > 5) & (n_rel <= 30)), ("31+ relevant", n_rel > 30)):
        print(f"  {label:<24} n={mask.sum():>3}  BM25 {per_test['BM25']['ndcg@10'][mask].mean():.4f}  "
              f"Dense {per_test['Dense']['ndcg@10'][mask].mean():.4f}  dense better on {np.mean(diff[mask] > 0):.0%}")

    top_b, top_d = topk_indices(bm25, 10), topk_indices(dense, 10)
    overlap = [len(set(top_b[i]) & set(top_d[i])) for i in range(len(qids))]
    print(f"\nTop-10 overlap BM25 vs dense: mean {np.mean(overlap):.2f} of 10; top-1 identical: "
          f"{np.mean(top_b[:, 0] == top_d[:, 0]):.1%}")
    title = {d["id"]: d["title"] for d in docs}
    for label, i in (("dense wins most", int(np.argmax(diff))), ("BM25 wins most", int(np.argmin(diff)))):
        q = qids[i]
        print(f"\nExample ({label}): {q} {qs[i]['question']!r}, {n_rel[i]} relevant; "
              f"nDCG@10 BM25 {per_test['BM25']['ndcg@10'][i]:.3f} vs dense {per_test['Dense']['ndcg@10'][i]:.3f}")
        for name, top in (("BM25", top_b), ("dense", top_d)):
            for j in top[i][:3]:
                print(f"    {name:<5} grade {qrels[q].get(dids[j], 0)}  {title[dids[j]][:90]}")


def print_paired(label, a, b, metrics=("ndcg@10", "recall@10", "mrr@10", "p@1")):
    cells = []
    for m in metrics:
        mean, low, high = paired_bootstrap_ci(a[m], b[m])
        cells.append(f"{m} {mean:+.4f} [{low:+.4f}, {high:+.4f}]")
    print(f"  {label:<38} " + "  ".join(cells))


def baseline_analysis():
    """N4: zero-judge baselines; alpha tuned on dev only."""
    from extension.nfcorpus import ALPHAS, FIXED_ALPHA, baselines, evaluate, method_scores
    alpha_dev, rows = baselines()
    tuned = f"Fixed alpha={alpha_dev:.1f} (dev-tuned)"
    transferred = f"Fixed alpha={FIXED_ALPHA} (SQuAD)"
    for set_name in SETS:
        print(f"\n== {set_name} ==")
        for r in (r for r in rows if r["set"] == set_name):
            print(f"  {r['method']:<32} nDCG@10 {r['ndcg@10']:.4f}  R@10 {r['recall@10']:.4f}  "
                  f"MRR@10 {r['mrr@10']:.4f}  P@1 {r['p@1']:.4f}")
    per = evaluate("nfcorpus-test", method_scores("nfcorpus-test", {transferred: FIXED_ALPHA, tuned: alpha_dev}))
    print("\n== Paired differences on test (95% CI, 10,000 resamples) ==")
    for a, b in ((transferred, "Dense only"), (tuned, "Dense only"), (transferred, "BM25 only"),
                 ("RRF (k=60)", "Dense only"), ("RRF (k=60)", transferred), (tuned, transferred), (tuned, "RRF (k=60)")):
        print_paired(f"{a} - {b}", per[a], per[b])
    grid_test = evaluate("nfcorpus-test", method_scores("nfcorpus-test", {f"{a:.1f}": a for a in ALPHAS}))
    print("\n== Alpha sensitivity, nDCG@10 (test row descriptive only, never used for selection) ==")
    dev_grid = evaluate("nfcorpus-dev", method_scores("nfcorpus-dev", {f"{a:.1f}": a for a in ALPHAS}))
    print("  alpha " + " ".join(f"{a:>6.1f}" for a in ALPHAS))
    for label, g in (("dev ", dev_grid), ("test", grid_test)):
        print(f"  {label}  " + " ".join(f"{summarize(g[f'{a:.1f}'])['ndcg@10']:>6.4f}" for a in ALPHAS))
    write_rows([{"alpha": float(a), "dev_ndcg@10": summarize(dev_grid[f"{a:.1f}"])["ndcg@10"],
                 "test_ndcg@10_descriptive": summarize(grid_test[f"{a:.1f}"])["ndcg@10"]} for a in ALPHAS],
               RESULTS / "alpha_sensitivity.csv")
    bm25, _ = load_scores("nfcorpus-test")
    zero = (bm25 > 0).sum(axis=1) == 0
    print(f"\nTest queries where BM25 matches nothing (fusion = dense ranking): {zero.sum()}")
    for name in (transferred, tuned, "RRF (k=60)", "Dense only"):
        print(f"  {name:<32} nDCG@10 on those {per[name]['ndcg@10'][zero].mean():.4f}, on the rest {per[name]['ndcg@10'][~zero].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="NFCorpus analyses")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("retrieval")
    sub.add_parser("baselines")
    args = parser.parse_args()
    if args.command == "retrieval":
        retrieval()
    elif args.command == "baselines":
        baseline_analysis()


if __name__ == "__main__":
    main()
