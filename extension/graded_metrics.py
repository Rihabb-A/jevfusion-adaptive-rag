"""
Phase 4, Step N2: retrieval metrics for graded, multi-document relevance
(BEIR NFCorpus). The SQuAD metrics in baseline/evaluate.py assume exactly
one gold paragraph per question; these do not, and they leave that code
untouched.

Definitions, matching BEIR (which scores with pytrec_eval / trec_eval):

- Relevant: judged grade >= RELEVANT_GRADE (1), trec_eval's default.
  Unjudged documents and grade 0 count as not relevant.
- nDCG@k:  DCG@k / IDCG@k with DCG@k = sum over ranks i = 1..k of
  grade_i / log2(i + 1). The gain is the grade itself (linear), as in
  trec_eval's ndcg_cut, not 2^grade - 1. IDCG@k uses the query's judged
  grades sorted from high to low, truncated at k.
- Recall@k: relevant documents in the top k / all relevant documents of
  the query (so it cannot reach 1 when a query has more than k relevant).
- MRR@k:   1 / rank of the first relevant document within the top k, else
  0 (BEIR's MRR@k).
- P@1:     1 if the top-ranked document is relevant, else 0.

Scores are averaged over the queries that have judgments (all NFCorpus
dev and test queries do). Ranking is by score, best first, with ties
broken by corpus order (stable sort), as in the rest of this project.
trec_eval instead breaks ties by document id; with real-valued scores
ties are rare, and their effect is measured where they occur.

BEIR also drops a retrieved document whose id equals the query id; in
NFCorpus query ids (PLAIN-*) and document ids (MED-*) never coincide, so
that rule has no effect here.
"""

import math

import numpy as np

from extension.jev_judge import topk_indices

RELEVANT_GRADE = 1
K = 10
METRICS = ("ndcg@10", "recall@10", "mrr@10", "p@1")


def dcg(grades):
    """Discounted cumulative gain of grades in rank order (rank 1 first)."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(grades, 1) if g > 0)


def ndcg_at_k(ranked_grades, judged_grades, k=K):
    """`ranked_grades`: grades of the retrieved documents, rank order
    (0 for unjudged). `judged_grades`: every judged grade of the query."""
    ideal = dcg(sorted((g for g in judged_grades if g > 0), reverse=True)[:k])
    return dcg(ranked_grades[:k]) / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_grades, n_relevant, k=K):
    if n_relevant == 0:
        return 0.0
    return sum(g >= RELEVANT_GRADE for g in ranked_grades[:k]) / n_relevant


def mrr_at_k(ranked_grades, k=K):
    for rank, g in enumerate(ranked_grades[:k], 1):
        if g >= RELEVANT_GRADE:
            return 1.0 / rank
    return 0.0


def precision_at_1(ranked_grades):
    return float(bool(ranked_grades) and ranked_grades[0] >= RELEVANT_GRADE)


def per_query_metrics(scores, query_ids, doc_ids, qrels, k=K):
    """Per-query metric arrays (rows of `scores` follow `query_ids`, columns
    follow `doc_ids`) for paired comparisons; see `summarize` for means.
    Every query must have judgments in `qrels`."""
    missing = [q for q in query_ids if q not in qrels]
    if missing:
        raise ValueError(f"{len(missing)} queries have no judgments, e.g. {missing[:3]}")
    top = topk_indices(scores, k)
    out = {name: np.zeros(len(query_ids)) for name in METRICS}
    for row, query_id in enumerate(query_ids):
        judged = qrels[query_id]
        ranked = [judged.get(doc_ids[j], 0) for j in top[row]]
        n_relevant = sum(g >= RELEVANT_GRADE for g in judged.values())
        out["ndcg@10"][row] = ndcg_at_k(ranked, judged.values(), k)
        out["recall@10"][row] = recall_at_k(ranked, n_relevant, k)
        out["mrr@10"][row] = mrr_at_k(ranked, k)
        out["p@1"][row] = precision_at_1(ranked)
    return out


def summarize(per_query):
    return {name: float(np.mean(values)) for name, values in per_query.items()}
