# JevFusion on a second dataset: BEIR NFCorpus

**Question:** do our SQuAD findings (Jev J1 beats DAT's GPT-4o-mini judge;
top-3 judging hurts; adaptive judges beat fixed fusion) hold on a dataset
with **many relevant documents per query, graded labels and short topical
queries**?

**Short answer:** partly. Jev keeps its **cost and speed advantage** (2.6x
cheaper than DAT, faster on 94% of queries) with **no significant quality
difference (inconclusive at 323 queries)**; the **top-3
penalty disappears**, but top-3 brings no gain; and **no judge beats a
fixed fusion weight tuned on NFCorpus's dev split**, so the SQuAD result
"adaptive judges beat fixed fusion" does not generalize.

All numbers below are on the NFCorpus **test** split (323 queries),
zero-shot: the DAT prompt, Jev rubric, Jev's `top_level` mapping and the
J3 aggregation (option C) are exactly those chosen on SQuAD. Full step log
with every command, problem and decision: [`PROGRESS.md`](../PROGRESS.md) §12.

## 1. Dataset and how it differs from SQuAD

BEIR NFCorpus (official BEIR archive, MD5 verified): 3,633 medical
documents (title + abstract), 324 dev and 323 test queries,
`data/nfcorpus_manifest.json`.

| | SQuAD (our test set) | NFCorpus (test) |
|---|---|---|
| Queries | 2,996 | 323 |
| Corpus | 621 paragraphs (743 chars) | 3,633 documents (1,591 chars) |
| Relevant per query | exactly 1 | mean 38.2, median 16 |
| Labels | exact (answer-containing paragraph) | graded 1/2, from citation links, incomplete |
| Queries | questions, mean 10.1 words | topics or article titles, median 2 words |
| Primary metric | P@1 | nDCG@10 |

## 2. Setup

- **Retrievers:** the SQuAD ones, on title + text: BM25 (`rank_bm25`, k1
  1.5, b 0.75, no stemming) and `text-embedding-3-large` cosine; full
  corpus scored; ties by corpus order.
- **Fusion:** min-max normalization per query, `alpha * dense + (1 -
  alpha) * BM25` (DAT Eq. 3-7), unchanged.
- **Judges:** DAT (GPT-4o-mini, Appendix A prompt, pinned OpenAI endpoint,
  temperature 0); Jev J1 (`jev-1.13.0`, top-1 per retriever, DAT rubric as
  typed scores, `top_level` + DAT Eq. 6); Jev J3 (top-3 per retriever,
  option C). Documents sent whole (largest request ~8k tokens, far under
  the limits).
- **Baselines:** BM25, dense, RRF (k = 60), fixed alpha 0.6 transferred
  from SQuAD, and fixed alpha **0.8 tuned on NFCorpus dev** (rule fixed
  before running: best dev nDCG@10).
- **Metrics:** nDCG@10 (primary; linear gains as in BEIR/trec_eval),
  Recall@10, MRR@10, P@1; relevant = grade >= 1. Verified per query
  against `pytrec_eval`, BEIR's scorer (identical to 1e-16).
- **Statistics:** paired bootstrap over queries, 10,000 resamples, 95% CIs.

## 3. Results

| Method | nDCG@10 | Recall@10 | MRR@10 | P@1 | Judge cost / 1,000 queries |
|---|---:|---:|---:|---:|---:|
| BM25 only | 0.3067 | 0.1522 | 0.5097 | 0.4272 | $0 |
| Dense only | 0.4220 | 0.2108 | 0.6253 | 0.5418 | $0 |
| RRF (k=60) | 0.3699 | 0.1787 | 0.6017 | 0.5263 | $0 |
| Fixed alpha 0.6 (from SQuAD) | 0.4035 | 0.2039 | 0.6196 | 0.5325 | $0 |
| **Fixed alpha 0.8 (dev-tuned)** | **0.4252** | **0.2110** | **0.6424** | **0.5604** | $0 |
| DAT (GPT-4o-mini) | 0.4120 | 0.2042 | 0.6308 | 0.5480 | $0.158 |
| Jev J1 | 0.4173 | 0.2069 | 0.6372 | 0.5542 | **$0.062** |
| Jev J3 | 0.4158 | 0.2073 | 0.6290 | 0.5449 | $0.164 |

**Paired nDCG@10 differences** (95% CI; full table with all four metrics
in `results/nfcorpus/paired_comparisons.csv`):

| Comparison | nDCG@10 difference | Reading |
|---|---|---|
| Jev J1 - DAT | +0.0053 [-0.0028, +0.0135] | not significant |
| Jev J1 - Jev J3 | +0.0015 [-0.0050, +0.0082] | not significant |
| Jev J3 - DAT | +0.0038 [-0.0046, +0.0127] | not significant |
| Jev J1 - fixed 0.6 (from SQuAD) | +0.0138 [+0.0042, +0.0239] | J1 better |
| Jev J1 - fixed 0.8 (dev-tuned) | -0.0079 [-0.0159, +0.0000] | J1 lower (borderline) |
| DAT - fixed 0.8 (dev-tuned) | -0.0132 [-0.0214, -0.0056] | DAT worse |
| Jev J3 - fixed 0.8 (dev-tuned) | -0.0094 [-0.0170, -0.0020] | J3 worse |
| Jev J1 - RRF | +0.0475 [+0.0318, +0.0638] | J1 better |
| Jev J1 - dense only | -0.0047 [-0.0151, +0.0057] | not significant |
| DAT - dense only | -0.0100 [-0.0218, +0.0021] | not significant |
| Jev J3 - dense only | -0.0062 [-0.0172, +0.0048] | not significant |

**Statistical power:** with 323 queries, the smallest nDCG@10 difference
between judges that can be detected with 80% power (5% level) is **0.96
points** (J1 vs J3), **1.18** (J1 vs DAT) and **1.22** (J3 vs DAT);
column `ndcg@10_mde_80pct_power` of `results/nfcorpus/paired_comparisons.csv`
gives it for every comparison. Non-significant differences here are
**inconclusive, not proof of equality**.

**Headroom:** choosing the best grid alpha per query with knowledge of the
labels (an upper bound, not a method) would reach nDCG@10 0.4622, +3.7
points over the dev-tuned fixed weight. The judges capture none of it:
they land 0.8-1.3 points below that fixed weight.

## 4. What drives these results

**Retrieval regime (N3-N4).** Dense is 11.5 nDCG points ahead of BM25
on test (dev: 0.3915 vs 0.2664; SQuAD: the two were level). The two retrievers' top-1 documents agree on
only 26.9% of queries (SQuAD: 66.8%). A fixed blend barely beats dense
(+0.3, n.s.), and the SQuAD weight 0.6 is significantly *worse* than dense
(-1.85): BM25 must earn its weight query by query. RRF is the weakest
fusion, partly because on 25 queries BM25 matches no document at all.

**DAT's neutral default is miscalibrated here.** Eq. 6 sets alpha = 0.5
whenever the two top-1 judgments tie, which happens for 35% (DAT) to 47%
(J1) of queries. On SQuAD 0.5 was close to the best fixed weight (0.6);
on NFCorpus the best is 0.8, so every tie gives BM25 too much weight.

**Top-1 says little about a top-10.** nDCG@10 scores ranks 1-10. Even when
both retrievers return the same top document (87 queries), alpha still
changes nDCG@10, which could not happen for SQuAD's P@1.

**The judges rarely pick BM25 when it is right.** On the 60 queries where
BM25's own ranking is better, the judges lean toward BM25 only 17-27% of
the time. Where two judges disagree, the one leaning more toward dense
usually wins.

**Judge relevance and label relevance only partly agree** (Spearman
between judge score and official grade of the judged documents):

| Judge | Spearman | Mean score for grade 0 / 1 / 2 |
|---|---:|---|
| DAT (GPT-4o-mini), top-1 | 0.433 | 2.44 / 4.09 / 4.02 |
| Jev J1, top-1 (level) | 0.473 | 2.52 / 4.09 / 4.55 |
| Jev J3, all 6 documents (level) | 0.511 | 2.14 / 3.88 / 4.39 |

Jev's judgments track the labels better than DAT's and separate highly
from partially relevant documents (DAT does not), but all judges give
4-5 to about a third of grade-0 documents (possibly unlabeled relevant
ones; the labels are incomplete). Better judgments did not translate into
better rankings under DAT's alpha rule.

**Why top-3 no longer hurts but does not help (RQ2).** When a retriever's
#1 is relevant, its #2 and #3 are relevant 39-60% of the time on NFCorpus
(never on SQuAD), and Jev scores them higher (about 3 instead of 1-2), so
averaging dilutes the correct retriever much less. J3 leans toward the
better-ranking retriever most often of all judges (53.7%, wrong way
12.8%), yet its alphas cluster near 0.5-0.6 and do not follow where the
relevant documents are.

## 5. Latency

Same protocol as on SQuAD: the same 100 random test queries (seed 0) for
all three judges, one request at a time, judges rotating order per query,
2 warm-up calls per judge discarded, fresh uncached calls timed end to end
on the client (judge call only, network included; not retrieval or a full
RAG pipeline). HTTP attempts were counted to detect retries.
`results/nfcorpus/latency_summary.csv`, per call `latency_test.csv`.

| Judge | Calls OK | p50 | p95 | Max | Retries | Repeats its cached judgment |
|---|---:|---:|---:|---:|---:|---:|
| DAT (GPT-4o-mini) | 100/100 | 1,432 ms | 2,568 ms | 4,039 ms | 0 | 96% |
| **Jev J1** | 100/100 | **916 ms** | **1,658 ms** | 2,861 ms | 0 | 89% |
| Jev J3 | 100/100 | 968 ms | 1,589 ms | 2,352 ms | 0 | 78% (all 6 levels) |

Paired per query: **Jev J1 is faster than DAT on 94% of queries, by a
median 496 ms (95% CI [459, 528] ms)**; J3 by 449 ms [424, 492], on 95%.
No order effect (each judge's median is similar in every position). Jev
J3 reads 2.7x more tokens than J1 but is only ~50 ms slower. Absolute
times are lower than in our SQuAD measurement (taken on another day), so
compare judges within one run, not across datasets. Repeatability: Jev J1
repeats its own levels on 89% of queries (SQuAD: 95%), DAT on 96%.

## 6. Answers to the research questions

| | SQuAD | NFCorpus |
|---|---|---|
| **RQ1** Jev J1 - DAT | +1.64 P@1, significant; 2.3x cheaper; 527 ms faster | +0.53 nDCG@10, not significant (inconclusive); **2.6x cheaper; 496 ms faster** |
| **RQ2** J3 - J1 | -1.90 P@1, significant | -0.15 nDCG@10, not significant; 2.7x J1's cost |
| **RQ3** best judge - tuned fixed alpha | J1 +3.27 P@1 (significant) | J1 -0.79 nDCG@10 (borderline); DAT and J3 significantly lower |

- **RQ1:** Jev remains a much cheaper drop-in for DAT's judge with no
  measurable quality loss; whether it is *better* than GPT-4o-mini is
  dataset-dependent (yes on SQuAD; undetermined on NFCorpus).
- **RQ2:** SQuAD's top-3 penalty was a single-gold effect: with multiple
  relevant documents it disappears, but top-3 judging still does not
  improve fusion and is not worth its cost.
- **RQ3:** adaptive judges beat RRF and a *transferred* fixed weight, but
  not a fixed weight tuned on in-domain dev data. Their advantage over
  fixed fusion on SQuAD does not generalize to NFCorpus.

## 7. Limitations and threats to validity

- 323 test queries (detection limit 0.96-1.22 nDCG@10 points between judges).
- Incomplete, citation-based labels; unjudged documents count as not
  relevant, which may penalize judges that find relevant unlabeled
  documents.
- BM25 is deliberately unstemmed (the SQuAD setting): nDCG@10 0.3067 vs
  Anserini's 0.3218 for the same title + text index; adding Porter
  stemming to our code reproduces 0.3218 exactly.
- The dev-tuned baseline uses in-domain dev data; the judges were run
  zero-shot with SQuAD-chosen settings, by design.
- Untested hypotheses (would change the approved method): a dense-leaning
  neutral alpha, an nDCG-oriented aggregation of judgments, a stemmed BM25.

## 8. Reproduce

Prerequisites: the project environment and `.env` (OpenRouter and Jev
keys). From the repository root:

| # | Command | Produces | Time | Cost |
|---|---|---|---|---|
| 1 | `python -m extension.beir_data --dataset nfcorpus` | download, MD5 check, verification, manifest | ~1 min | free |
| 2 | `python -m extension.nfcorpus retrieve --set nfcorpus-dev` | BM25 + dense scores (embeds the corpus) | ~5 min | ~$0.16 |
| 3 | `python -m extension.nfcorpus retrieve --set nfcorpus-test` | BM25 + dense scores for test | ~1 min | <$0.01 |
| 4 | `python -m extension.nfcorpus_analysis retrieval` | retrieval quality, BM25 diagnostics | ~1 min | free |
| 5 | `python -m extension.nfcorpus_analysis baselines` | zero-judge baselines, dev-tuned alpha | ~1 min | free |
| 6 | `python -m extension.nfcorpus_judges check-sizes` | context-limit check | seconds | free |
| 7 | `python -m extension.nfcorpus_judges dat --set nfcorpus-test` | DAT judgments | ~2 min | ~$0.05 |
| 8 | `python -m extension.nfcorpus_judges jev --k 1 --set nfcorpus-test` | Jev J1 judgments | ~1 min | ~$0.02 |
| 9 | `python -m extension.nfcorpus_judges jev --k 3 --set nfcorpus-test` | Jev J3 judgments | ~2 min | ~$0.05 |
| 10 | `python -m extension.nfcorpus_analysis dat` / `j1` / `j3` | per-judge analyses | ~1 min each | free |
| 11 | `python -m extension.nfcorpus_analysis final` | `results/nfcorpus/final_comparison.csv`, `paired_comparisons.csv` | ~1 min | free |
| 12 | `python -m extension.latency --dataset nfcorpus --n 100` | fair latency | ~15 min | ~$0.06 |

Optional pilots: add `--set nfcorpus-dev --limit 20` to steps 7-9. Every
reply is cached by its exact request and re-runs cost nothing; spend is
logged in `results/budget_ledger.csv`; NFCorpus runs stop before NFCorpus
spending would pass $2. The BM25 stemming diagnostic in step 4 needs
`snowballstemmer` (optional, not a project dependency).

## 9. Cost

| Item | Cost |
|---|---:|
| Dense embeddings (corpus + queries) | $0.1602 |
| DAT (pilot + test) | $0.0544 |
| Jev J1 (pilot + test) | $0.0211 |
| Jev J3 (pilot + test) | $0.0564 |
| Latency measurement (306 calls) | $0.0389 |
| **NFCorpus total** | **$0.3310** |
| **Project total (SQuAD + NFCorpus)** | **$1.7144 of $20** |
