# Extension: JevFusion

**JevFusion replaces DAT's general-purpose LLM judge with Jev, a
dedicated decision model, and tests whether judging the top-3 paragraphs
per retriever beats judging only the top-1.**

Everything here builds on our DAT reproduction in [`baseline/`](../baseline/README.md)
(Hsu & Tzeng, 2025, [arXiv:2503.23013](https://arxiv.org/abs/2503.23013)) and
calls into it rather than copying it: same corpus, same retrieval scores,
same fusion, same metrics, same budget ledger. Only the judge and the way
its judgments become a fusion weight are new.

**Status:** full runs on our complete SQuAD test set (2,996 questions),
not a smoke test. Total API spend for the whole project: **$1.38 of the
$20 cap.**

## Headline result

| SQuAD test, 2,996 questions | P@1 | MRR@20 | Judge cost / 1,000 questions | Judge latency p50 |
|---|---:|---:|---:|---:|
| Fixed alpha = 0.6 (best zero-cost baseline) | 0.8571 | 0.9087 | $0 | none |
| DAT, GPT-4o-mini judge (our reproduction) | 0.8735 | 0.9167 | $0.110 | 2.92 s |
| **JevFusion J1 (Jev judges top-1)** | **0.8899** | **0.9268** | **$0.047** | **2.35 s** |
| JevFusion J3 (Jev judges top-3) | 0.8708 | 0.9198 | $0.118 | 2.54 s |

- **J1 beats DAT** by +1.64 points P@1 (95% CI [+1.00, +2.27]), at 2.3x
  lower judge cost and a median 0.53 s faster per question.
- **J3 does not beat J1:** -1.90 points P@1 (95% CI [-2.60, -1.23]). On
  SQuAD, judging more paragraphs hurts. We explain why below.

---

## 1. The limitation of DAT we address

DAT sets a per-question fusion weight `alpha` between BM25 and dense
retrieval by asking an LLM to score the top-1 paragraph from each
retriever on a 0-5 rubric (paper §4, Eq. 6). The paper itself names the
cost of this LLM call as DAT's main limitation (Appendix B): every query
needs a general-purpose LLM call, which adds latency and cost.

We ask two questions DAT leaves open:

- **Can a dedicated decision model do DAT's judging job** better, cheaper
  or faster than a general-purpose LLM?
- **Is DAT's top-1-only design a limitation?** The judge sees one
  paragraph per retriever. More evidence (top-3) might give better
  weights.

## 2. Research questions

- **RQ1:** Can Jev judging the top-1 paragraphs (J1) replace DAT's LLM
  judge without meaningfully lowering P@1 / MRR@20, at lower cost and
  latency?
- **RQ2:** Does judging the top-3 paragraphs per retriever (J3) give
  better fusion weights than top-1?
- **RQ3:** Are any gains worth the judge's cost compared with zero-call
  baselines (tuned fixed alpha, reciprocal rank fusion)?

| | Answer on SQuAD |
|---|---|
| RQ1 | **Yes, and better than DAT:** +1.64 points P@1 (significant), 2.3x cheaper, faster |
| RQ2 | **No:** J3 is 1.90 points below J1 (significant) and costs 2.5x more |
| RQ3 | **Yes for quality:** J1 beats fixed alpha 0.6 by +3.27 points and RRF by +7.94, for $0.047 per 1,000 questions |

## 3. What changed relative to DAT

| | DAT (baseline) | JevFusion J1 | JevFusion J3 |
|---|---|---|---|
| Judge model | GPT-4o-mini via OpenRouter (pinned to the OpenAI endpoint) | **Jev `jev-1.13.0`** (TypeSafe System One API) | Jev `jev-1.13.0` |
| Paragraphs judged | top-1 dense, top-1 BM25 | same two | **top-3 dense, top-3 BM25** |
| How the judge is asked | one free-text prompt (paper Appendix A), reply "d b" | one request: both paragraphs in `state`, one typed 6-level `score` question per paragraph with DAT's rubric as the levels | same, six `score` questions |
| Judge output | two integers 0-5 | per paragraph: decimal score, probability of each level 0-5, confidence | same |
| Score -> integer | already integer | **most likely level** (`top_level`, chosen on the tune set) | most likely level |
| Alpha | DAT Eq. 6 | **DAT Eq. 6, unchanged** | rank-weighted mean of levels (weights 1, 1/2, 1/3), rounded, then DAT Eq. 6 (option C, chosen on tune) |
| Fusion, normalization, metrics | DAT Eq. 3-7 | unchanged | unchanged |

**What J1 changes, precisely.** J1 is not only a model swap. Three things
differ from DAT at once: (1) the model, (2) the interface (typed score
questions answered with a probability per level, instead of a free-text
reply), and (3) the integer is Jev's most likely level instead of a
generated number. DAT's rubric levels, the paragraphs, the alpha rule and
everything downstream are identical. Our RQ1 result is therefore a
comparison of **"DAT's judge" vs "Jev with DAT's rubric"**, which is the
practical replacement question, not a pure model-vs-model test.

Files: [`jev_judge.py`](jev_judge.py) (requests, validation, cache),
[`jev_fusion.py`](jev_fusion.py) (judgments to alpha, evaluation, paired
tests), [`rrf.py`](rrf.py) (RRF baseline), [`latency.py`](latency.py)
(fair latency), [`final_table.py`](final_table.py) (final table).
Nothing in `baseline/` was copied or changed for the extension.

## 4. Experimental setup

**Data** (from the baseline): SQuAD v1.1, sampled with DAT's article rule
(§5.1). **Test:** 13 dev-split articles, 621 paragraphs, 2,996 questions.
**Tune:** a disjoint set of 13 train-split articles, 623 paragraphs, 2,865
questions. Each question has exactly one gold paragraph.

**Retrieval** (from the baseline): BM25 (`rank_bm25`, k1 1.5, b 0.75) and
`text-embedding-3-large` cosine, both scoring the full corpus; min-max
normalization; fused score `alpha * dense + (1 - alpha) * BM25`. All
methods use the identical cached score matrices, so every comparison is on
the same candidates.

**Jev requests.** One request per question. `state` = the question plus
the judged paragraphs labelled `dense_1`, `bm25_1` (J1) or `dense_1..3`,
`bm25_1..3` (J3); one `score` question per paragraph with DAT's six rubric
levels (0 off-track ... 5 direct hit). If both retrievers return the same
paragraph, it is sent under both labels, as DAT's prompt shows it twice.
Every reply is validated (served by `jev-1.13.0`, an answer for every
paragraph, score within 0-5, probabilities for exactly levels 0-5);
invalid replies are recorded, never defaulted.

**Pinned versions:** `JEV_MODEL=jev-1.13.0` (the `jev-latest` alias can
move to a new release without notice); GPT-4o-mini pinned to the OpenAI
endpoint, temperature 0, seed 42.

**Protocol, fixed before any test result was seen:**

1. **Every choice is made on the tune set only:** the J1 score mapping
   and the J3 alpha option, each by a rule declared before looking
   (highest tune P@1, then MRR@20, then the option closest to DAT). Test
   questions were used only for final numbers.
2. **RQ1 margin:** J1 "matches" DAT if the 95% CI of (J1 - DAT) P@1 lies
   above -1 point, "beats" DAT if it lies above 0. (DAT's own gain over
   fixed alpha is only 1.6 points, so losing more than 1 would give away
   most of it.)
3. **Paired statistics:** every comparison is on the same questions, with
   a 95% percentile bootstrap over questions (10,000 resamples).
4. **J3 was run on test as pre-registered**, even after tune suggested it
   would lose, so its negative result is a test-set result.

**Metrics** (DAT's): Precision@1, MRR@20, alpha-selection accuracy, and
the hybrid-sensitive subset (questions whose gold rank changes across the
alpha grid; 1,056 test questions). Definitions in `baseline/README.md`.

## 5. Results

All on the SQuAD test set (2,996 questions). Machine-readable:
`results/final_comparison.csv`, `results/jev_j1_test.csv`,
`results/jev_j3_test.csv`, `results/rrf_test.csv`,
`results/latency_summary.csv`.

### 5.1 Main comparison

| Method | P@1 | MRR@20 | Alpha acc. | Hybrid P@1 | Hybrid MRR@20 |
|---|---:|---:|---:|---:|---:|
| BM25 only | 0.7610 | 0.8289 | 0.7894 | 0.3494 | 0.5288 |
| Dense only | 0.7730 | 0.8489 | 0.8194 | 0.3835 | 0.5855 |
| RRF (k = 60) | 0.8104 | 0.8743 | n/a | 0.4896 | 0.6576 |
| Fixed alpha = 0.6 | 0.8571 | 0.9087 | 0.8975 | 0.6222 | 0.7554 |
| DAT (GPT-4o-mini) | 0.8735 | 0.9167 | 0.9169 | 0.6686 | 0.7779 |
| **JevFusion J1** | **0.8899** | **0.9268** | **0.9356** | **0.7150** | **0.8067** |
| JevFusion J3 | 0.8708 | 0.9198 | 0.9219 | 0.6610 | 0.7867 |

### 5.2 Paired comparisons (95% bootstrap CIs)

| Comparison | P@1 difference | MRR@20 difference | Rank 1 gained / lost | Verdict |
|---|---|---|---|---|
| **J1 - DAT** | **+0.0164 [+0.0100, +0.0227]** | +0.0101 [+0.0063, +0.0142] | 73 / 24 | **better** (whole CI above 0) |
| J1 - fixed 0.6 | +0.0327 [+0.0240, +0.0414] | +0.0181 [+0.0128, +0.0235] | 139 / 41 | better |
| J1 - RRF | +0.0794 [+0.0681, +0.0908] | +0.0526 [+0.0452, +0.0601] | 284 / 46 | better |
| **J3 - J1** | **-0.0190 [-0.0260, -0.0123]** | -0.0070 [-0.0110, -0.0030] | 28 / 85 | **worse** |
| J3 - DAT | -0.0027 [-0.0097, +0.0047] | +0.0031 [-0.0012, +0.0076] | 58 / 66 | no significant difference |
| J3 - fixed 0.6 | +0.0137 [+0.0060, +0.0217] | +0.0110 [+0.0064, +0.0157] | 91 / 50 | better |
| DAT - fixed 0.6 | +0.0164 [+0.0080, +0.0247] | +0.0079 [+0.0029, +0.0130] | 107 / 58 | better (reproduces the paper's claim) |

### 5.3 Where the differences come from (P@1 by question group)

Groups are defined by whether each retriever's top-1 paragraph is the gold.

| Group | Questions | Fixed 0.6 | DAT | J1 | J3 |
|---|---:|---:|---:|---:|---:|
| Both retrievers return the same paragraph | 2,001 | 0.9550 | 0.9550 | 0.9550 | 0.9550 |
| Only dense's top-1 is right | 405 | 0.7284 | 0.8716 | **0.9531** | 0.8469 |
| Only BM25's top-1 is right | 369 | 0.8618 | 0.9106 | **0.9539** | 0.8862 |
| Different paragraphs, both wrong | 221 | **0.1991** | 0.0769 | 0.0769 | 0.1267 |

On the 2,001 same-paragraph questions no alpha can change rank 1. All of
J1's gain over DAT (+49 rank-1 hits) comes from the 774 questions where
exactly one retriever is right.

### 5.4 Cost and latency

| Judge | Mean input tokens / question | Cost / question | Cost / 1,000 questions | Full test run | Latency p50 / p95 | Repeats its own judgment |
|---|---:|---:|---:|---:|---|---:|
| DAT (GPT-4o-mini) | ~724 | $0.000110 | $0.110 | $0.331 | 2.92 s / 4.57 s | 94% |
| **JevFusion J1** | 1,118 | **$0.000047** | **$0.047** | $0.141 | **2.35 s / 4.14 s** | 95% |
| JevFusion J3 | 2,805 | $0.000118 | $0.118 | $0.353 | 2.54 s / 4.33 s | 80% (all 6 levels) |

- **Cost:** Jev reads more tokens than DAT's prompt, but costs $0.042 per
  1M input tokens against GPT-4o-mini's $0.15 (and Jev output is free).
  Costs are the billed amounts of the full runs (OpenRouter-reported for
  DAT; input tokens x Jev's list price for Jev).
- **Latency** was measured separately, fairly: the same 100 random test
  questions for all three judges, one request at a time, judges rotating
  order per question, 2 warm-up calls discarded, fresh uncached calls
  timed client-side (including network). Paired per question, **J1 is
  faster than DAT on 81% of questions, by a median 527 ms (95% CI [413,
  633] ms)**; J3 by 346 ms [158, 545], on 70%. The gain is real but
  modest: network round trips are a large share of every call.
- **Repeatability:** re-asking the same request gives the same judgment on
  94% (DAT) and 95% (J1) of questions; GPT-4o-mini is not fully
  deterministic even at temperature 0.

### 5.5 Tune-set choices (made before touching test)

| J1 score mapping (tune, 2,865 questions) | P@1 | MRR@20 |
|---|---:|---:|
| **Most likely level, `top_level` (chosen)** | **0.8363** | 0.8852 |
| Decimal score rounded, `round` | 0.8356 | 0.8852 |

| J3 alpha option (tune) | P@1 | MRR@20 |
|---|---:|---:|
| A: mean of levels, rounded, DAT Eq. 6 | 0.8098 | 0.8719 |
| B: mean decimal / 5, then (J_D + 0.05) / (J_D + J_B + 0.10) (plan v3 formula) | 0.8126 | 0.8733 |
| **C: rank-weighted mean (1, 1/2, 1/3), rounded, DAT Eq. 6 (chosen)** | **0.8147** | 0.8757 |
| (reference: J1) | 0.8363 | 0.8852 |
| (reference: fixed alpha 0.6) | 0.7962 | 0.8611 |

The two J1 mappings are practically tied (2 questions apart); the rule
picked `top_level`. Every J3 option was already below J1 on tune.

## 6. Why J1 beats DAT

Both judges lean alpha toward the right retriever about equally often;
the difference is how rarely Jev leans the **wrong** way. On the 774 test
questions where exactly one retriever is right:

| Judge | Leans the right way | Neutral (alpha 0.5) | Leans the wrong way | Mean alpha when only dense is right | Mean alpha when only BM25 is right |
|---|---:|---:|---:|---:|---:|
| DAT (GPT-4o-mini) | 88.2% | 3.4% | **8.4%** | 0.835 | 0.210 |
| **J1** | 88.9% | 9.8% | **1.3%** | **0.958** | **0.102** |

- **Fewer harmful mistakes.** When Jev is unsure it scores both paragraphs
  alike, which leaves alpha at 0.5, a safe blend. GPT-4o-mini more often
  commits to the wrong side.
- **More decisive when it is sure.** Jev gives a clear level 5 ("direct
  hit") more often, so DAT's special case alpha = 0 or 1 fires for 802
  questions with J1 vs 463 with DAT. Its weights are both more extreme
  (0.958 / 0.102) and more often right.
- **Plausible mechanism (not separately tested):** Jev answers each score
  question with a full probability distribution over the six levels, and
  we take its most likely level; a generated pair of digits from a chat
  model has no such calibration step. The two judges agree on alpha for
  80.8% of questions; the remaining 19.2% decide the outcome.

## 7. Why J3 is worse than J1 on SQuAD

The diagnosis was done on the tune set and confirmed on test.

1. **SQuAD has exactly one gold paragraph per question.** If a retriever's
   #1 is the gold, its #2 and #3 are wrong by construction, and Jev
   correctly scores them low:

   | Tune set: Jev's mean level | #1 is gold: #1 / #2 / #3 | #1 is wrong: #1 / #2 / #3 |
   |---|---|---|
   | Dense | 4.94 / 2.21 / 1.64 | 3.25 / 3.45 / 2.62 |
   | BM25 | 4.95 / 1.30 / 0.91 | 1.79 / 2.17 / 1.50 |

2. **Averaging punishes the retriever that got it right.** Its clear 5 is
   pulled toward 2-3, while the other retriever, whose #1 is wrong, often
   holds the gold at #2 or #3 and is pulled up. On test, J3 leans the
   right way on only 62.0% of one-right questions (J1: 88.9%) and the
   wrong way on 9.8% (J1: 1.3%).
3. **J3 is biased toward dense retrieval** (mean alpha 0.574; only 0.453
   when BM25 alone is right). Dense's lower-ranked paragraphs are on-topic
   near misses, which DAT's rubric rewards as "good wrong results" (levels
   3-4). BM25's share keywords but drift off-topic (levels 1-2). Averaging
   over three paragraphs therefore favours dense systematically.
4. **It is the aggregation, not Jev.** Jev gives the same top-1 paragraph
   the same level in J1 and inside a J3 request for 92% of questions, with
   identical mean levels.
5. **Control (plan v3 asks to separate "more paragraphs" from "a different
   formula"):** option C applied to the top-1 paragraph alone is exactly
   J1 (unit-tested). So the whole J3 - J1 gap is caused by adding
   paragraphs #2 and #3, not by the formula.
6. **J3's one strength:** when both retrievers are wrong (221 test
   questions), J3 finds the gold at rank 1 for 12.7% vs 7.7% for J1 and
   DAT, because it pushes alpha to extremes less. Fixed 0.6 (19.9%) is
   still best there.

This supports DAT's own claim that the top-1 result is a sufficient
signal, **on a single-gold benchmark**. Whether J3 helps when several
documents per query are relevant is an open question (see §9).

## 8. Reproduce

Prerequisites: the baseline set up and run through step 6 of
[`baseline/README.md`](../baseline/README.md) (SQuAD samples, retrieval
scores, DAT judge replies). Add `JEV_API_KEY` to `.env`; `.env.example`
pins `JEV_MODEL=jev-1.13.0` and `JEV_API_BASE=https://api.typesafe.ai/v1`.

| # | Command | Produces | Time | Cost |
|---|---|---|---|---|
| 1 | `python -m extension.jev_judge --set tune --k 1` | Jev J1 judgments, tune | ~24 min | ~$0.13 |
| 2 | `python -m extension.jev_fusion --set tune --compare-mappings` | mapping choice (`top_level`) | seconds | free |
| 3 | `python -m extension.jev_judge --set test --k 1` | Jev J1 judgments, test | ~32 min | ~$0.14 |
| 4 | `python -m extension.jev_fusion --set test` | J1 vs DAT, fixed; `results/jev_j1_test.csv` | seconds | free |
| 5 | `python -m extension.jev_judge --set tune --k 3` | Jev J3 judgments, tune | ~20 min | ~$0.33 |
| 6 | `python -m extension.jev_fusion --set tune --compare-j3` | J3 option choice (C); `results/jev_j3_options_tune.csv` | seconds | free |
| 7 | `python -m extension.jev_judge --set test --k 3` | Jev J3 judgments, test | ~13 min | ~$0.35 |
| 8 | `python -m extension.jev_fusion --set test --j3` | J3 vs J1, DAT, fixed; `results/jev_j3_test.csv` | seconds | free |
| 9 | `python -m extension.rrf` | RRF baseline; `results/rrf_test.csv` | seconds | free |
| 10 | `python -m extension.latency --n 100` | fair latency; `results/latency_*.csv` | ~12 min | ~$0.03 |
| 11 | `python -m extension.final_table` | `results/final_comparison.csv` | seconds | free |

Optional pilot before step 1: `python -m extension.jev_judge --set tune --k 1 --limit 20` (under $0.01).

Every reply is cached by its exact request, so re-running costs nothing;
spend is logged to `results/budget_ledger.csv` and runs stop before
exceeding `BUDGET_CAP_USD`. After each run the cache is re-read from disk;
if it warns that replies are missing on disk (this happened twice, likely
OneDrive syncing the file mid-write), re-run the same command and only the
missing questions are sent. Tests: `pytest` (72 tests, including Jev
request building, reply validation and retries against fake responses).

Run times are with 8 parallel requests and vary with API load.

## 9. Limitations and what full validation would require

- **One dataset.** SQuAD is English and has a single gold paragraph per
  question. The J3 result in particular may not transfer to benchmarks
  where several documents per query are relevant (for example BEIR
  NFCorpus or FiQA), and J1's advantage needs checking on other domains
  and on the paper's second dataset, DRCD (Chinese). Full validation
  would mean repeating the baseline and steps 1-11 per dataset (about
  $1.30-1.70 of API cost per dataset with ~3,000-question tune and test
  sets: embeddings, DAT, J1 and J3 at current prices) plus adapting the
  metrics to graded, multi-document relevance labels.
- **J1 changes interface as well as model** (§3); we did not separate the
  two effects.
- **Prompts and rubric not optimized** for either judge; both use DAT's
  rubric. Results could change with prompt engineering.
- **The baseline is DAT-style:** our article sample differs from the
  authors' (no seed was published); see `baseline/README.md`.
- **Latency** was measured client-side from one location on 100 questions
  per judge, and includes network time.
- **Versions:** `jev-1.13.0` and the October 2026 GPT-4o-mini snapshot;
  later versions may behave differently.
- **Retrieval only:** we measure retrieval quality, not final RAG answer
  quality.
- **Weak spot of all adaptive judges:** when both retrievers are wrong,
  fixed alpha beats DAT, J1 and J3 (§5.3). A natural follow-up is to use
  J1's judgments but fall back toward a fixed blend when both top-1
  paragraphs look weak; not tested here.

## 10. Costs

| Item | Cost |
|---|---:|
| Jev J1: pilot, tune, test (incl. 6 re-judged) | $0.274 |
| Jev J3: tune (incl. 8 re-judged), test | $0.688 |
| Latency measurement (2 runs x 3 judges, incl. warm-up) | $0.056 |
| **Extension total** | **$1.018** |
| Baseline (embeddings, DAT judge) | $0.365 |
| **Project total** | **$1.383 of $20** |

Full ledger: `results/budget_ledger.csv`. Working log with every step,
decision and problem: [`PROGRESS.md`](../PROGRESS.md).
