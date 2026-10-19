# JevFusion: project progress log

A detailed, cumulative record of everything done in this project: what we
did, how, why, what it cost and what came out. Updated after every
completed step until the project ends.

**Last updated:** 20 October 2026, after Phase 4 Step N2 (graded
metrics for NFCorpus, on branch `extension/nfcorpus`). Earlier the same
day: the baseline PR (#1) was merged into `main` and `extension/jev` was
pushed. **The SQuAD work (Phases 1 and 2) is complete;** the extension
PR is for the team to open. **Phase 4 (BEIR NFCorpus) is in progress**,
one approved step at a time (§12).

**Every result table and analysis in one place:** [§11 Consolidated
results](#11-consolidated-results-every-table-and-analysis).

The formal deliverables are separate and are what graders read:
`baseline/README.md` (DAT reproduction), `extension/README.md` (JevFusion,
not written yet), `report/REPORT.md` and `slides/`. This file is the
working log behind them.

---

## 1. The project in one paragraph

Course: Data Science for Conversational AI, team project (topic:
grounding / RAG). The required shape is **one paper -> reproduce its
baseline and at least one published result -> build and evaluate our own
extension**, within **USD 20 total API spend**. Our paper is **DAT**
(Hsu & Tzeng, 2025, [arXiv:2503.23013](https://arxiv.org/abs/2503.23013)):
hybrid retrieval where an LLM judges the top-1 paragraph from BM25 and from
dense retrieval, and those two scores set a per-question fusion weight
alpha. Our extension, **JevFusion**, replaces DAT's GPT-4o-mini judge with
**Jev** (TypeSafe's decision model) and tests judging the **top-3**
paragraphs per retriever instead of top-1.

**Research questions** (plan v3, §2):
- **RQ1:** Can Jev top-1 (J1) replace the LLM judge without losing P@1 /
  MRR@20, at lower cost and latency?
- **RQ2:** Does Jev top-3 (J3) give better weights than Jev top-1?
- **RQ3:** Are any gains worth Jev's cost compared with zero-call
  baselines (tuned fixed alpha, RRF)?

---

## 2. Status at a glance

| Phase | Step | What | Status | Cost |
|---|---|---|---|---|
| 0 | 0.1-0.6 | Repo, approval, environment, API access, smoke tests | done | ~$0.0001 |
| 1 | 1.1 | SQuAD test and tune samples | done | free |
| 1 | 1.2 | BM25 and dense retrieval, cached | done | $0.034 |
| 1 | 1.3 | Free baselines and alpha grid search, checkpoint vs paper | done | free |
| 1 | 1.4.1-1.4.7 | DAT judge (GPT-4o-mini): code, host pin, pilot, full run, alphas | done | $0.331 |
| 1 | 1.5 | DAT evaluation vs paper, significance test | done | free |
| 1 | 1.6 | `baseline/README.md`, commits, push, PR | done; **PR #1 merged into `main`** (merge commit `f5da052`) | free |
| 2 | 2.1 | Branch, Issue 3 text, design decisions | done; **Issue 3 to open on GitHub** | free |
| 2 | 2.2 | Jev price and budget estimate; pin `jev-1.13.0` | done | free |
| 2 | 2.3 | Jev judge code (J1/J3) with offline tests | done | free |
| 2 | 2.4 | Jev J1 pilot (20 tune questions) | done | $0.001 |
| 2 | 2.5 | Jev J1 on full tune set; choose score mapping | done: **top_level** | $0.132 |
| 2 | 2.6 | Jev J1 on test set, compare with DAT | done: **J1 beats DAT** | $0.141 |
| 2 | 2.7.1 | J3 on tune set; choose alpha option | done: **C chosen; J3 below J1 on tune** | $0.335 |
| 2 | 2.7.2 | J3 (option C) on test set, compare with J1 and DAT | done: **J3 < J1 (significant), J3 ≈ DAT** | $0.353 |
| 2 | 2.8 | Control: top-1 with J3's formula | done by construction (option C with k=1 is exactly J1) | free |
| 2 | 2.9 | RRF baseline | done: **RRF weakest fusion; all judges beat it** | free |
| 2 | 2.10 | Fair latency, final comparison table | done: **Jev J1 fastest, cheapest, best** | $0.056 |
| 2 | 2.11 | `extension/README.md`, verification, PR and Issue 3 text | done; **pushed (approved 20 Oct); PR to open** | free |
| 4 | N0 | NFCorpus plan (no downloads, code or API calls) | done; plan and 4 decisions approved (§12) | free |
| 4 | N1 | Protect SQuAD data; download and verify NFCorpus | done: **verified, matches BEIR; SQuAD files unchanged** | free |
| 4 | N2 | Graded-relevance metrics + tests | done: **matches pytrec_eval (BEIR's scorer) exactly** | free |
| 4 | N3 | BM25 + dense retrieval on NFCorpus, sanity check vs BEIR | done: dense 0.4220 vs BM25 0.3067 nDCG@10; BM25 gap = stemming | $0.160 |
| 4 | N4 | Zero-judge baselines (BM25, dense, RRF, fixed 0.6, dev-tuned alpha) | done: dev alpha 0.8 = dense; 0.6 hurts; RRF weakest | free |
| 4 | N5 | Judge pipeline for named datasets + SQuAD regression check | done: 10/10 SQuAD result files byte-identical | free |
| 4 | N6 | DAT (GPT-4o-mini) on NFCorpus | done: **DAT below dev-tuned fixed alpha** (-1.32 nDCG) | $0.054 |
| 4 | N7 | Jev J1 on NFCorpus | done: **J1 = DAT (+0.53 n.s.) at 2.6x lower cost; below fixed 0.8** | $0.021 |
| 4 | N8 | Jev J3 on NFCorpus | done: **J3 = J1 (-0.15 n.s.); SQuAD penalty gone, no gain** | $0.056 |
| 4 | N9 | Full analysis | **in progress** | free |
| 4 | N10 | Fair latency | to do | ~$0.05 |
| 3 | 3.1-3.3 | Report, slides (max 8), defense | to do | free |

**Spend so far: $1.383 of $20** (see §8).

**Headline so far (test set, 2,996 questions):** Jev J1 P@1 **0.8899** vs
DAT (GPT-4o-mini) 0.8735 vs fixed alpha 0.6 0.8571. Jev J1 beats DAT by
+1.64 points P@1 (95% CI [+1.00, +2.27]) at less than half the cost per
question.

**RQ2 answer (confirmed on test):** judging the top-3 paragraphs per
retriever (J3) is *worse* than top-1 (J1): P@1 0.8708 vs 0.8899, -1.90
points (95% CI [-2.60, -1.23]), at 2.5x the cost. On single-gold SQuAD
the extra paragraphs dilute the top-1 signal (§5, Steps 2.7.1-2.7.2).

**RQ3 answer:** every judge beats both zero-call baselines. The
strongest zero-call baseline is fixed alpha 0.6 (P@1 0.8571); RRF is
weaker (0.8104). Jev J1 beats fixed 0.6 by +3.27 points and RRF by +7.94,
for $0.047 per 1,000 questions and a median 2.4 s judge call.

**RQ1 latency part:** measured fairly (same 100 questions, one request at
a time), Jev J1's judge call is faster than DAT's on 81% of questions,
by a median 527 ms (95% CI [413, 633] ms): p50 2.35 s vs 2.92 s.

**Overall:** Jev J1 is the best method on every axis we measured: highest
P@1 and MRR@20, 2.3x cheaper than DAT, and faster.

---

## 3. Phase 0: setup and approval

**Planning documents.** `JevFusion_Implementation_Plan_v2.md` was reviewed
against the DAT paper; it had targeted BEIR / nDCG@10 and a smoothed alpha
formula, none of which DAT uses. **Plan v3** (`JevFusion_Implementation_Plan_v3.md`)
fixed this: reproduce DAT on **SQuAD** with **P@1 / MRR@20**, DAT's exact
0-5 rubric and Eq. 6 alpha, `text-embedding-3-large`, and keep BEIR as an
optional later generalization test. Both plans and the paper PDF are kept
local and are not committed.

**Approval.** Idea pitch (Issue 1) approved by the instructor before
implementation.

**Repository.** Private repo created from the course template:
`github.com/Rihabb-A/jevfusion-adaptive-rag`. Branch protection on private
repos is not enforced on a free personal GitHub plan, so the PR-review rule
is followed by convention.

**Environment.** Python 3.13.3, virtual environment `.venv`, dependencies
in `requirements.txt` (`python-dotenv`, `pytest`, `numpy`, `rank-bm25`,
`requests`, `tqdm`; no provider SDKs, all APIs are called over HTTP).
`python scripts/verify_setup.py` passes (it was extended to accept
`OPENROUTER_API_KEY` and `JEV_API_KEY`).

**API providers** (keys only in the gitignored `.env`; `.env.example`
documents every variable):

| Use | Provider | Model | Price (checked Oct 2026) |
|---|---|---|---|
| Dense embeddings | OpenRouter | `openai/text-embedding-3-large` | $0.13 / 1M tokens |
| DAT judge | OpenRouter, pinned to the OpenAI endpoint | `openai/gpt-4o-mini` | $0.15 in / $0.60 out per 1M |
| JevFusion judge | TypeSafe API (`api.typesafe.ai/v1/systemone`) | `jev-1.13.0` (pinned) | $0.042 / 1M input, output free |

**Why not Jev through OpenRouter:** OpenRouter only lists
`typesafe/jev-router`, a router that uses Jev to pick *another* model to
answer. Its answers would not come from Jev and its price varies, so it
cannot serve as the judge. Jev is called directly.

**Smoke tests** (one call each, both passed; the test scripts were later
removed from the repo by decision, results recorded here):
- Jev: relevant passage -> `noul` 0.96, DAT-rubric score 4.88; irrelevant
  passage -> `noul` 0.01, score 0.00. About 510 input tokens, 0.3-0.6 s.
- OpenRouter: embeddings rank the relevant passage above the irrelevant
  (cosine 0.494 vs 0.037); GPT-4o-mini with DAT's Appendix A prompt
  returned `4 0` in the exact required format. Total cost $0.00008.

---

## 4. Phase 1: DAT reproduction (`baseline/`)

Full details, commands and the deviations list: [`baseline/README.md`](baseline/README.md).

### 1.1 SQuAD samples (`baseline/prepare_squad.py`)
- DAT's rule (§5.1): shuffle articles with a seed, take whole articles
  (all paragraphs and questions) until the next would exceed 3,000
  questions. Gold = the paragraph containing the answer.
- **test** set from SQuAD v1.1 **dev**: 13 articles, 621 paragraphs,
  **2,996 questions** (paper: 13 / 585 / 2,976). Dev was chosen because
  its 5.1 questions per paragraph matches the paper's sample.
- **tune** set from SQuAD v1.1 **train**: 13 articles, 623 paragraphs,
  2,865 questions. Used only for choosing settings, never for reported
  numbers.
- Seed 42; checksums and article lists in `data/squad_sample_manifest.json`.

### 1.2 Retrieval (`baseline/retrieve.py`)
- BM25: `rank_bm25.BM25Okapi` (k1 1.5, b 0.75), lowercase word tokens.
- Dense: `text-embedding-3-large` (3,072 dimensions), cosine similarity.
- Every paragraph is scored for every question (corpus is small, so no
  top-N cutoff). Score matrices cached in `data/cache/`; embeddings cached
  by text hash. Cost $0.034 for both sets.

### 1.3 Free baselines and checkpoint (`baseline/evaluate.py`, `baseline/dat_fusion.py`)
- Fusion: per-question min-max normalization, then
  `alpha * dense + (1 - alpha) * BM25` (paper Eq. 3-5).
- Metrics: P@1, MRR@20, alpha-selection accuracy, hybrid-sensitive subset
  (questions whose gold rank changes across alpha 0.0-1.0; this definition
  is the one consistent with the paper's tables).
- Alpha grid search on **tune** chose **alpha\* = 0.6**, the same value as
  the paper.
- Checkpoint on test: BM25 P@1 0.7610 (paper 0.7594), dense 0.7730
  (0.7396), fixed 0.6 0.8571 (0.8461). Close enough to proceed to the
  paid judge.

### 1.4 DAT judge (`baseline/dat_judge.py`)
- **1.4.1** Top-1 paragraph per retriever per question. Both retrievers
  return the same paragraph for 66.8% of questions; only one is right for
  774 questions (25.8%), which is where a judge can help.
- **1.4.2** Code: Appendix A prompt word for word (PDF spacing restored),
  strict parser (two integers 0-5 on the first line, else rejected),
  append-only cache keyed by the exact request. 15 tests.
- **1.4.3** Host pinned: OpenRouter -> OpenAI endpoint only, no fallback;
  temperature 0, seed 42, max_tokens 10 (paper states no settings).
- **1.4.4** Pilot, 20 questions: 20/20 parsed, all from OpenAI,
  $0.000108 per question. 7 of 8 disagreements pushed alpha toward the
  right retriever.
- **1.4.5** Full run, 2,996 questions, 8 parallel requests: 45 min,
  $0.329, **0 errors, 0 unparseable replies**, all served by OpenAI.
- **1.4.6** Retry step not needed (nothing failed).
- **1.4.7** Alphas: 67.9% are 0.5 (mostly same-paragraph questions). On
  the 774 disagreements, alpha moved toward the right retriever 88.2% of
  the time. Saved in `results/dat_alphas_test.csv`.

### 1.5 DAT results (test set, 2,996 questions)

| Method | P@1 ours | P@1 paper | MRR@20 ours | MRR@20 paper |
|---|---:|---:|---:|---:|
| BM25 only | 0.7610 | 0.7594 | 0.8289 | 0.8223 |
| Dense only | 0.7730 | 0.7396 | 0.8489 | 0.8119 |
| Fixed alpha 0.6 | 0.8571 | 0.8461 | 0.9087 | 0.8997 |
| **DAT (GPT-4o-mini)** | **0.8735** | **0.8676** | **0.9167** | **0.9093** |

- Hybrid-sensitive subset (ours n = 1,056, paper 1,111): DAT P@1 0.6686
  (paper 0.6805), fixed 0.6222 (paper 0.6229).
- DAT vs fixed 0.6, paired bootstrap on the same questions: **P@1
  +0.0164, 95% CI [+0.0080, +0.0247]**; MRR@20 +0.0079, CI [+0.0029,
  +0.0130]. Both significant. DAT gains rank 1 on 107 questions, loses it
  on 58.
- Our gain is a bit smaller than the paper's (+0.0215), consistent with
  our stronger dense retriever.
- Tables: `results/dat_reproduction.csv`.

### 1.6 Write-up and Git
- `baseline/README.md`: paper-vs-ours tables, 7 reproduce commands with
  time and cost, 9 documented deviations, what was not reproduced (GPT-4o
  ~$5.30, DeepSeek-R1-Distill-Qwen-14B needs a GPU, DRCD).
- Branch `baseline/setup` pushed (3 commits); no secrets in any commit
  (scanned). PR to `main` and Issue 2 are for the team to open and review.

**Not reproduced:** DRCD (the paper's Chinese dataset). Decided to move to
Phase 2 first and return to DRCD only if time allows.

---

## 5. Phase 2: JevFusion extension (`extension/`, branch `extension/jev`)

### 2.1 Design decisions
1. **One Jev request per question**, with all judged paragraphs in the
   `state` and one `score` question per paragraph. Jev bills the state
   once per request, so this is cheapest, and it shows Jev both
   paragraphs side by side as DAT's prompt does.
2. **DAT's 0-5 rubric** as a 6-level Jev `score` question, so J1 differs
   from DAT only in the judge.
3. **All Jev-specific choices are made on the tune set**; test questions
   are used only for final numbers.
4. If both retrievers return the same paragraph, it is still sent under
   both labels (as DAT shows it twice).

Issue 3 motivation text was drafted (in the chat); the issue itself is to
be opened on GitHub.

### 2.2 Price, budget and version
- Jev 1.13: **$0.042 per 1M input tokens, output free**, about 3.6x
  cheaper per token than GPT-4o-mini. Context 64k per request.
- `jev-latest` is an alias that can move to a new release without notice
  (1.14 came out 14 Oct 2026), so we **pinned `JEV_MODEL=jev-1.13.0`** in
  `.env` and `.env.example` (commit `f561ebb`).
- Phase 2 budget estimate: about $1 in total for Jev (J1 and J3 on tune
  and test).

### 2.3 Jev judge code (`extension/jev_judge.py`)
- `--k 1` (J1: dense_1, bm25_1) or `--k 3` (J3: top-3 from each).
  Measured request size: about 900 tokens (k=1) and 2,700 (k=3).
- Keeps per paragraph: decimal score, probability of each level 0-5, most
  likely level, confidence.
- Every reply validated: served by `jev-1.13.0`, an answer for every
  paragraph, score in 0-5, probabilities for exactly 6 levels. Invalid
  replies are recorded as such, never defaulted.
- Retries 429/5xx (honoring Retry-After), not 401/422. 8 parallel
  requests, budget stop, ledger entry.
- Reuses the baseline's retrieval scores, reply cache, question sampler
  and budget ledger; nothing in `baseline/` changed. 10 offline tests with
  fake API replies.

### 2.4 Pilot (20 tune questions)
- 20/20 valid, all `jev-1.13.0`, $0.00094.
- On the 10 questions where only one retriever was right, Jev scored the
  right one higher **10 of 10** times.
- **$0.000047 per question vs DAT's $0.000110** (about 2.3x cheaper; Jev
  uses more tokens, ~1,100, but each costs much less).
- Noted: Jev can be confident when wrong (scored a wrong shared paragraph
  4.98); to examine in error analysis.

### 2.5 J1 on the tune set and the score mapping (`extension/jev_fusion.py`)
- Full tune run: 2,865 questions, 23.8 min, $0.132, **2,865/2,865 valid**,
  0 errors, all `jev-1.13.0`. Mean 1,107 input tokens per question.
- DAT's Eq. 6 needs integers; Jev returns decimals (e.g. 4.36). Two
  mappings compared, with the selection rule fixed **before** looking:
  higher tune P@1, then higher MRR@20, then `top_level`.

| Tune set (2,865 questions) | P@1 | MRR@20 | Alpha acc. | Hybrid P@1 |
|---|---:|---:|---:|---:|
| Fixed alpha 0.6 | 0.7962 | 0.8611 | 0.8457 | 0.5726 |
| Jev J1, `top_level` (most likely level) | **0.8363** | 0.8852 | 0.8991 | 0.6601 |
| Jev J1, `round` (decimal rounded) | 0.8356 | 0.8852 | 0.8960 | 0.6586 |

- **Chosen: `top_level`.** The two are practically tied (2 questions
  apart in P@1); the rule picks `top_level`, which is also the option
  closest to DAT's integer output.
- On tune, J1 beats fixed 0.6 by +4.0 points P@1. There is no DAT
  (GPT-4o-mini) run on tune to compare with; the J1-vs-DAT comparison
  happens on test in Step 2.6.
- Test questions have not been used for any Jev decision.

### 2.6 J1 on the test set vs DAT (`python -m extension.jev_fusion --set test`)

**Agreed before running:** mapping `top_level` (from 2.5) and a
non-inferiority margin of **1 point P@1**: Jev "matches" DAT if the 95% CI
of (Jev - DAT) P@1 lies above -0.01, "beats" DAT if it lies above 0. Both
are fixed constants in `extension/jev_fusion.py`.

**Run:** 2,996 questions, 32.1 min with 8 parallel requests, $0.141,
2,996/2,996 valid, all `jev-1.13.0`, mean 1,118 input tokens per question.

**Incident:** after the run, 6 of the 2,996 replies were missing from the
cache file on disk (6 consecutive questions; the in-memory count had
hidden it). `load_judgments` refused to continue, which is how it was
caught. Likely cause: OneDrive syncing the file while it was being
appended to (unconfirmed). Fix: re-ran the same command, which judged
only the 6 missing questions ($0.0003); and every Jev run now re-reads its
cache from disk at the end and warns about anything missing. The DAT and
Jev tune caches were checked and are complete.

**Results** (`results/jev_j1_test.csv`, per question in `results/jev_j1_alphas_test.csv`):

| Test set (2,996 questions) | P@1 | MRR@20 | Alpha acc. | Hybrid P@1 | Cost / question |
|---|---:|---:|---:|---:|---:|
| BM25 only | 0.7610 | 0.8289 | 0.7894 | 0.3494 | $0 |
| Dense only | 0.7730 | 0.8489 | 0.8194 | 0.3835 | $0 |
| Fixed alpha 0.6 | 0.8571 | 0.9087 | 0.8975 | 0.6222 | $0 |
| DAT (GPT-4o-mini) | 0.8735 | 0.9167 | 0.9169 | 0.6686 | $0.000110 |
| **Jev J1** | **0.8899** | **0.9268** | **0.9356** | **0.7150** | **$0.000047** |

Paired comparisons (same questions, 10,000 bootstrap resamples):

| | P@1 diff, 95% CI | MRR@20 diff, 95% CI | Rank 1 gained / lost |
|---|---|---|---|
| Jev J1 - DAT | **+0.0164 [+0.0100, +0.0227]** | +0.0101 [+0.0063, +0.0142] | 73 / 24 |
| Jev J1 - fixed 0.6 | +0.0327 [+0.0240, +0.0414] | +0.0181 [+0.0128, +0.0235] | 139 / 41 |

**Verdict (RQ1): Jev J1 is better than DAT**, with the whole interval
above 0, so it clears the agreed margin with room to spare. Jev J1's gain
over fixed alpha is twice DAT's (+3.27 vs +1.64 points).

**Where the gain comes from** (P@1 by question group):

| Group | n | Fixed 0.6 | DAT | Jev J1 | Jev - DAT (hits) |
|---|---:|---:|---:|---:|---:|
| Same paragraph from both retrievers | 2,001 | 0.9550 | 0.9550 | 0.9550 | +0 |
| Only dense right | 405 | 0.7284 | 0.8716 | **0.9531** | +33 |
| Only BM25 right | 369 | 0.8618 | 0.9106 | **0.9539** | +16 |
| Different paragraphs, both wrong | 221 | **0.1991** | 0.0769 | 0.0769 | +0 |

- All of Jev's gain is on the 774 questions where exactly one retriever is
  right. Both judges lean the right way about equally often (Jev 88.9%,
  DAT 88.2%), but **Jev almost never leans the wrong way: 1.3% vs DAT's
  8.4%**. When unsure, Jev scores both sides alike (neutral 9.8% vs 3.4%).
- Jev is more decisive: it gives a clear level 5 more often, so DAT's
  "one side is a direct hit" rule (alpha 0 or 1) fires for 802 questions
  vs 463 with GPT-4o-mini.
- The two judges give the same alpha for 80.8% of questions.
- **Weak spot of both adaptive judges:** when both retrievers are wrong
  (221 questions), fixed 0.6 finds the gold paragraph at rank 1 for 19.9%
  of them but DAT and Jev only 7.7%. Pushing alpha to an extreme there
  removes the blend that sometimes surfaces the right paragraph. Worth
  discussing in the report; top-3 judgments (J3) might help here.

**Cost:** Jev J1 costs **$0.000047 per question vs DAT's $0.000110**
(2.3x cheaper). Latency is not compared yet: both runs used 8 parallel
requests under load; a fair measurement is Step 2.10.

### 2.7.1 J3 on the tune set and the choice of alpha formula

**Run** (`python -m extension.jev_judge --set tune --k 3`): one request per
question with the top-3 paragraphs from each retriever (6 paragraphs, 6
score questions). 2,865/2,865 valid, all `jev-1.13.0`, 19.7 min, mean
2,775 input tokens per question, **$0.000117 per question**, total $0.335.
The disk check caught 8 replies lost on disk (the OneDrive issue again);
they were re-judged for $0.001.

**Options compared** (agreed before looking; rule: highest tune P@1, then
MRR@20, then A). Each turns 3 judgments per retriever into one alpha:

| Option | Combine the 3 judgments per retriever | Alpha from |
|---|---|---|
| A `A_mean_dat` | mean of the most likely levels, rounded | DAT Eq. 6 |
| B `B_smooth` | mean decimal score / 5 | (J_D + 0.05) / (J_D + J_B + 0.10), plan v3 §4.1 |
| C `C_rank_weighted_dat` | weighted mean of levels, weights 1, 1/2, 1/3, rounded | DAT Eq. 6 |

**Results** (`python -m extension.jev_fusion --set tune --compare-j3`, saved in `results/jev_j3_options_tune.csv`):

| Tune set (2,865 questions) | P@1 | MRR@20 | Alpha acc. | Hybrid P@1 |
|---|---:|---:|---:|---:|
| Fixed alpha 0.6 | 0.7962 | 0.8611 | 0.8457 | 0.5726 |
| **Jev J1 (reference)** | **0.8363** | **0.8852** | **0.8991** | **0.6601** |
| Jev J3, A mean -> DAT rule | 0.8098 | 0.8719 | 0.8691 | 0.6023 |
| Jev J3, B smooth ratio | 0.8126 | 0.8733 | 0.8621 | 0.6084 |
| **Jev J3, C rank-weighted -> DAT rule** | **0.8147** | **0.8757** | 0.8789 | 0.6129 |

**Chosen: option C** (`CHOSEN_J3_OPTION` in `extension/jev_fusion.py`).
But every J3 option is **below J1** (C: -2.16 points P@1), though all
still beat fixed alpha.

**Why J3 loses to J1** (diagnosis on the tune set only):
1. **SQuAD has exactly one gold paragraph per question.** If a retriever's
   #1 is the gold, its #2 and #3 are wrong by construction, and Jev
   correctly scores them low: dense #1 right -> mean levels #1 4.94, #2
   2.21, #3 1.64; BM25 #1 right -> 4.95, 1.30, 0.91.
2. **Averaging therefore punishes the retriever that got it right.** Its
   clear 5 is pulled toward 2-3, while the other retriever (wrong #1) often
   holds the gold at #2 or #3 and is pulled up (dense #1 wrong -> #2 level
   3.45 is higher than its #1 at 3.25).
3. **Result: the signal is diluted and sometimes flipped.** On the 898 tune
   questions where exactly one retriever is right, alpha leans the right
   way 64.8% of the time with J3-C vs 87.9% with J1, and the wrong way
   12.8% vs 1.6%. J3 loses 79 rank-1 hits there.
4. **It is not Jev getting confused by more context:** Jev gives the same
   top-1 paragraph the same level in J1 and in J3 92% of the time, with
   identical mean levels (4.42 / 3.97). The loss comes from the
   aggregation, not from the judgments.
5. **Why C wins among the three:** it keeps the most weight on #1, so it
   dilutes the top-1 signal least. The ordering A < B < C follows how much
   each relies on #1.
6. **Where J3 does help:** when both retrievers are wrong (355 tune
   questions), J3 doubles J1's P@1 there (0.0873 vs 0.0394, +17 hits)
   because it pushes alpha to extremes less often. Fixed 0.6 is still best
   on that group (0.1465). The gain is far smaller than the loss elsewhere.

**Control (Step 2.8) is answered by construction for option C:** with only
the top-1 paragraph, option C reduces exactly to J1 (unit-tested). So the
J3-C vs J1 gap is entirely the effect of adding paragraphs #2 and #3 under
the same formula, not of a formula change.

**Not done, deliberately:** trying more weightings after seeing these
numbers (for example 1, 0.1, 0.1) would be fitting the tune set after the
fact. The pre-agreed choice (C) stands.

**Interpretation for RQ2 (provisional, tune only):** on SQuAD, top-1 is
the better signal; this supports DAT's own claim that the top-1 result is
"sufficient". J3 might behave differently on benchmarks with several
relevant documents per question (e.g. BEIR FiQA or NFCorpus), which is a
limitation to state and possible future work.

### 2.7.2 J3 (option C) on the test set (`python -m extension.jev_fusion --set test --j3`)

Run as pre-registered, even though tune suggested J3 would lose: a
confirmed negative result on test is stronger than a tune-only one.

**Run:** 2,996/2,996 valid, all `jev-1.13.0`, no replies lost on disk this
time, 12.8 min, mean 2,805 input tokens, $0.353 ($0.000118 per question).

**Results** (`results/jev_j3_test.csv`):

| Test set (2,996 questions) | P@1 | MRR@20 | Alpha acc. | Hybrid P@1 | Cost / question |
|---|---:|---:|---:|---:|---:|
| Fixed alpha 0.6 | 0.8571 | 0.9087 | 0.8975 | 0.6222 | $0 |
| DAT (GPT-4o-mini) | 0.8735 | 0.9167 | 0.9169 | 0.6686 | $0.000110 |
| **Jev J1** | **0.8899** | **0.9268** | **0.9356** | **0.7150** | **$0.000047** |
| Jev J3 (option C) | 0.8708 | 0.9198 | 0.9219 | 0.6610 | $0.000118 |

| Paired (same 2,996 questions) | P@1 diff, 95% CI | MRR@20 diff, 95% CI | Rank 1 gained / lost |
|---|---|---|---|
| J3 - J1 | **-0.0190 [-0.0260, -0.0123]** | -0.0070 [-0.0110, -0.0030] | 28 / 85 |
| J3 - DAT | -0.0027 [-0.0097, +0.0047] | +0.0031 [-0.0012, +0.0076] | 58 / 66 |
| J3 - fixed 0.6 | +0.0137 [+0.0060, +0.0217] | +0.0110 [+0.0064, +0.0157] | 91 / 50 |

**P@1 by question group (test):**

| Group | n | Fixed 0.6 | DAT | J1 | J3 |
|---|---:|---:|---:|---:|---:|
| Same paragraph | 2,001 | 0.9550 | 0.9550 | 0.9550 | 0.9550 |
| Only dense right | 405 | 0.7284 | 0.8716 | **0.9531** | 0.8469 |
| Only BM25 right | 369 | 0.8618 | 0.9106 | **0.9539** | 0.8862 |
| Different, both wrong | 221 | **0.1991** | 0.0769 | 0.0769 | 0.1267 |

**Conclusions:**
- **RQ2 answered: no.** J3 is significantly worse than J1 on test (-1.90
  points P@1, whole CI below 0), matching the tune-set diagnosis: with one
  gold paragraph per question, a correct retriever's #2 and #3 are wrong
  by construction and pull its score down.
- J3 is statistically indistinguishable from DAT (CI spans 0) and still
  significantly better than fixed 0.6, so it is not useless, just
  dominated by J1: worse quality at 2.5x J1's cost.
- J3's one strength holds on test: when both retrievers are wrong it
  finds the gold at rank 1 for 12.7% of questions vs 7.7% for J1 and DAT,
  because it pushes alpha to extremes less. Fixed 0.6 (19.9%) is still
  best there. This points to a possible future design: use top-1 to set
  alpha, and fall back toward a fixed blend when both top-1s look weak.
  (Not tested; noted as future work.)
- **Step 2.8 (control)** needs no extra run: option C with only the top-1
  paragraph is exactly J1 (unit-tested), so the whole J3-J1 gap is caused
  by adding paragraphs #2 and #3, not by the formula.

### 2.9 RRF baseline (`python -m extension.rrf`, free)

**What:** Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR
2009): each paragraph scores `1/(60 + dense rank) + 1/(60 + BM25 rank)`.
It uses rank positions only, so it needs no score normalization, no alpha
and no judge. k = 60 is the standard value, not tuned. Plan v3 includes it
to check that our min-max score fusion is not an artifact.

**Results** (`results/rrf_test.csv`):

| Method | Tune P@1 | Test P@1 | Test MRR@20 | Test hybrid P@1 |
|---|---:|---:|---:|---:|
| RRF (k=60) | 0.7414 | 0.8104 | 0.8743 | 0.4896 |
| Fixed alpha 0.6 | 0.7962 | 0.8571 | 0.9087 | 0.6222 |
| DAT (GPT-4o-mini) | - | 0.8735 | 0.9167 | 0.6686 |
| **Jev J1** | 0.8363 | **0.8899** | **0.9268** | **0.7150** |
| Jev J3 (option C) | 0.8147 | 0.8708 | 0.9198 | 0.6610 |

| Method minus RRF (test, paired) | P@1, 95% CI | MRR@20, 95% CI | Rank 1 gained / lost |
|---|---|---|---|
| Fixed 0.6 | +0.0467 [+0.0370, +0.0564] | +0.0345 [+0.0287, +0.0403] | 184 / 44 |
| DAT (GPT-4o-mini) | +0.0631 [+0.0517, +0.0744] | +0.0424 [+0.0353, +0.0495] | 252 / 63 |
| **Jev J1** | **+0.0794 [+0.0681, +0.0908]** | +0.0526 [+0.0452, +0.0601] | 284 / 46 |
| Jev J3 | +0.0604 [+0.0487, +0.0718] | +0.0455 [+0.0383, +0.0527] | 252 / 71 |

**Conclusions:**
- RRF is the **weakest fusion method**: better than either retriever alone
  (BM25 0.7610, dense 0.7730) but 4.7 points below fixed 0.6, and every
  alpha-based method beats it significantly.
- **Why:** RRF throws away how *confident* each retriever is. When the two
  disagree on #1, RRF just compares positions (1st vs 2nd...), while score
  fusion also sees that, say, BM25's #1 is far ahead of its #2. On SQuAD
  those score gaps carry real information.
- So min-max score fusion is **not** an artifact hurting the baselines; it
  is the stronger approach here, and fixed alpha 0.6 is the strongest
  zero-call baseline to beat.
- **RQ3 (quality):** all judges beat both zero-call baselines; Jev J1 by
  the largest margin (+3.27 points over fixed 0.6). Cost: Jev J1 adds
  about $0.047 per 1,000 questions for about 33 more correct top-1 results
  per 1,000. Latency is measured next (Step 2.10).

### 2.10 Fair latency and the final comparison table

**Why a separate measurement:** the judge runs used 8 parallel requests
under load, so their logged latencies say more about load than about the
judges. Plan v3 (§5.2) asks for p50 / p95 judge latency.

**Design** (`python -m extension.latency --n 100`):
- the **same 100 random test questions** (seed 0) for all three judges;
- **one request at a time**, never parallel;
- per question the judges take turns in a **rotating order**, so each
  judge goes first, second and third equally often and a slow network
  patch hits them alike;
- **2 warm-up calls per judge**, discarded (cold connections);
- **fresh, uncached calls**, same request format, model and settings as the
  real runs, timed end to end from the client (includes network);
- each fresh reply is compared with the cached reply for the same request
  (**repeatability**).

**A bug, caught and fixed:** the first run's repeatability column showed
0% for DAT, impossible at temperature 0. The fresh DAT scores were a
Python tuple `(3, 4)` and the cached ones a JSON list `[3, 4]`, which
never compare equal. Timings were unaffected, but the fresh replies had
not been saved, so the comparison was fixed, fresh replies are now saved,
and the **whole measurement was re-run** so every number comes from one
consistent run. A unit test documents the issue.

**Results** (second run; `results/latency_summary.csv`, per call in `results/latency_test.csv`):

| Judge | Calls OK | p50 | p95 | Mean | Max | Same as cached |
|---|---:|---:|---:|---:|---:|---:|
| DAT (GPT-4o-mini) | 100/100 | 2,923 ms | 4,569 ms | 3,172 ms | 10,665 ms | 94% |
| **Jev J1** | 100/100 | **2,355 ms** | **4,139 ms** | **2,576 ms** | 5,638 ms | 95% |
| Jev J3 | 100/100 | 2,543 ms | 4,331 ms | 2,783 ms | 6,416 ms | 80% (all 6 levels) |

Paired per question (same questions, 10,000 bootstrap resamples):

| | Median difference vs DAT | 95% CI | Faster than DAT on |
|---|---:|---|---:|
| **Jev J1** | **-527 ms** | [-633, -413] ms | **81%** of questions |
| Jev J3 | -346 ms | [-545, -158] ms | 70% of questions |

- **Order effects:** none visible; each judge's median is similar whether
  it went first, second or third (DAT 2.88-2.98 s, J1 2.28-2.42 s, J3
  2.51-2.59 s).
- **First run, for reference** (same design; only its repeatability
  column was wrong): p50 DAT 2,572 ms, J1 2,135 ms, J3 2,220 ms. J1 was
  again about 17-19% faster than DAT. That run had one extreme J3 call of
  451 s (likely a stalled connection), which is why medians and p95 are
  reported rather than means.
- **What latency means here:** these are client-side times including the
  network round trip, which is a large share of every call. Jev is faster
  but not dramatically so in this setting (about 0.5 s per question).
- **Repeatability:** DAT and Jev J1 repeat their own judgments on about
  95% of questions. GPT-4o-mini is not fully deterministic even at
  temperature 0 (one flip went from "0 0" to "5 5"; both still give
  alpha 0.5). J3's 80% counts a question as repeated only if all 6 levels
  match.
- **Cost of this step:** $0.056 for two runs of 306 calls each.

**Final comparison table** (`python -m extension.final_table`, recomputed
from the caches; `results/final_comparison.csv`): see §11.1.

### 2.11 `extension/README.md`, verification, PR and Issue 3 text

**What:** replaced the template placeholder in `extension/README.md` with
the graded write-up of the extension. **Why:** the course template
requires `extension/README.md` to state what changed vs the baseline, the
limitation addressed, a real comparison, and full-validation requirements.
**Cost:** free (no API calls).

**Contents of `extension/README.md`:** headline table; the DAT limitation
addressed (paper Appendix B); RQ1-RQ3 with answers; a side-by-side of
what changed vs DAT, stating plainly that J1 changes the model *and* the
interface (typed score questions) *and* the integer mapping, so RQ1
compares "DAT's judge" with "Jev with DAT's rubric"; setup and the
pre-declared protocol; result tables (main, paired CIs, question groups,
cost and latency, tune-set choices); why J1 beats DAT; why J3 is below J1;
11 reproduce commands with time and cost; limitations and what full
validation would require (other datasets, ~$1.30-1.70 each); cost
breakdown (extension $1.018, project $1.383).

**Verification** (a check script in the session scratchpad, not
committed):
1. Every number in the result CSVs (`final_comparison`, `jev_j1_test`,
   `jev_j3_test`, `jev_j3_options_tune`, `rrf_test`, `latency_summary`,
   ledger total), 76 values, was searched for in both documents.
   `PROGRESS.md`: 76/76 after fixes. `extension/README.md`: 69/76; the 7
   absent are baseline-vs-reference differences it deliberately omits or
   shows in the reverse direction (DAT - fixed, fixed - RRF).
2. Reverse check: every 4-decimal number in the README was looked up in
   the result files and `baseline/README.md`. 36 were values only ever
   printed (paired MRR CIs, question groups, tune `round` mapping); all 36
   were **recomputed from the caches and matched exactly**.

**Problems found and fixed:**
- **Rounding error:** Jev J1's p50 latency is 2,354.9 ms = **2.35 s**, but
  2.36 s had been written (rounding the already-rounded 2,355 ms) in
  `PROGRESS.md` (3 places) and the README. Corrected everywhere.
- **Incomplete table:** §11.1 listed the BM25 and dense differences vs
  fixed 0.6 as point values although `final_comparison.csv` has their CIs;
  CIs added, plus a hybrid MRR@20 column.
- **Cost estimate:** a draft said "$0.80 per dataset" for full validation;
  a new dataset needs embeddings, DAT, J1 and J3, about $1.30-1.70.
  Corrected before committing.

**Tests:** 72 passed. **Commit:** `981583d`.

**Pull request plan (not pushed; awaiting approval).** `extension/jev`
was branched from `baseline/setup`, whose PR is not merged yet. The clean
order: (1) open and merge the baseline PR `baseline/setup -> main`; (2)
push `extension/jev` and open `extension/jev -> main`, which then shows
only the extension commits. PR and Issue 3 texts were drafted in the
conversation for the team to post.

---

## 6. Decisions log

| When | Decision | Why |
|---|---|---|
| Phase 0 | Reproduce DAT on SQuAD with P@1/MRR@20, not BEIR/nDCG | Course requires the paper's own result; DAT used SQuAD |
| Phase 0 | OpenRouter for embeddings + DAT judge | One account; same models as the paper |
| Phase 0 | Jev called directly, not `typesafe/jev-router` | The router answers with other models |
| 1.1 | Test from SQuAD dev, tune from train | Disjoint sets; dev matches the paper's density |
| 1.2 | Score the full corpus, no top-N | Small corpus; removes "missing document" edge cases |
| 1.4.3 | Pin OpenAI endpoint, temperature 0, seed 42 | Reproducibility; never mix hosts |
| 1.4.5 | 8 parallel requests | 2+ hours sequential -> 45 min; results identical |
| 1.6 | Skip DRCD for now | Course needs one result; time goes to the extension |
| 2.1 | One Jev request per question, DAT rubric, same paragraphs | Cheapest and a clean judge-only comparison |
| 2.2 | Pin `jev-1.13.0` | Aliases move without notice |
| 2.5 | Mapping `top_level` | Pre-declared tune-set rule |
| 2.6 | Non-inferiority margin 1 point P@1, fixed before the test run | "Without meaningfully reducing quality" needs a number; DAT itself only gains 1.6 points |
| 2.6 | Re-read caches from disk after every Jev run | 6 replies were lost on disk (likely OneDrive sync) |
| 2.7.1 | J3 options A, B, C and the selection rule, fixed before the run | Few candidates, chosen on tune only, to avoid overfitting |
| 2.7.1 | J3 option C chosen; no extra weightings tried afterwards | Pre-declared rule; post-hoc tuning would bias the result |
| 2.7.2 | Run J3 on test despite the negative tune result | Pre-registered evaluation; a test-set negative result is stronger evidence |
| 2.8 | No separate control run | Option C with k=1 is exactly J1, so J1 already is the control |
| 2.9 | RRF with the standard k = 60, not tuned | Standard published setting; a zero-judge baseline should not get extra tuning |
| 2.10 | Latency: same 100 questions, sequential, rotating order, warm-up, fresh calls | Parallel-run timings measure load, not the judges |
| 2.10 | Re-run the whole latency measurement after fixing the repeatability bug | All reported numbers from one consistent run |
| 2.10 | Report p50 / p95, not means | One stalled call (451 s) in the first run shows means are fragile |
| 2.11 | State in the README that J1 changes model, interface and mapping together | Honest framing: RQ1 is "replace DAT's judge", not a pure model comparison |
| 2.11 | Merge the baseline PR before opening the extension PR | `extension/jev` contains the baseline commits; this keeps the extension PR diff clean |
| 2.11 | NFCorpus as a separate phase after the SQuAD review | Requested by the user; keeps the completed SQuAD results untouched |

---

## 7. Repository, branches and commits

| Commit | Branch | Content |
|---|---|---|
| `aa337f5` | baseline/setup | OpenRouter/Jev config, setup check, requirements |
| `5a594ae` | baseline/setup | Steps 1.1-1.3: sampling, retrieval, baselines |
| `df31448` | baseline/setup | Steps 1.4-1.6: DAT judge, evaluation, README (pushed) |
| `f561ebb` | extension/jev | Pin `jev-1.13.0` |
| `d15f905` | extension/jev | Step 2.3: Jev judge + tests |
| `4d0010d` | extension/jev | Step 2.5: J1 alphas, mapping choice + tests |
| `631680e` | extension/jev | This progress log (through 2.5) |
| `e1bf59a` | extension/jev | Step 2.6: J1 vs DAT on test, disk check after runs |
| `8a41832` | extension/jev | This progress log (through 2.6) |
| `13e2c25` | extension/jev | Step 2.7.1: J3 options, tune choice (C) |
| `f5875de` | extension/jev | This progress log (through 2.7.1) |
| `a83a773` | extension/jev | Step 2.7.2: J3 on test, group breakdown |
| `1eaca5f` | extension/jev | This progress log (through 2.8) |
| `0807900` | extension/jev | Step 2.9: RRF baseline + tests |
| `66fe58d` | extension/jev | This progress log (through 2.9) |
| `d8fa29f` | extension/jev | Step 2.10: latency, final table |
| `d6595d9` | extension/jev | This progress log (through 2.10, §11 consolidated results) |
| `981583d` | extension/jev | Step 2.11: `extension/README.md`; PROGRESS fixes |

| `f5da052` | main | Merge pull request #1 from `baseline/setup` (merge commit, not squash) |

`extension/jev` starts from `baseline/setup`. Since PR #1 was merged with
a merge commit, `origin/main...extension/jev` contains only the 16
extension commits (23 files), so the extension PR needs no rebase.
`extension/jev` was **pushed on 20 Oct 2026 after approval** (secret
scan of all branch commits clean; no `.env` or cache files among the 23
changed files). **72 unit tests** pass (`pytest`).

Local only (gitignored): `.env`, `.venv/`, `data/` (SQuAD, score matrices,
embedding cache, judge caches). Not committed by choice: the paper PDF and
the plan documents.

---

## 8. Budget

From `results/budget_ledger.csv` (every paid batch, billed cost):

| Component | Calls | Cost |
|---|---:|---:|
| Smoke tests (before the ledger existed) | 4 | ~$0.0001 |
| Dense embeddings (test + tune) | 57 | $0.0343 |
| DAT judge, GPT-4o-mini (pilot + test) | 2,996 | $0.3309 |
| Jev J1 (pilot + tune) | 2,865 | $0.1332 |
| Jev J1 test (incl. 6 re-judged) | 3,002 | $0.1410 |
| Jev J3 tune (incl. 8 re-judged) | 2,873 | $0.3349 |
| Jev J3 test | 2,996 | $0.3530 |
| Latency measurement, 2 runs x 3 judges (incl. warm-up) | 612 | $0.0561 |
| **Total** | | **$1.383 of $20** |

No further spend is planned for the core project. Optional extras: DAT
with GPT-4o (~$5.30), GPT-4o-mini top-3 control (~$0.70), DRCD (~$0.40).

---

## 9. Open items for the team

- Review the SQuAD work: `baseline/README.md`, `extension/README.md`, §11.
- ~~Open the PR `baseline/setup -> main`, review, merge~~ done (PR #1,
  20 Oct 2026). Make sure the baseline issue is closed with its
  acceptance criteria ticked (GitHub numbers issues and PRs together,
  so check the issue's real number before writing "Closes #N").
- ~~Approve pushing `extension/jev`~~ done. Open Issue 3 and the extension
  PR (`main...extension/jev`) with the drafted texts; teammate review;
  merge with a merge commit; close Issue 3.
- Phase 3: `report/REPORT.md`, slides (max 8), defense rehearsal.
- Decide later: DRCD; the optional GPT-4o-mini top-3 control run.

## 10. Next step

**For the team:** open Issue 3 and the PR
`https://github.com/Rihabb-A/jevfusion-adaptive-rag/compare/main...extension/jev`
(texts drafted in the conversation), review, merge. The PR shows only the
extension commits.

**After the SQuAD review:** the BEIR NFCorpus phase starts with a plan
only (no downloads, code or API calls): proposed small steps, what is
reused, what is new (notably multi-document, graded relevance labels,
which our SQuAD code assumes away: one `gold` paragraph per question),
cost and time per step, and the recommended first step.

---

## 11. Consolidated results: every table and analysis

All numbers are on the SQuAD **test** set (2,996 questions) unless marked
**tune**. Paired differences use the same questions for both methods,
with 95% percentile bootstrap intervals from 10,000 resamples.

### 11.1 Final comparison

`results/final_comparison.csv` (`python -m extension.final_table`).

| Method | P@1 | MRR@20 | Hybrid P@1 | Hybrid MRR@20 | P@1 vs DAT [95% CI] | P@1 vs fixed 0.6 [95% CI] | Judge $ / 1k questions | Judge p50 / p95 |
|---|---:|---:|---:|---:|---|---|---:|---|
| BM25 only | 0.7610 | 0.8289 | 0.3494 | 0.5288 | -0.1125 [-0.1248, -0.0998] | -0.0961 [-0.1085, -0.0841] | $0 | none |
| Dense only | 0.7730 | 0.8489 | 0.3835 | 0.5855 | -0.1005 [-0.1132, -0.0878] | -0.0841 [-0.0981, -0.0704] | $0 | none |
| RRF (k=60) | 0.8104 | 0.8743 | 0.4896 | 0.6576 | -0.0631 [-0.0744, -0.0517] | -0.0467 [-0.0564, -0.0370] | $0 | none |
| Fixed alpha 0.6 | 0.8571 | 0.9087 | 0.6222 | 0.7554 | -0.0164 [-0.0247, -0.0080] | reference | $0 | none |
| DAT (GPT-4o-mini) | 0.8735 | 0.9167 | 0.6686 | 0.7779 | reference | +0.0164 [+0.0080, +0.0247] | $0.110 | 2.92 s / 4.57 s |
| **Jev J1** | **0.8899** | **0.9268** | **0.7150** | **0.8067** | **+0.0164 [+0.0100, +0.0227]** | **+0.0327 [+0.0240, +0.0414]** | **$0.047** | **2.35 s / 4.14 s** |
| Jev J3 | 0.8708 | 0.9198 | 0.6610 | 0.7867 | -0.0027 [-0.0097, +0.0047] | +0.0137 [+0.0060, +0.0217] | $0.118 | 2.54 s / 4.33 s |

Every difference cell comes from the paired bootstrap (same questions,
10,000 resamples).

### 11.2 DAT reproduction vs the paper (SQuAD)

| Method | P@1 paper | P@1 ours | MRR@20 paper | MRR@20 ours | Alpha acc. paper | Alpha acc. ours |
|---|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.7594 | 0.7610 | 0.8223 | 0.8289 | 0.7981 | 0.7894 |
| Dense only | 0.7396 | 0.7730 | 0.8119 | 0.8489 | 0.7789 | 0.8194 |
| Fixed alpha 0.6 | 0.8461 | 0.8571 | 0.8997 | 0.9087 | 0.8975 | 0.8975 |
| DAT (GPT-4o-mini) | 0.8676 | 0.8735 | 0.9093 | 0.9167 | 0.9194 | 0.9169 |

Hybrid-sensitive subset (paper n = 1,111, ours 1,056):

| Method | P@1 paper | P@1 ours | MRR@20 paper | MRR@20 ours | Alpha acc. paper | Alpha acc. ours |
|---|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.3906 | 0.3494 | 0.5420 | 0.5288 | 0.4590 | 0.4025 |
| Dense only | 0.3375 | 0.3835 | 0.5143 | 0.5855 | 0.4077 | 0.4877 |
| Fixed alpha 0.6 | 0.6229 | 0.6222 | 0.7493 | 0.7554 | 0.7255 | 0.7093 |
| DAT (GPT-4o-mini) | 0.6805 | 0.6686 | 0.7750 | 0.7779 | 0.7840 | 0.7642 |

DAT's gain over fixed 0.6: paper +0.0215 P@1, ours +0.0164 [+0.0080,
+0.0247]; MRR@20 paper +0.0096, ours +0.0079 [+0.0029, +0.0130].

### 11.3 All paired comparisons

| Comparison | P@1 diff [95% CI] | MRR@20 diff [95% CI] | Rank 1 gained / lost | Reading |
|---|---|---|---|---|
| DAT - fixed 0.6 | +0.0164 [+0.0080, +0.0247] | +0.0079 [+0.0029, +0.0130] | 107 / 58 | DAT better (reproduces the paper's claim) |
| **J1 - DAT** | **+0.0164 [+0.0100, +0.0227]** | +0.0101 [+0.0063, +0.0142] | 73 / 24 | **J1 better** (pre-agreed margin: 1 point) |
| J1 - fixed 0.6 | +0.0327 [+0.0240, +0.0414] | +0.0181 [+0.0128, +0.0235] | 139 / 41 | J1 better |
| J3 - J1 | -0.0190 [-0.0260, -0.0123] | -0.0070 [-0.0110, -0.0030] | 28 / 85 | J3 worse |
| J3 - DAT | -0.0027 [-0.0097, +0.0047] | +0.0031 [-0.0012, +0.0076] | 58 / 66 | no significant difference |
| J3 - fixed 0.6 | +0.0137 [+0.0060, +0.0217] | +0.0110 [+0.0064, +0.0157] | 91 / 50 | J3 better |
| Fixed 0.6 - RRF | +0.0467 [+0.0370, +0.0564] | +0.0345 [+0.0287, +0.0403] | 184 / 44 | fixed better |
| DAT - RRF | +0.0631 [+0.0517, +0.0744] | +0.0424 [+0.0353, +0.0495] | 252 / 63 | DAT better |
| J1 - RRF | +0.0794 [+0.0681, +0.0908] | +0.0526 [+0.0452, +0.0601] | 284 / 46 | J1 better |
| J3 - RRF | +0.0604 [+0.0487, +0.0718] | +0.0455 [+0.0383, +0.0527] | 252 / 71 | J3 better |

### 11.4 P@1 by question group

Groups are defined by the two retrievers' top-1 paragraphs.

| Group | n | Fixed 0.6 | DAT | Jev J1 | Jev J3 |
|---|---:|---:|---:|---:|---:|
| Same paragraph from both | 2,001 | 0.9550 | 0.9550 | 0.9550 | 0.9550 |
| Only dense right | 405 | 0.7284 | 0.8716 | **0.9531** | 0.8469 |
| Only BM25 right | 369 | 0.8618 | 0.9106 | **0.9539** | 0.8862 |
| Different paragraphs, both wrong | 221 | **0.1991** | 0.0769 | 0.0769 | 0.1267 |

Jev J1 minus DAT, in rank-1 hits: +0, +33, +16, +0.

### 11.5 Judge behavior on the 774 questions where exactly one retriever is right

| Judge | Alpha leans the right way | Neutral (0.5) | Leans the wrong way | Mean alpha, only dense right | Mean alpha, only BM25 right |
|---|---:|---:|---:|---:|---:|
| DAT (GPT-4o-mini) | 88.2% | 3.4% | 8.4% | 0.835 | 0.210 |
| **Jev J1** | **88.9%** | 9.8% | **1.3%** | **0.958** | **0.102** |
| Jev J3 | 62.0% | 28.2% | 9.8% | 0.764 | 0.453 |

Right direction for Jev J1 only: 44 questions; for DAT only: 39.

### 11.6 Alpha distributions (share of 2,996 questions)

All three judges' alphas lie on DAT's 0.1 grid (J3 option C also ends in
DAT's rule, which rounds to one decimal).

| Judge | 0.0 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DAT (GPT-4o-mini) | 6.3% | 3.1% | 2.3% | 1.6% | 67.9% | 2.0% | 2.5% | 5.0% | 9.2% | 0.521 |
| Jev J1 | 10.5% | 0.1% | 0.0% | 0.6% | 70.9% | 0.5% | 0.0% | 1.1% | 16.3% | 0.532 |
| Jev J3 | 1.8% | 1.0% | 0.7% | 5.4% | 50.8% | 22.8% | 2.8% | 4.4% | 10.1% | 0.574 |

(0.1 and 0.9 never occur: DAT's formula cannot produce them; scores of 5
trigger the 0.0 / 1.0 special cases instead.)

| Pair of judges | Same alpha |
|---|---:|
| DAT vs Jev J1 | 80.8% |
| DAT vs Jev J3 | 50.1% |
| Jev J1 vs Jev J3 | 51.5% |

Extreme alphas (exactly 0 or 1): DAT 463, Jev J1 802, Jev J3 358.

Most common score pairs (dense, BM25): DAT 5/5 60.6%, 4/1 4.6%, 1/5
2.9%, 1/4 2.9%, 5/1 2.6%; Jev J1 levels 5/5 1,983, 5/0 165, 4/5 126, 3/5
122, 5/1 110.

### 11.7 Cost, tokens, latency and repeatability

| Judge | Calls per question | Mean input tokens | $ per question | $ per 1k questions | Full test run | Latency p50 / p95 (sequential) | Repeats own judgment |
|---|---:|---:|---:|---:|---:|---|---:|
| DAT (GPT-4o-mini) | 1 | ~724 | $0.000110 | $0.110 | $0.331 | 2.92 s / 4.57 s | 94% |
| **Jev J1** | 1 | 1,118 | **$0.000047** | **$0.047** | $0.141 | **2.35 s / 4.14 s** | 95% |
| Jev J3 | 1 | 2,805 | $0.000118 | $0.118 | $0.353 | 2.54 s / 4.33 s | 80% (6 of 6) |

Jev uses more tokens per question than DAT, but each Jev input token costs
$0.042 / 1M vs $0.15 / 1M (and Jev output is free). Paired latency vs DAT:
J1 -527 ms [-633, -413], faster on 81% of questions; J3 -346 ms [-545,
-158], faster on 70%.

### 11.8 Tune-set tables (choices made before touching test)

Alpha grid search for the fixed baseline (tune, 2,865 questions):

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | **0.6** | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P@1 | 0.6883 | 0.7065 | 0.7267 | 0.7511 | 0.7710 | 0.7878 | **0.7962** | 0.7955 | 0.7784 | 0.7382 | 0.6918 |
| MRR@20 | 0.7594 | 0.7783 | 0.7983 | 0.8200 | 0.8376 | 0.8518 | **0.8611** | 0.8621 | 0.8512 | 0.8227 | 0.7843 |

Jev J1 score mapping (Step 2.5):

| Mapping | P@1 | MRR@20 | Alpha acc. | Hybrid P@1 |
|---|---:|---:|---:|---:|
| **top_level (chosen)** | **0.8363** | 0.8852 | 0.8991 | 0.6601 |
| round | 0.8356 | 0.8852 | 0.8960 | 0.6586 |

Jev J3 alpha option (Step 2.7.1):

| Option | P@1 | MRR@20 | Alpha acc. | Hybrid P@1 |
|---|---:|---:|---:|---:|
| A mean -> DAT rule | 0.8098 | 0.8719 | 0.8691 | 0.6023 |
| B smooth ratio | 0.8126 | 0.8733 | 0.8621 | 0.6084 |
| **C rank-weighted -> DAT rule (chosen)** | **0.8147** | 0.8757 | 0.8789 | 0.6129 |
| (reference: Jev J1) | 0.8363 | 0.8852 | 0.8991 | 0.6601 |

J3 diagnosis (tune): Jev's mean level for each retriever's #1, #2 and #3
paragraph, split by whether that retriever's #1 is the gold:

| Retriever | #1 is gold: #1 / #2 / #3 | #1 is wrong: #1 / #2 / #3 |
|---|---|---|
| Dense | 4.94 / 2.21 / 1.64 | 3.25 / 3.45 / 2.62 |
| BM25 | 4.95 / 1.30 / 0.91 | 1.79 / 2.17 / 1.50 |

Where the gold paragraph sits in each retriever's top-3 (tune): dense
#1 1,982, #2 326, #3 146, not in top-3 411; BM25 #1 1,972, #2 247, #3 97,
not in top-3 549. Jev gives the same top-1 paragraph the same level in J1
and inside J3 for 92% of questions.

### 11.9 Key findings

1. **The DAT reproduction holds:** every metric is within about 1 point of
   the paper, and DAT's gain over fixed alpha is significant.
2. **Jev is a better judge than GPT-4o-mini for DAT:** +1.64 points P@1
   (significant), 2.3x cheaper, about 0.5 s faster per question.
3. **Why:** Jev almost never pushes alpha toward the wrong retriever (1.3%
   vs 8.4%). When unsure it stays neutral; when sure it commits fully
   (alpha 0 or 1 for 802 questions vs 463).
4. **Top-3 judging (J3) hurts on SQuAD:** -1.90 points vs J1. With one gold
   paragraph per question, a correct retriever's #2 and #3 are wrong by
   construction, so averaging dilutes its signal.
5. **J3 is biased toward dense retrieval** (mean alpha 0.574; only 0.453
   when BM25 alone is right). Dense's lower-ranked paragraphs are on-topic
   near misses, which DAT's rubric rewards as "good wrong results" (levels
   3-4); BM25's share keywords but drift off-topic (levels 1-2).
6. **Both adaptive judges are weak when both retrievers are wrong:** fixed
   0.6 gets 19.9% of those at rank 1, DAT and J1 only 7.7%. Committing to
   one retriever removes the blend that sometimes surfaces the answer. J3
   is in between (12.7%).
7. **Score-based fusion beats rank-based fusion here:** RRF is the weakest
   fusion method; score gaps carry information on SQuAD.

### 11.10 Limitations

- One dataset (SQuAD, English) with a single gold paragraph per question;
  results, especially the J3 finding, may differ on multi-relevance
  benchmarks (BEIR FiQA, NFCorpus) or on DRCD.
- Our article sample differs from the authors' (no seed published), so
  the reproduction is DAT-style, not an exact replication.
- One judge prompt / rubric per judge, not optimized; Jev received DAT's
  rubric as typed score levels, GPT-4o-mini as DAT's free-text prompt.
- Latency is client-side from one location and includes network time; 100
  questions per judge.
- Jev 1.13 was pinned; results may change with later versions.
- Retrieval quality, not final answer quality: no generator was evaluated.

### 11.11 Possible future work (see also §12)

- A hybrid of J1 and fixed alpha: use Jev's top-1 judgments, but fall back
  toward a fixed blend when both top-1 paragraphs look weak (targets
  finding 6).
- J3 on benchmarks with several relevant documents per question.
- DRCD and DAT with GPT-4o, to complete the paper's tables.

---

## 12. Phase 4 plan: BEIR NFCorpus (proposed 20 Oct 2026, awaiting approval)

**Goal.** Test on a second dataset whether (1) Jev J1 generalizes beyond
SQuAD, (2) J3 does better when several documents per query are relevant,
(3) adaptive fusion stays competitive with fixed alpha and RRF. Report
positive and negative findings alike; the completed SQuAD results stay
untouched.

**Step N0 (done): audit of SQuAD-only assumptions in our code.**

| Assumption | Where | Why it breaks on NFCorpus | Plan |
|---|---|---|---|
| One `gold` paragraph per query | `prepare_squad.load_set`, `evaluate.gold_ranks` (P@1, MRR@20, hybrid subset, alpha accuracy), `dat_judge.dat_alphas` (top-1 "correct"), `jev_fusion` groups/agreement, `rrf`, `final_table` (51 references) | NFCorpus has many relevant documents per query with graded labels | New graded-relevance evaluation; SQuAD code left as is |
| Set names `test` / `tune` only | CLI `choices`, `score_path`, judge cache paths, `latency.Judges("test")`, `final_table` | A run named `test` would **overwrite SQuAD caches** | All NFCorpus files under distinct names (`nfcorpus-dev`, `nfcorpus-test`) |
| SQuAD folder layout | `prepare_squad` (`data/squad/...`) | BEIR format differs (corpus, queries, qrels per split) | New BEIR loader writing `data/nfcorpus/...` |
| Paper reference numbers | `evaluate.PAPER_SQUAD` | No DAT paper numbers for NFCorpus | Reference: BEIR's published BM25 nDCG@10 as a sanity check |
| One shared embedding cache file | `data/cache/embeddings_*.npz` (rewritten on save) | Rewriting it risks the SQuAD embeddings (OneDrive lost writes twice) | Back up `data/cache/` before any NFCorpus run |

**Reused unchanged:** `retrieve.bm25_scores`, `dense_scores`,
`EmbeddingCache`, `tokenize`; `dat_fusion.minmax`, `fuse`, `dat_alpha`;
`evaluate.paired_bootstrap_ci`; `dat_judge` prompt, parser, settings,
`call_judge`, `JudgeCache`, `request_key`; `jev_judge` request building,
`call_jev`, `parse_jev`, validation; `jev_fusion.j1_alphas`,
`multi_alphas`, `CHOSEN_MAPPING` (`top_level`), `CHOSEN_J3_OPTION` (C);
`rrf.ranks_from_scores`, `rrf_scores`; `budget`.

**New:** BEIR download/verification; graded metrics (nDCG@10 with linear
gains as in BEIR/pytrec_eval, Recall@10, MRR@10, P@1 with "relevant" =
grade >= 1); judge loops that accept a dataset's candidates (small,
backward-compatible change with a regression check that SQuAD outputs are
byte-identical); NFCorpus analysis (J3 vs J1 by number of relevant
documents, relevance of #2/#3, judge behavior); tests.

| Step | What | Cost | Time |
|---|---|---|---|
| N1 | Branch `extension/nfcorpus`; back up `data/cache`; download NFCorpus from the official BEIR source with checksum; verify corpus, queries, dev/test splits, qrels (relevant docs per query, grade distribution, lengths); manifest | free | ~15 min |
| N2 | Graded metrics module + tests on hand-built examples | free | ~30 min |
| N3 | BM25 (free), then dense embeddings of the corpus and dev/test queries; sanity check vs BEIR's published BM25 | ~$0.15-0.25 | ~20 min |
| N4 | Zero-call baselines: BM25, dense, RRF, fixed alpha 0.6 (zero-shot from SQuAD); dev-tuned alpha reported separately | free | minutes |
| N5 | Make the judge loops dataset-aware; regression check: SQuAD results reproduced exactly, no API calls | free | ~30 min |
| N6 | DAT (GPT-4o-mini): 20-query pilot, then test (and dev) | ~$0.12 | ~10 min |
| N7 | Jev J1: test (and dev) | ~$0.04 | ~10 min |
| N8 | Jev J3: test (and dev) | ~$0.11 | ~10 min |
| N9 | Analysis: full comparison, paired CIs, J3 vs J1 under multiple relevance, judge behavior, NFCorpus vs SQuAD | free | ~1 h |
| N10 | Fair latency on 100 NFCorpus test queries (optional) | ~$0.04 | ~15 min |
| N11 | `extension/README.md` NFCorpus section, PROGRESS, commit; push only after approval | free | ~1 h |

**Estimated total:** about $0.55 (range $0.40-0.90); project total about
$1.9-2.3 of $20. Roughly 1-2 working days.

**Decisions to confirm before N1:** index `title + text` for both
retrievers (BEIR convention); report on the test split, use dev only for
tuned variants (reported separately); keep DAT's prompt and Jev's rubric
unchanged for the zero-shot run although NFCorpus queries are short
topics, not questions; primary metric nDCG@10. Caveat: about 323 test
queries, about 9x fewer than SQuAD, so confidence intervals will be wider.

**Approved 20 Oct 2026:** all four decisions as proposed (index title +
text; dev for tuning, test for final results; DAT prompt, Jev rubric,
`top_level` and option C unchanged for zero-shot; nDCG@10 primary, with
Recall@10, MRR@10 and P@1 secondary).

### N1 Protect SQuAD data; download and verify NFCorpus (free)

**Branch:** `extension/nfcorpus`, created from `extension/jev` (the
extension PR is not merged yet; `extension/jev` itself was not changed).

**Protection, before anything else:**
- Copied all 18 SQuAD data and cache files (`data/cache`, `data/squad`,
  the SQuAD manifest; 172 MB) to
  `C:\Users\RihabAtwi\jevfusion-backups\2026-10-20-before-nfcorpus\`,
  outside OneDrive, with SHA-256 checksums of originals and copy
  (identical) and the Git commit at backup time (`48179a5`).
- After N1, the checksums were recomputed: **all 18 files byte-identical**.
  No SQuAD code or result file changed (`git status`).
- NFCorpus files live only under `data/nfcorpus/` (gitignored, 12 MB);
  the committed record is `data/nfcorpus_manifest.json`.

**Source:** the official BEIR link and MD5 from the BEIR README
(`github.com/beir-cellar/beir`):
`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip`,
MD5 `a89dba18a62ef92f7d323ec890a0d38d` (**matched**), SHA-256
`efe5be03...671b0b` (full value and per-file hashes in the manifest).
Extraction refuses any archive entry that would land outside
`data/nfcorpus/raw` (unit-tested).

**Command:** `python -m extension.beir_data --dataset nfcorpus`
(new `extension/beir_data.py`: download, MD5 check, safe extraction, BEIR
loader, `doc_text` = title + text, verification, manifest; 5 new tests).

**Verification** (no problems found: no duplicate ids, no empty
documents, every judged query and document exists, no query shared
between splits):

| | BEIR README | Ours |
|---|---:|---:|
| Corpus documents | 3.6K | **3,633** (all with a title) |
| Test queries | 323 | **323** |
| Avg relevant docs per test query | 38.2 | **38.19** |

| Split | Queries with labels | Judgments | Grade 1 | Grade 2 | Relevant per query: mean / median / max | Queries with a grade-2 doc |
|---|---:|---:|---:|---:|---|---:|
| train | 2,590 | 110,575 | 110,575 | 0 | 42.69 / 18 / 1,363 | 0 |
| **dev** | **324** | 11,385 | 10,864 | 521 | 35.14 / 18.5 / 919 | (mean 1.61 per query) |
| **test** | **323** | 12,334 | 11,758 | 576 | 38.19 / 16 / 475 | 119 |

- 3,237 queries in total (2,590 + 324 + 323). Grades are only 1 and 2
  (no 0 rows); train has grade 1 only and is not used.
- Documents: mean 234 words / 1,591 characters (title + text), max
  10,090 characters; **about 2.1x longer** than SQuAD test paragraphs (743
  characters).
- Queries: mean 3.3 words, **median 2** (e.g. "myelopathy", "betel nuts",
  "Increasing Muscle Strength with Fenugreek").

**How NFCorpus differs from SQuAD (consequences for the next steps):**
1. **Many relevant documents per query** (median 16; 60% of test queries
   have more than 10), so Recall@10 cannot reach 1 for most queries. Only
   23 test queries are SQuAD-like with a single relevant document. This is
   the setting where J3 might help.
2. **Graded labels** (2 = highly, 1 = partially relevant; 95% are grade 1;
   only 119 test queries have any grade-2 document), so nDCG@10 needs
   graded gains.
3. **Loose, indirect relevance:** labels come from citation links, e.g.
   "Healthy Chocolate Milkshakes" counts a cherry-and-gout study as
   relevant (grade 1). DAT's rubric ("directly answers the question") may
   score such documents low even though the labels count them, a likely
   source of judge-label disagreement to examine in N9.
4. **Short topic queries, not questions:** the DAT prompt and Jev rubric
   are kept unchanged for zero-shot (approved), but they assume a
   question with an answer.
5. **Bigger corpus (3,633 vs 621) and longer documents:** full-corpus
   scoring is still cheap; judge requests will be roughly 2x longer.

**Problems encountered:** none.

**Cost:** $0. **Tests:** 77 passed (72 + 5 new).

**Next:** Step N2 (free): graded-relevance metrics (nDCG@10 with linear
gains as in BEIR / pytrec_eval, Recall@10, MRR@10, P@1 with relevant =
grade >= 1) in a new module, tested on hand-built examples, without
changing the SQuAD metric code.

**Git (N1):** commit `ce144bc` on `extension/nfcorpus` (not pushed).

### N2 Graded-relevance metrics (free)

**What and why:** the SQuAD metrics (`baseline/evaluate.py`) assume one
gold paragraph per question. NFCorpus has many relevant documents per
query with grades 1 and 2, so a separate module was written next to the
SQuAD code; **no existing file was modified**.

**Files created:** `extension/graded_metrics.py`,
`tests/test_graded_metrics.py`. **Modified:** only `PROGRESS.md`.

**Definitions** (documented in the module; matching BEIR, which scores
with `pytrec_eval` / `trec_eval`):

| Metric | Definition |
|---|---|
| Relevant | judged grade >= 1 (trec_eval default); unjudged and grade 0 = not relevant |
| **nDCG@10** (primary) | DCG@10 / IDCG@10, DCG = sum of grade_i / log2(i + 1) over ranks 1-10; **linear gain** (the grade itself, as trec_eval's `ndcg_cut`, not 2^grade - 1); IDCG from the query's judged grades sorted high to low, cut at 10 |
| Recall@10 | relevant documents in the top 10 / all relevant documents of the query |
| MRR@10 | 1 / rank of the first relevant document within the top 10, else 0 (BEIR's MRR@k) |
| P@1 | 1 if the top document is relevant, else 0 |

Scores are averaged over the queries with judgments (all 323 test and 324
dev queries). Per-query arrays are returned so the existing paired
bootstrap (`evaluate.paired_bootstrap_ci`) can be reused. Ranking reuses
`jev_judge.topk_indices` (stable sort, ties in corpus order).

**Two documented differences from trec_eval, both without effect here:**
trec_eval breaks score ties by document id, we by corpus order (ties are
rare with real-valued scores; to be measured in N3 where BM25 may give
zero scores to most documents); BEIR drops retrieved documents whose id
equals the query id, which never happens in NFCorpus (PLAIN-* vs MED-*).

**Tests** (10 new, hand-computed values): perfect ranking = 1; a worked
nDCG example (0, 2, 1 against judged {2, 1}: 1.761860 / 2.630930 =
0.669672); linear-not-exponential gain; ideal cut at 10 when a query has 12
relevant; grade 0 and unjudged ignored; recall denominator = all relevant
(10/12); MRR cut at 10; P@1; an end-to-end score matrix including a tie;
error when a query has no judgments.

**Independent cross-check against `pytrec_eval`** (BEIR's actual scorer,
v0.5.11, installed only in the session scratchpad, not a project
dependency), per query, on all 4 metrics:

| Scenario | Queries | nDCG@10 mean (ours = pytrec_eval) | Max per-query difference |
|---|---:|---:|---:|
| Real NFCorpus test labels, random scores | 323 | 0.007316 | 2.8e-17 |
| Real NFCorpus test labels, grade-boosted scores | 323 | 0.804172 | 2.2e-16 |
| Synthetic labels with grades 0 / 1 / 2 | 200 | 0.147192 | 1.7e-16 |

Recall@10, MRR@10 and P@1 matched exactly (difference 0) in all three.
(MRR@10 was compared with pytrec_eval's `recip_rank` on the top-10 run.)

**Problem encountered:** one test failed on the first run because the
hand-typed constant was 0.669670 instead of 0.669672 (the code was right:
1.761860 / 2.630930 = 0.669672). The constant was corrected and the
calculation is shown in the test.

**SQuAD protection:** all 18 SQuAD data and cache files are still
byte-identical to the N1 backup; `git diff` shows no change in
`baseline/`, `results/` or the existing extension modules.

**Git (N2):** commit `34c1f89` on `extension/nfcorpus` (not pushed).

**Tests:** **87 passed** (77 existing + 10 new). **Cost:** $0. **Project
total: $1.383 of $20** (unchanged).

**Next: Step N3 (needs approval; paid, about $0.15-0.25).** Score all
3,633 NFCorpus documents for the 324 dev and 323 test queries with the
same retrievers as SQuAD: BM25 (free) on title + text, then
`text-embedding-3-large` embeddings (about 0.9-1.0M tokens at $0.13 / 1M),
cached under new NFCorpus-only names so no SQuAD cache is touched (the
shared embedding cache file is backed up). Sanity check: our BM25
nDCG@10 against BEIR's published BM25 result for NFCorpus; report score
ties in the top 10.

**Approved 20 Oct 2026:** autonomous execution of N3-N10 within the
approved methodology, local commits only, at most $2 additional NFCorpus
spend; stop on any serious blocker.

### N3 BM25 and dense retrieval on NFCorpus (done)

**Objective:** score the full 3,633-document corpus for every dev and
test query with the same two retrievers as SQuAD, check BM25 against the
published BEIR baseline, and understand where each retriever is strong.

**State checked before starting:** branch `extension/nfcorpus`; ledger
$1.3834 with no NFCorpus spend; both API keys set; 18/18 SQuAD files
byte-identical to the backup; 87 tests passing.

**Implemented** (`extension/nfcorpus.py`, new): `load_split` (BEIR data
for `nfcorpus-dev` / `nfcorpus-test`; queries sorted by id; document text
= title + text); `retrieve` (reuses `baseline.retrieve.bm25_scores` and
`embed_batch` unchanged); `NFCorpusEmbeddingCache` (the baseline cache
class, but in its own file under `data/cache/nfcorpus/`, so the shared
SQuAD embedding file is never rewritten); a fingerprint of query and
document order saved with each score matrix and checked on load; an
NFCorpus budget guard (stops if NFCorpus spend would pass $2) on top of
the project's $20 check; embeddings saved every 10 batches and verified
on disk after the run. Analysis: `extension/nfcorpus_analysis.py`
(`retrieval` subcommand). Tests: `tests/test_nfcorpus.py` (6).

**Commands:**

```
python -m extension.nfcorpus retrieve --set nfcorpus-dev --bm25-only
python -m extension.nfcorpus retrieve --set nfcorpus-test --bm25-only
python -m extension.nfcorpus retrieve --set nfcorpus-dev     # embeds corpus + dev queries
python -m extension.nfcorpus retrieve --set nfcorpus-test    # embeds test queries
python -m extension.nfcorpus_analysis retrieval              # one diagnostic row needs snowballstemmer
```

**Splits:** dev (324 queries) and test (323); train unused.
**Configuration:** BM25 = `rank_bm25.BM25Okapi` (k1 1.5, b 0.75),
lowercase `\w+` tokens, no stemming or stop words (exactly the SQuAD
setting); dense = `text-embedding-3-large` via OpenRouter, cosine; both
on title + text; full corpus scored; ties broken by corpus order.

**Cost:** 3,916 unique texts embedded for dev (3,633 documents + dev
queries; duplicate texts embedded once), 1,230,376 tokens, **$0.1599**;
test queries 1,705 tokens, $0.0002. **N3 total $0.1602** (pre-run
estimate $0.28 including a 50% margin). NFCorpus total $0.1602;
**project total $1.5436 of $20**. Score matrices and the NFCorpus
embedding cache are backed up outside OneDrive
(`jevfusion-backups/nfcorpus-cache-after-n3`). SQuAD: 18/18 files
byte-identical after the embedding run.

**Results** (`results/nfcorpus/retrieval.csv`):

| Set | Retriever | nDCG@10 | Recall@10 | MRR@10 | P@1 | R@100 | R@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| dev | BM25 | 0.2664 | 0.1099 | 0.4646 | 0.3889 | 0.2057 | 0.3861 |
| dev | Dense | 0.3915 | 0.2031 | 0.5918 | 0.4969 | 0.4093 | 0.7350 |
| **test** | **BM25** | **0.3067** | 0.1522 | 0.5097 | 0.4272 | 0.2368 | 0.4232 |
| **test** | **Dense** | **0.4220** | 0.2108 | 0.6253 | 0.5418 | 0.3912 | 0.7296 |

**BM25 vs the published baseline.** Reference: Anserini (Pyserini) BM25
"flat" run, which also indexes title + text in one field, on BEIR v1.0.0
NFCorpus test: nDCG@10 **0.3218**, R@100 0.2457, R@1000 0.3704
(castorini/anserini, `docs/reproduce/from-document-collection/beir-v1.0.0-nfcorpus.flat.md`;
its multifield run gives 0.3254, the BEIR paper's 0.325). Ours is **1.5
points lower**. Diagnosis, adding Anserini's choices one at a time
(diagnostic only; `results/nfcorpus/bm25_diagnostics.csv`):

| BM25 variant (test) | nDCG@10 | R@100 (matched docs only) |
|---|---:|---:|
| Ours (SQuAD setting) | 0.3067 | 0.2331 |
| k1 0.9, b 0.4 (Anserini defaults) | 0.3052 | 0.2348 |
| + Lucene stop words | 0.3059 | 0.2361 |
| **+ Porter stemming** | **0.3218** | **0.2471** |
| + Lucene idf / tf formula | 0.3208 | 0.2462 |

**Stemming accounts for the whole gap:** with it our implementation
reaches Anserini's 0.3218 exactly (dev shows the same pattern: 0.2664 to
0.2890). The BM25 code is therefore correct; the experiment keeps the
unstemmed SQuAD setting as approved, so **our BM25 is a slightly weaker
but methodologically consistent baseline, not an exact BEIR
reproduction**. Our higher R@1000 (0.4232 vs 0.3704) is an artifact of
ranking the full corpus: queries with few matching terms get zero-score
documents in corpus order, which an inverted-index run would not return;
counting only matching documents gives R@1000 0.3678, in line with
Anserini.

**Zero-match queries:** 79 test queries match fewer than 10 documents
with BM25, and **25 match none** (e.g. "myelopathy": every score is 0, so
BM25's "top results" are just the first documents of the corpus). This
matters later: for these queries BM25's top-1, which DAT and Jev judge,
is arbitrary, and min-max normalization turns an all-zero BM25 row into
zeros, so fusion falls back to dense alone.

**Dense vs BM25 (test, paired, 10,000 resamples):** dense is much
stronger: nDCG@10 +0.1154 [+0.0918, +0.1395], Recall@10 +0.0586 [+0.0430,
+0.0762], MRR@10 +0.1155 [+0.0758, +0.1568], P@1 +0.1146 [+0.0619,
+0.1703]. Per query: dense better on 167, BM25 on 60, tie 96. The two
are complementary: their top-10 lists share only 3.0 documents on
average, and the top-1 is the same for 26.9% of queries (SQuAD: 66.8%).

**By query type (test, mean nDCG@10):**

| Queries | n | BM25 | Dense | Dense better on |
|---|---:|---:|---:|---:|
| 1 word | 103 | 0.3739 | 0.4564 | 34% |
| 2-3 words | 89 | 0.3028 | 0.3901 | 49% |
| 4+ words (mostly article titles) | 131 | 0.2564 | 0.4167 | 67% |
| BM25 matches < 10 documents | 79 | 0.2528 | 0.3598 | 34% |
| 1-5 relevant documents | 86 | 0.2850 | 0.3548 | 24% |
| 6-30 relevant | 132 | 0.2398 | 0.3343 | 55% |
| 31+ relevant | 105 | 0.4085 | 0.5873 | 70% |

Dense's advantage grows with longer, descriptive queries and with many
relevant documents. On single-word and few-relevant queries dense wins
per query only 24-34% of the time (the rest are ties or BM25 wins),
though dense still has the higher mean.

**Examples.** Dense wins most: "myelopathy" (1 relevant): BM25 matches
nothing and returns arbitrary statin papers; dense finds the vegan
spinal-cord degeneration case (grade 1) at rank 1. BM25 wins most:
"Panama" (2 relevant): BM25 matches the literal word in the Kuna cocoa
studies (both relevant at ranks 1-2); dense drifts to pancreatic-cancer
papers.

**What it means and how it compares with SQuAD:** on SQuAD BM25 and dense
were nearly equal (P@1 0.761 vs 0.773); here dense is 11.5 nDCG points
ahead. The best fusion weight is therefore likely to lean strongly toward
dense, and the transferred SQuAD alpha (0.6) may be suboptimal (tested in
N4). **Cannot conclude yet:** anything about the judges.

**Problems:** the first guessed URLs for the Anserini pages returned 404;
the repository tree was listed through GitHub's API and the correct pages
found. The stemming diagnostic needed `snowballstemmer`, installed in the
session scratchpad only (not a project dependency); without it that
diagnostic row is skipped. A long shell command with an embedded text
block failed to parse once; nothing had been written, and the record was
re-appended from a file.

**Tests:** 93 passed (87 + 6 new). **Limitations:** BM25 unstemmed by
design; dense quality not compared with a published
text-embedding-3-large NFCorpus number (none verified).

**Next:** N4, the zero-judge baselines (BM25, dense, RRF, fixed alpha 0.6
transferred from SQuAD, and an alpha tuned on dev only).

**Git (N3):** commit `73d7808` on `extension/nfcorpus` (not pushed).

### N4 Zero-judge baselines on NFCorpus (done, free)

**Objective:** measure every method that needs no judge, so the judges in
N6-N8 have strong, honest baselines: BM25, dense, RRF (k = 60), fixed
alpha = 0.6 transferred unchanged from SQuAD, and a fixed alpha tuned on
**dev only**.

**Implemented:** `nfcorpus.method_scores` (reuses `dat_fusion.minmax` and
`fuse` and `rrf.rrf_scores` unchanged), `nfcorpus.evaluate` (N2 metrics),
`nfcorpus.choose_alpha` / `tune_alpha` (grid 0.0-1.0, step 0.1; rule fixed
before running: highest **dev** nDCG@10, then dev Recall@10, then the
alpha closest to 0.6; refuses any set other than `nfcorpus-dev`),
`nfcorpus.baselines`, and `nfcorpus_analysis baselines` (paired tests,
sensitivity, zero-match queries). Tests: 2 new (selection rule, dev-only
guard). Files: `extension/nfcorpus.py`, `extension/nfcorpus_analysis.py`,
`tests/test_nfcorpus.py`; results `results/nfcorpus/baselines.csv`,
`dev_alpha_grid.csv`, `alpha_sensitivity.csv`.

**Command:** `python -m extension.nfcorpus_analysis baselines`.
**Splits:** dev for the alpha search; test for every reported number.

**Dev alpha grid** (selection data):

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | **0.8** | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev nDCG@10 | 0.2664 | 0.3192 | 0.3286 | 0.3410 | 0.3516 | 0.3611 | 0.3732 | 0.3843 | **0.3941** | 0.3940 | 0.3915 |
| test nDCG@10 (descriptive only) | 0.3067 | 0.3478 | 0.3589 | 0.3679 | 0.3806 | 0.3922 | 0.4035 | 0.4139 | 0.4252 | 0.4296 | 0.4220 |

**Dev-tuned alpha = 0.8** (0.9 is 0.0001 lower on dev). The test row was
computed afterwards for sensitivity only; it peaks at 0.9 (0.4296), close
to the dev choice, so the dev selection transfers reasonably.

**Results:**

| Method | dev nDCG@10 | test nDCG@10 | test Recall@10 | test MRR@10 | test P@1 |
|---|---:|---:|---:|---:|---:|
| BM25 only | 0.2664 | 0.3067 | 0.1522 | 0.5097 | 0.4272 |
| **Dense only** | 0.3915 | **0.4220** | 0.2108 | 0.6253 | 0.5418 |
| RRF (k=60) | 0.3230 | 0.3699 | 0.1787 | 0.6017 | 0.5263 |
| Fixed alpha 0.6 (transferred from SQuAD) | 0.3732 | 0.4035 | 0.2039 | 0.6196 | 0.5325 |
| **Fixed alpha 0.8 (dev-tuned)** | 0.3941 | **0.4252** | 0.2110 | 0.6424 | 0.5604 |

**Paired differences on test** (95% CI, 10,000 resamples):

| Comparison | nDCG@10 | Recall@10 | MRR@10 | P@1 |
|---|---|---|---|---|
| Fixed 0.6 - dense | **-0.0185 [-0.0317, -0.0054]** | -0.0069 [-0.0137, -0.0003] | -0.0057 [-0.0334, +0.0209] | -0.0093 [-0.0495, +0.0310] |
| Fixed 0.8 (dev) - dense | +0.0032 [-0.0057, +0.0125] | +0.0002 [-0.0048, +0.0052] | +0.0171 [-0.0022, +0.0369] | +0.0186 [-0.0124, +0.0495] |
| Fixed 0.6 - BM25 | +0.0968 [+0.0794, +0.1152] | +0.0517 [+0.0374, +0.0680] | +0.1099 [+0.0801, +0.1404] | +0.1053 [+0.0650, +0.1486] |
| RRF - dense | **-0.0522 [-0.0708, -0.0343]** | -0.0321 [-0.0447, -0.0210] | -0.0235 [-0.0574, +0.0095] | -0.0155 [-0.0619, +0.0310] |
| RRF - fixed 0.6 | -0.0336 [-0.0470, -0.0214] | -0.0252 [-0.0363, -0.0157] | -0.0179 [-0.0413, +0.0038] | -0.0062 [-0.0372, +0.0248] |
| Fixed 0.8 (dev) - fixed 0.6 | **+0.0217 [+0.0138, +0.0301]** | +0.0071 [+0.0024, +0.0119] | +0.0228 [+0.0063, +0.0404] | +0.0279 [+0.0031, +0.0557] |
| Fixed 0.8 (dev) - RRF | +0.0553 [+0.0408, +0.0708] | +0.0323 [+0.0217, +0.0444] | +0.0407 [+0.0145, +0.0678] | +0.0341 [-0.0031, +0.0743] |

**What happened.**
1. **Fusion barely helps on NFCorpus.** The best fusion (dev-tuned 0.8) is
   only +0.32 nDCG points over dense alone, and that is not significant
   (CI spans 0). On SQuAD, fusion with alpha 0.6 beat the best single
   retriever by 8.4 points P@1.
2. **The transferred SQuAD alpha (0.6) hurts:** it is significantly
   *below* dense alone (-1.85 points). It gives BM25, which is 11.5 points
   weaker here, too much weight.
3. **Dev tuning works:** alpha 0.8 is significantly better than 0.6 (+2.17
   points), and the dev and test grids agree on the shape (rising to a
   flat peak at 0.8-0.9).
4. **RRF is the weakest fusion method again**, 5.2 points below dense.
   Part of this is specific to our full-corpus ranking: on the 25 test
   queries where BM25 matches nothing, RRF still adds BM25's arbitrary
   corpus-order "ranks" and falls to nDCG@10 0.0698, while score fusion
   ignores the all-zero BM25 row (min-max gives zeros) and keeps dense's
   0.2495. On the other 298 queries RRF is still below fixed 0.6 (0.3950
   vs 0.4164), so the weakness is not only that artifact.

**Why (plausible, supported by N3):** dense is much stronger than BM25 on
NFCorpus's short topical queries and loose, citation-based relevance, so
any weight on BM25 must be earned query by query. A fixed blend cannot do
that; it either matches dense (alpha near 1) or dilutes it.

**What it means for the research (RQ3).** On NFCorpus the bar for the
judges is "dense alone / alpha 0.8" (about 0.42-0.43 nDCG@10), not the
weak fixed 0.6. A judge earns its cost only if it moves weight toward BM25
on exactly the queries where BM25 helps (like "Panama" in N3) and keeps it
on dense elsewhere. Because DAT's rule returns alpha = 0.5 whenever both
top-1 documents get equal scores, and 0.5 is worse than dense here, the
judges start from a handicap on NFCorpus that they did not have on SQuAD
(where the best fixed alpha was 0.6, close to 0.5).

**How it compares with SQuAD:** SQuAD's tuned alpha (0.6) and NFCorpus's
(0.8) differ, so a single fixed weight does not transfer across datasets,
which is DAT's own motivation for per-query weights.

**Cannot conclude:** that fusion never helps on NFCorpus (the CI for 0.8
vs dense allows up to +1.25 points), or anything about the judges yet.

**Problems:** none. **Tests:** 95 passed (93 + 2). **Cost:** $0.
**Project total:** $1.5436 of $20 (NFCorpus $0.1602).

**Next:** N5, adapting the DAT and Jev judge code to named datasets, with
a proof that every SQuAD result is reproduced unchanged before any paid
judge run.

**Git (N4):** commit `5ba13c4` on `extension/nfcorpus` (not pushed).

### N5 Judge pipeline for named datasets + SQuAD regression check (done, free)

**Objective:** run the exact SQuAD judges (DAT prompt and settings, Jev
request format and validation, alpha rules) on NFCorpus candidates,
without any risk to the SQuAD results.

**Shared code changed (backward compatible, default behaviour unchanged):**
- `baseline/dat_judge.judge_set(..., candidates=None)`: optional rows for
  another dataset; without it, SQuAD candidates are built exactly as
  before. The cache file is still named by `set_name`.
- `extension/jev_judge.judge_set(..., candidates=None)`: same.
- `extension/jev_fusion.load_judgments(set_name, k, candidates=None)`: same.

**New:** `extension/nfcorpus_judges.py`: `build_dat_candidates` /
`build_jev_candidates` (top-k per retriever from the N3 score matrices,
document text = title + text, stable tie-breaking; a document in both
lists stays under both labels, as on SQuAD); `check-sizes` (context-limit
check); `run_dat` / `run_jev` (pre-run cost estimate against the $2
NFCorpus cap, then the unchanged judge, then a check that every reply is
in the cache **file**); `dat_alphas_nf` (DAT Eq. 6) and `jev_alphas_nf`
(`top_level` + Eq. 6 for J1, option C for J3, both fixed on SQuAD's tune
set). Tests: `tests/test_nfcorpus_judges.py` (7).

**Tests added** (all network calls faked; ledger and caches redirected to a
temporary folder): candidate inputs and rank positions, including a tie
and an all-zero BM25 row; J3 slot order and duplicate documents; DAT and
Jev top-1 agree on 20 random matrices; cache files separated by dataset
(`nfcorpus-test.jsonl` vs `test.jsonl`); `judge_set` uses the given
candidates and never builds SQuAD ones; without candidates it still builds
SQuAD's; a Jev round trip through `judge_set` and `load_judgments`.

**SQuAD regression check (before any paid NFCorpus judge run):** all ten
SQuAD result-producing commands were re-run from their caches:

```
python -m baseline.evaluate --tune-alpha
python -m baseline.dat_judge --set test --alphas
python -m baseline.evaluate --dat
python -m extension.jev_fusion --set tune --compare-mappings
python -m extension.jev_fusion --set test
python -m extension.jev_fusion --set tune --compare-j3
python -m extension.jev_fusion --set test --j3
python -m extension.jev_fusion --set tune --j3
python -m extension.rrf
python -m extension.final_table
```

Result: **all 10 SQuAD result files byte-identical** to their committed
versions (`baseline_checkpoint`, `dat_alphas_test`, `dat_reproduction`,
`jev_j1_test`, `jev_j1_alphas_test`, `jev_j3_options_tune`,
`jev_j3_test`, `jev_j3_tune`, `rrf_test`, `final_comparison`), the
pre-declared choices recomputed unchanged (`top_level`, option C), the
budget ledger unchanged (no API calls), and the 18 SQuAD data/cache files
byte-identical to the backup.

**Request sizes** (`python -m extension.nfcorpus_judges check-sizes`,
about 4 characters per token):

| Set | DAT prompt mean / max | Jev J1 request mean / max | Jev J3 request mean / max | J3 state + longest question, max |
|---|---|---|---|---|
| dev | 1,169 / 2,413 | 1,303 / 2,581 | 3,914 / 8,264 | 6,968 |
| test | 1,177 / 2,012 | 1,310 / 2,148 | 3,941 / 6,990 | 5,694 |
| limit | 128,000 | 64,000 | 64,000 | 32,000 |

Everything fits with a wide margin, so documents are sent whole, as on
SQuAD (no truncation, no methodology change). NFCorpus requests are about
1.6x (DAT) to 1.4x (J3) the size of SQuAD's.

**One methodological note (no change made):** for the 25 test queries
where BM25 matches nothing, BM25's "top-1" is simply the first document
of the corpus. The judges see it as BM25's best result, exactly as the
method defines it; a good judge should score it low and move alpha to
dense.

**Problems:** none. **Tests:** 102 passed (95 + 7). **Cost:** $0.
**Project total:** $1.5436 of $20 (NFCorpus $0.1602).

**Next:** N6, DAT with GPT-4o-mini: a 20-query dev pilot, then the full
test set (323 queries).

**Git (N5):** commit `47e467b` on `extension/nfcorpus` (not pushed).

### N6 DAT (GPT-4o-mini) on NFCorpus (done)

**Objective:** test whether the reproduced DAT method still improves on
the zero-judge baselines on a second, very different dataset, using the
identical judge (GPT-4o-mini via OpenRouter pinned to OpenAI, temperature
0, seed 42, max 10 tokens, the Appendix A prompt) and alpha rule (Eq. 6).
These are the same candidates Jev judges in N7.

**Commands:**

```
python -m extension.nfcorpus_judges dat --set nfcorpus-dev --limit 20   # pilot
python -m extension.nfcorpus_judges dat --set nfcorpus-test             # full test
python -m extension.nfcorpus_analysis dat                               # analysis
```

**Pilot (20 dev queries, seed 0):** 20/20 replies parsed, all served by
OpenAI with the pinned settings, mean 1,063 prompt tokens (max 2,138),
$0.00016 per call, all on disk. **Full test run:** 323/323 replies, 0 HTTP
errors, 0 unparseable, all verified on disk; 337,074 input tokens, 1.5
min with 8 parallel requests. **Cost:** pilot $0.0032 + test $0.0511 =
**$0.0544**. NFCorpus total $0.2145; **project total $1.5980 of $20**.
Per-query judge scores and alphas: `results/nfcorpus/dat_alphas_test.csv`.

**Results (test, 323 queries):**

| Method | nDCG@10 | Recall@10 | MRR@10 | P@1 |
|---|---:|---:|---:|---:|
| Dense only | 0.4220 | 0.2108 | 0.6253 | 0.5418 |
| Fixed 0.8 (dev-tuned) | 0.4252 | 0.2110 | 0.6424 | 0.5604 |
| Fixed 0.6 (SQuAD) | 0.4035 | 0.2039 | 0.6196 | 0.5325 |
| **DAT (GPT-4o-mini)** | **0.4120** | 0.2042 | 0.6308 | 0.5480 |

| DAT minus | nDCG@10 [95% CI] | Recall@10 | MRR@10 | P@1 |
|---|---|---|---|---|
| Dense only | -0.0100 [-0.0218, +0.0021] | -0.0067 [-0.0137, +0.0003] | +0.0055 [-0.0181, +0.0288] | +0.0062 [-0.0279, +0.0402] |
| **Fixed 0.8 (dev)** | **-0.0132 [-0.0214, -0.0056]** | -0.0068 [-0.0120, -0.0021] | -0.0116 [-0.0264, +0.0027] | -0.0124 [-0.0372, +0.0124] |
| Fixed 0.6 (SQuAD) | +0.0085 [-0.0002, +0.0173] | +0.0002 [-0.0042, +0.0043] | +0.0111 [-0.0041, +0.0276] | +0.0155 [-0.0093, +0.0402] |
| RRF | +0.0422 [+0.0282, +0.0572] | +0.0255 [+0.0151, +0.0375] | +0.0290 [+0.0036, +0.0557] | +0.0217 [-0.0155, +0.0588] |
| BM25 only | +0.1054 [+0.0861, +0.1257] | +0.0520 [+0.0373, +0.0690] | +0.1210 [+0.0886, +0.1545] | +0.1207 [+0.0805, +0.1641] |

**What happened.** DAT is **significantly worse than the dev-tuned fixed
alpha 0.8** (-1.32 nDCG@10 points) and not distinguishable from dense
alone (-1.00, CI spans 0). Its gain over the transferred SQuAD alpha 0.6
(+0.85) is borderline (CI touches 0). It clearly beats RRF and BM25. Per
query vs fixed 0.8: better on 52, worse on 94, unchanged on 177. **The
SQuAD finding "DAT beats fixed alpha" does not hold on NFCorpus.**

**Alpha distribution:** 0.5 for 35.0% of queries, 1.0 for 22.3%, 0.0 for
3.1%, mean 0.622 (best fixed weight on dev: 0.8). Most common score pairs
(dense, BM25): 5/5 (82), 4/1 (35), 4/2 (22), 1/4 (19), 1/0 (17), 4/4 (17).

**nDCG@10 by top-1 relevance group** (relevant = official grade >= 1):

| Group | n | DAT | Dense only | Fixed 0.8 (dev) | Fixed 0.6 (SQuAD) |
|---|---:|---:|---:|---:|---:|
| Same document from both retrievers | 87 | 0.4734 | 0.4905 | 0.4911 | 0.4783 |
| Different, both relevant (new on NFCorpus) | 53 | 0.7577 | 0.7791 | 0.7826 | 0.7626 |
| Only dense top-1 relevant | 60 | 0.5265 | **0.5839** | 0.5466 | 0.4664 |
| Only BM25 top-1 relevant | 23 | **0.4843** | 0.3120 | 0.4579 | 0.4914 |
| Different, neither relevant | 100 | 0.0901 | 0.1013 | 0.0981 | 0.0901 |

**Direction of alpha** (83 queries where exactly one top-1 is relevant):
toward the relevant retriever 79.5%, neutral 6.0%, wrong way **14.5%**
(SQuAD: 88.2% / 3.4% / 8.4%). When only dense is right: right 88.3%, mean
alpha 0.797. When only BM25 is right: right only **56.5%**, mean alpha 0.461.

**Rubric alignment** (DAT's 0-5 score vs the official grade of the 646
judged top-1 documents): Spearman **+0.433**.

| Official grade | n | Mean DAT score | Scored >= 4 | Scored <= 1 |
|---|---:|---:|---:|---:|
| 0 (not judged relevant) | 333 | 2.44 | 38% | 41% |
| 1 (partially relevant) | 216 | 4.09 | 81% | 7% |
| 2 (highly relevant) | 97 | 4.02 | 75% | 5% |

**Why might it have happened (supported by the data above):**
1. **Alpha 0.5 is a costly default on NFCorpus.** DAT returns 0.5 whenever
   both top-1 documents get equal scores (35% of queries). On SQuAD 0.5 was
   close to the best fixed weight (0.6); here the best is 0.8, so every
   neutral decision gives BM25 more weight than is good on average.
2. **Rank 1 is not all that matters.** On SQuAD, when both retrievers
   returned the same paragraph, no alpha could change P@1. NFCorpus's
   nDCG@10 also scores ranks 2-10, so alpha still matters for the 87
   same-document queries, and DAT's 0.5 there (vs 0.8) costs about 1.8
   points on that group (0.4734 vs 0.4911).
3. **The rubric does not separate grade 1 from grade 2** (mean scores 4.09
   vs 4.02) and rates 38% of grade-0 documents at 4 or 5. Some of those may
   be relevant documents missing from NFCorpus's incomplete labels, so this
   is partly a label problem, not only a judge problem (cannot be told
   apart here). Either way, the judge's view of "relevant" and the labels'
   view only partly agree (rho 0.43), so judge-chosen weights are tuned to
   a different target than the metric.
4. **Where DAT does help:** when only BM25's top-1 is relevant (23
   queries) it beats dense by 17 points (0.4843 vs 0.3120). The set is
   small, and the gain is outweighed by losses where dense is right.
5. **Zero-match BM25 queries are handled well:** on the 25 queries where
   BM25 matches nothing, DAT gave BM25's arbitrary top-1 a mean score of
   0.08 and alpha = 1.0 for 56% (mean 0.804).

**Examples.** Gain vs fixed 0.8: "hormonal dysfunction" (alpha 0.3, 0.258
vs 0.000), "The Actual Benefit of Diet vs. Drugs" (BM25's top-1 is grade
2; alpha 0.4; 0.658 vs 0.435). Loss: "blood clots": both top-1 documents
are relevant (grade 1) but DAT chose alpha 0.0 and the ranking collapsed
to BM25's (0.220 vs 0.855); "oral intraepithelial neoplasia" (alpha 0.2,
neither top-1 relevant; 0.269 vs 0.554).

**What it means (RQ3, and the context for RQ1):** on NFCorpus, an LLM
judge with DAT's rule is not worth its cost compared with a fixed alpha
tuned on dev, and not better than dense alone. This is a negative result
for DAT's generality. For RQ1 the question becomes whether Jev can do
better than DAT here.

**Cannot conclude:** that DAT is worse than dense (not significant), or
whether the low rubric alignment is due to the judge or to incomplete
labels.

**Problems:** one new analysis test had a wrong expectation (a Spearman
correlation with tied grades is 0.9487, not 1); the code was right, the
test now checks both the tied and the untied case. **Tests:** 104 passed
(102 + 2). **Limitations:** 23 queries in the only-BM25-relevant group;
group sizes limit fine-grained conclusions.

**Next:** N7, Jev J1 on the same candidates (pilot on dev, then test).

**Git (N6):** commit `bf3d9ff` on `extension/nfcorpus` (not pushed).

### N7 Jev J1 on NFCorpus (done)

**Objective (RQ1 on a second dataset):** does Jev, judging the same two
top-1 documents as DAT with the SQuAD-chosen `top_level` mapping and DAT's
Eq. 6 unchanged, match or beat GPT-4o-mini at lower cost?

**Commands:**

```
python -m extension.nfcorpus_judges jev --k 1 --set nfcorpus-dev --limit 20   # pilot
python -m extension.nfcorpus_judges jev --k 1 --set nfcorpus-test             # full test
python -m extension.nfcorpus_analysis j1                                      # analysis
```

**Pilot (20 dev queries):** 20/20 valid, all served by `jev-1.13.0`, mean
1,485 input tokens (max 2,669), $0.000062 per query, all on disk.
**Full test run:** 323/323 valid, 0 invalid, 0 HTTP errors, all on disk;
473,553 input tokens, 1.2 min. **Cost:** pilot $0.0012 + test $0.0199 =
**$0.0211**. NFCorpus total $0.2357; **project total $1.6191 of $20**.
Per-query levels and alphas: `results/nfcorpus/jev_j1_alphas_test.csv`.
Code added: `j1` analysis (judge-vs-judge disagreement, query length,
rubric alignment for Jev's levels and decimal scores) in
`extension/nfcorpus_analysis.py`.

**Results (test, 323 queries):**

| Method | nDCG@10 | Recall@10 | MRR@10 | P@1 | Judge $ / query |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.4220 | 0.2108 | 0.6253 | 0.5418 | $0 |
| Fixed 0.8 (dev-tuned) | **0.4252** | 0.2110 | **0.6424** | **0.5604** | $0 |
| DAT (GPT-4o-mini) | 0.4120 | 0.2042 | 0.6308 | 0.5480 | $0.000158 |
| **Jev J1** | **0.4173** | 0.2069 | 0.6372 | 0.5542 | **$0.000062** |

| Jev J1 minus | nDCG@10 [95% CI] | Recall@10 | MRR@10 | P@1 |
|---|---|---|---|---|
| **DAT (GPT-4o-mini)** | **+0.0053 [-0.0028, +0.0135]** | +0.0027 [-0.0027, +0.0083] | +0.0065 [-0.0103, +0.0235] | +0.0062 [-0.0217, +0.0341] |
| Dense only | -0.0047 [-0.0151, +0.0057] | -0.0039 [-0.0097, +0.0014] | +0.0120 [-0.0088, +0.0328] | +0.0124 [-0.0186, +0.0433] |
| Fixed 0.8 (dev) | -0.0079 [-0.0159, +0.0000] | -0.0041 [-0.0104, +0.0019] | -0.0052 [-0.0232, +0.0122] | -0.0062 [-0.0372, +0.0217] |
| Fixed 0.6 (SQuAD) | +0.0138 [+0.0042, +0.0239] | +0.0030 [-0.0024, +0.0086] | +0.0176 [-0.0033, +0.0393] | +0.0217 [-0.0124, +0.0557] |
| RRF | +0.0475 [+0.0318, +0.0638] | +0.0282 [+0.0178, +0.0399] | +0.0355 [+0.0057, +0.0657] | +0.0279 [-0.0155, +0.0712] |
| BM25 only | +0.1107 [+0.0902, +0.1322] | +0.0547 [+0.0401, +0.0713] | +0.1275 [+0.0935, +0.1624] | +0.1269 [+0.0805, +0.1734] |

**What happened.**
1. **J1 vs DAT:** +0.53 nDCG@10 points, **not significant** (CI [-0.28,
   +1.35]), at **2.6x lower judge cost** ($0.000062 vs $0.000158 per
   query). Per query: J1 better on 54, worse on 43, same on 226. (The
   1-point non-inferiority margin was pre-agreed for SQuAD's P@1, not for
   NFCorpus; read descriptively, the whole CI lies above -1 point.)
2. **Neither judge beats the zero-judge reference:** J1 is -0.79 below the
   dev-tuned fixed alpha (CI upper end 0.0000, borderline) and -0.47 below
   dense (n.s.); it is significantly above fixed 0.6 (+1.38), RRF and BM25.
3. **The SQuAD mechanism does not repeat.** On SQuAD, Jev won because it
   almost never pushed alpha the wrong way (1.3% vs DAT 8.4%). Here, on the
   83 queries where exactly one top-1 is relevant, J1 leans the right way
   68.7% (DAT 79.5%), neutral 15.7% (DAT 6.0%), wrong way 15.7% (DAT
   14.5%). When only **BM25's** top-1 is relevant, J1 leans toward BM25 in
   only **30.4%** of cases (DAT 56.5%); its mean alpha there is 0.522.
4. **Where J1's small edge comes from:** J1 and DAT pick the same alpha on
   51.1% of queries. Where J1 leans more toward dense (101 queries), it is
   better 45 times and worse 23 (mean +0.0304); where it leans more toward
   BM25 (57), it is better only 9 times and worse 20 (mean -0.0238). J1's
   gain is from leaning toward the stronger retriever (dense) more often,
   not from better per-query discrimination.

**Alpha distribution (J1):** 0.5 for **46.7%** (DAT 35.0%), 1.0 for 35.6%,
0.0 for 8.7%, mean 0.651. Most common level pairs (dense, BM25): 5/5 (79),
5/3 (42), 3/3 (27), 4/4 (23), 5/1 (18), 5/4 (18).

**nDCG@10 by top-1 relevance group:**

| Group | n | Jev J1 | DAT | Dense only | Fixed 0.8 (dev) |
|---|---:|---:|---:|---:|---:|
| Same document | 87 | 0.4734 | 0.4734 | 0.4905 | 0.4911 |
| Different, both relevant | 53 | 0.7769 | 0.7577 | 0.7791 | 0.7826 |
| Only dense top-1 relevant | 60 | 0.5433 | 0.5265 | 0.5839 | 0.5466 |
| Only BM25 top-1 relevant | 23 | 0.4424 | 0.4843 | 0.3120 | 0.4579 |
| Different, neither relevant | 100 | 0.0966 | 0.0901 | 0.1013 | 0.0981 |

**Rubric alignment** (646 judged top-1 documents):

| Judge | Spearman vs grade | Mean score: grade 0 / 1 / 2 | Scored >= 4: grade 0 / 1 / 2 |
|---|---:|---|---|
| DAT (GPT-4o-mini) | 0.433 | 2.44 / 4.09 / 4.02 | 38% / 81% / 75% |
| **Jev J1, most likely level** | **0.473** | 2.52 / 4.09 / **4.55** | 35% / 68% / **93%** |
| Jev J1, decimal score | **0.494** | 2.50 / 3.58 / 4.26 | 18% / 37% / 69% |

Jev's judgments agree better with the official labels and, unlike DAT,
**separate highly relevant (grade 2) from partially relevant (grade 1)**
documents. Both judges still give 4-5 to about a third of grade-0
documents (possibly unlabeled relevant documents; cannot be told apart).

**nDCG@10 by query length:**

| Queries | n | Jev J1 | DAT | Dense only | Fixed 0.8 (dev) |
|---|---:|---:|---:|---:|---:|
| 1 word | 103 | 0.4721 | 0.4744 | 0.4564 | 0.4731 |
| 2-3 words | 89 | 0.3848 | 0.3647 | 0.3901 | 0.3907 |
| 4+ words | 131 | 0.3964 | 0.3951 | 0.4167 | 0.4110 |

Short topical queries are not a particular weakness of Jev: on one-word
queries both judges match the best fixed weight and beat dense; both fall
behind dense on longer, title-style queries, where dense is strongest.

**Zero-match BM25 queries (25):** Jev's mean level for BM25's arbitrary
top-1 is 0.04; alpha 1.0 for 52% (mean 0.772). The rest stay at 0.5 when
Jev also scores the dense document 0 (Eq. 6: 0 vs 0 gives 0.5).

**Examples vs DAT.** Gain: "blood clots" (both top-1 relevant): J1 kept
alpha 0.5 (0.789) where DAT went to 0.0 (0.220); "prenatal vitamins": J1
alpha 1.0 (0.258 vs 0.000). Loss: "The Actual Benefit of Diet vs. Drugs":
BM25's top-1 is grade 2, but J1 went to alpha 1.0 (0.168 vs DAT's 0.658);
"Keeping Your Hands Warm With Citrus": J1 alpha 0.0 toward an irrelevant
hand-rub study (0.118 vs 0.504).

**Why might it have happened (plausible, partly supported):** Jev judges
the documents better (rubric alignment) but DAT's Eq. 6 turns equal levels
into 0.5, and with Jev's sharper levels (more 5/5 and 3/3 ties) that
happens on 47% of queries, where 0.5 is worse than leaning to dense. The
judge's quality is bottlenecked by the alpha rule, not only by judgment.
(Hypothesis; a rule with a dense-leaning default would test it, but that
would be a methodology change and was not run.)

**What it means for RQ1:** on NFCorpus, Jev matches DAT's quality (no
significant difference, point estimate slightly better) at 2.6x lower
cost, so the cost advantage generalizes; the quality *advantage* seen on
SQuAD (+1.64 points, significant) does **not** generalize. **Cannot
conclude** equivalence (a non-significant difference is not proof of
equality; the CI allows -0.28 to +1.35 points), nor that Jev's better
rubric alignment would help under a different alpha rule.

**Problems:** none. **Tests:** 104 passed. **Limitations:** 323 queries;
the only-BM25-relevant group has 23.

**Next:** N8, Jev J3 (top-3 per retriever, option C unchanged).

**Git (N7):** commit `47e63ff` on `extension/nfcorpus` (not pushed).

### N8 Jev J3 on NFCorpus (done)

**Objective (RQ2 on a multi-relevance dataset):** on SQuAD, judging the
top-3 documents per retriever (J3) was significantly worse than top-1
because a correct retriever's #2 and #3 were wrong by construction. Does
that change when many documents per query are relevant? Option C
(rank-weighted mean of levels, weights 1, 1/2, 1/3, then DAT Eq. 6) was
kept exactly as chosen on SQuAD's tune set.

**Commands:**

```
python -m extension.nfcorpus_judges jev --k 3 --set nfcorpus-dev --limit 20   # pilot
python -m extension.nfcorpus_judges jev --k 3 --set nfcorpus-test             # full test
python -m extension.nfcorpus_analysis j3                                      # analysis
```

**Pilot (20 dev):** 20/20 valid, 6 judgments each, `jev-1.13.0`, mean
3,978 input tokens (max 7,903), $0.000167 per query. **Full test:**
323/323 valid, 0 errors, all on disk; 1,264,125 input tokens, 1.4 min.
**Cost:** pilot $0.0033 + test $0.0531 = **$0.0564**. NFCorpus total
$0.2921; **project total $1.6756 of $20**. Per-query levels, grades and
alphas: `results/nfcorpus/jev_j3_alphas_test.csv`. Code added: `j3`
analysis (multi-relevance inside the top-3, agreement with the
better-ranking retriever, J3 vs J1 by number of relevant documents).

**Results (test, 323 queries):**

| Method | nDCG@10 | Recall@10 | MRR@10 | P@1 | Judge $ / query |
|---|---:|---:|---:|---:|---:|
| Fixed 0.8 (dev-tuned) | **0.4252** | 0.2110 | **0.6424** | **0.5604** | $0 |
| Dense only | 0.4220 | 0.2108 | 0.6253 | 0.5418 | $0 |
| Jev J1 | 0.4173 | 0.2069 | 0.6372 | 0.5542 | $0.000062 |
| **Jev J3** | **0.4158** | 0.2073 | 0.6290 | 0.5449 | $0.000164 |
| DAT (GPT-4o-mini) | 0.4120 | 0.2042 | 0.6308 | 0.5480 | $0.000158 |

| Jev J3 minus | nDCG@10 [95% CI] | Recall@10 | MRR@10 | P@1 |
|---|---|---|---|---|
| **Jev J1** | **-0.0015 [-0.0082, +0.0050]** | +0.0004 [-0.0032, +0.0042] | -0.0082 [-0.0237, +0.0067] | -0.0093 [-0.0341, +0.0155] |
| DAT | +0.0038 [-0.0046, +0.0127] | +0.0032 [-0.0013, +0.0078] | -0.0018 [-0.0157, +0.0114] | -0.0031 [-0.0279, +0.0186] |
| Dense only | -0.0062 [-0.0172, +0.0048] | -0.0035 [-0.0094, +0.0023] | +0.0037 [-0.0196, +0.0269] | +0.0031 [-0.0310, +0.0372] |
| **Fixed 0.8 (dev)** | **-0.0094 [-0.0170, -0.0020]** | -0.0037 [-0.0090, +0.0017] | -0.0134 [-0.0299, +0.0027] | -0.0155 [-0.0433, +0.0124] |
| Fixed 0.6 (SQuAD) | +0.0123 [+0.0042, +0.0209] | +0.0034 [-0.0006, +0.0076] | +0.0094 [-0.0078, +0.0275] | +0.0124 [-0.0155, +0.0402] |
| RRF | +0.0460 [+0.0302, +0.0620] | +0.0286 [+0.0182, +0.0404] | +0.0273 [-0.0002, +0.0556] | +0.0186 [-0.0217, +0.0588] |
| BM25 only | +0.1092 [+0.0887, +0.1306] | +0.0551 [+0.0402, +0.0721] | +0.1193 [+0.0864, +0.1533] | +0.1176 [+0.0743, +0.1641] |

**What happened.**
1. **The SQuAD penalty disappears, but no gain appears.** J3 - J1 is -0.15
   nDCG@10 points, not significant (CI [-0.82, +0.50]); on SQuAD it was
   -1.90 P@1 points and significant. Per query: J3 better on 43, worse on
   44, same on 236. J3 vs DAT: +0.38 (n.s.). Like the other judges, J3 is
   significantly below the dev-tuned fixed alpha (-0.94).
2. **Multiple relevance weakens the dilution, as hypothesised.** Inside
   each retriever's top-3 (official labels):

   | Retriever | Relevant at #1 / #2 / #3 | If #1 relevant: #2 / #3 relevant | Jev levels #1 / #2 / #3 (if #1 relevant) | (SQuAD, if #1 gold) |
   |---|---|---|---|---|
   | Dense | 54.2% / 42.7% / 35.0% | 60.0% / 49.1% | 4.43 / 3.31 / 2.99 | 4.94 / 2.21 / 1.64 |
   | BM25 | 42.7% / 32.2% / 24.1% | 55.1% / 39.1% | 4.18 / 3.02 / 2.19 | 4.95 / 1.30 / 0.91 |

   On SQuAD, a correct retriever's #2 and #3 could never be relevant; here
   they are relevant 39-60% of the time and Jev scores them higher (about
   3 instead of about 1-2), so averaging pulls the correct retriever down
   much less. Some dilution remains: the levels still fall with rank.
3. **J3 judges best but does not rank best.** Over all 1,938 judged
   documents, J3's levels track the official grades best of the three
   judges (Spearman **0.511**; mean level 2.14 / 3.88 / 4.39 for grades
   0 / 1 / 2; J1 top-1 only: 0.473; DAT: 0.433). Measured against the
   retriever that actually ranks better on its own (by its nDCG@10; 227
   queries where they differ), J3 leans toward it most often and wrong
   least:

   | Judge | Toward the better retriever | Neutral (0.5) | Wrong way | When BM25 is better: toward BM25 |
   |---|---:|---:|---:|---:|
   | DAT | 50.7% | 30.8% | 18.5% | 26.7% |
   | Jev J1 | 45.4% | 41.4% | 13.2% | 16.7% |
   | **Jev J3** | **53.7%** | 33.5% | **12.8%** | 25.0% |

   Yet these better decisions do not add up to a better nDCG@10: the
   neutral share is still a third, and **when BM25 is the better
   retriever (60 queries), all three judges lean toward it only 17-27% of
   the time.**
4. **By number of relevant documents:** with 31+ relevant (105 queries)
   J3 is better than J1 on 28 and worse on 19 (means 0.5648 vs 0.5638);
   with 6-30 relevant (132), better on 10 and worse on 21 (0.3269 vs
   0.3307); with 1-5 (86), 5 vs 4 (0.3704 vs 0.3714). A weak hint that
   top-3 helps only with very many relevant documents; not significant.
5. **No systematic dense bias this time** (mean alpha J3 0.630, J1 0.651,
   DAT 0.622; on SQuAD J3 was the most dense-leaning). J3 compresses
   alpha toward the middle (0.6 for 12.4% of queries, 0.5 for 36.8%), and
   it does not follow where the relevant documents are: when BM25 has more
   relevant documents in its top-3 (34 queries), J3's mean alpha is still
   0.541 (J1: 0.494).
6. **Where J3 and J1 disagree** (45.8% of queries): when J3 leans more
   toward dense (62), it wins 27 / loses 10; when it leans more toward
   BM25 (86), it wins 16 / loses 34. As for J1 vs DAT, moving weight to
   dense is what pays on NFCorpus.

**Examples vs J1.** Gains: "growth promoters" and "Atkins diet" (J3 alpha
1.0, 0.573 vs 0.290 and 0.696 vs 0.429). Losses: "factory farming
practices" (0.389 vs 0.812) and "Diabetics Should Take Their Pulses"
(dense's top-1 is grade 2, but J3 stays at 0.5: 0.283 vs 0.531).

**What it means for RQ2:** the SQuAD result ("top-3 hurts") was largely a
consequence of single-gold labels: with multiple relevant documents, J3
is no longer worse than J1, and its judgments are the best aligned with
the labels. But **top-3 judging still does not improve fusion**, and at
2.7x J1's cost (similar to DAT's) it is **not justified** on NFCorpus.

**Cannot conclude:** that J3 equals J1 (n.s. is not equivalence; CI
-0.82 to +0.50 points) or that it would help with a different alpha rule
or on datasets with even more relevant documents per query (hypotheses).

**Problems:** none. **Tests:** 104 passed. **Limitations:** 323 queries;
subgroup sizes (e.g. 34 BM25-more queries) are small.

**Next:** N9, the full consolidated analysis recomputed from the saved
outputs.
