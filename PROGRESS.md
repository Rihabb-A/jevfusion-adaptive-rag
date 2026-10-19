# JevFusion: project progress log

A detailed, cumulative record of everything done in this project: what we
did, how, why, what it cost and what came out. Updated after every
completed step until the project ends.

**Last updated:** 19 October 2026, after Step 2.7.2 (J3 on the test set).

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
| 1 | 1.6 | `baseline/README.md`, commits, push | done; **PR awaiting team review** | free |
| 2 | 2.1 | Branch, Issue 3 text, design decisions | done; **Issue 3 to open on GitHub** | free |
| 2 | 2.2 | Jev price and budget estimate; pin `jev-1.13.0` | done | free |
| 2 | 2.3 | Jev judge code (J1/J3) with offline tests | done | free |
| 2 | 2.4 | Jev J1 pilot (20 tune questions) | done | $0.001 |
| 2 | 2.5 | Jev J1 on full tune set; choose score mapping | done: **top_level** | $0.132 |
| 2 | 2.6 | Jev J1 on test set, compare with DAT | done: **J1 beats DAT** | $0.141 |
| 2 | 2.7.1 | J3 on tune set; choose alpha option | done: **C chosen; J3 below J1 on tune** | $0.335 |
| 2 | 2.7.2 | J3 (option C) on test set, compare with J1 and DAT | done: **J3 < J1 (significant), J3 ≈ DAT** | $0.353 |
| 2 | 2.8 | Control: top-1 with J3's formula | done by construction (option C with k=1 is exactly J1) | free |
| 2 | 2.9 | RRF baseline | **next** | free |
| 2 | 2.10 | Full comparison: quality, CIs, cost, fair latency | to do | small |
| 2 | 2.11 | `extension/README.md`, PR, close Issue 3 | to do | free |
| 3 | 3.1-3.3 | Report, slides (max 8), defense | to do | free |

**Spend so far: $1.327 of $20** (see §8).

**Headline so far (test set, 2,996 questions):** Jev J1 P@1 **0.8899** vs
DAT (GPT-4o-mini) 0.8735 vs fixed alpha 0.6 0.8571. Jev J1 beats DAT by
+1.64 points P@1 (95% CI [+1.00, +2.27]) at less than half the cost per
question.

**RQ2 answer (confirmed on test):** judging the top-3 paragraphs per
retriever (J3) is *worse* than top-1 (J1): P@1 0.8708 vs 0.8899, -1.90
points (95% CI [-2.60, -1.23]), at 2.5x the cost. On single-gold SQuAD
the extra paragraphs dilute the top-1 signal (§5, Steps 2.7.1-2.7.2).

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

`extension/jev` starts from `baseline/setup` (the baseline PR is not
merged yet) and is not pushed yet. **66 unit tests** pass (`pytest`).

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
| **Total** | | **$1.327 of $20** |

Remaining planned spend: latency measurement (Step 2.10), small. Optional: GPT-4o-mini top-3 control (~$0.70), DAT with
GPT-4o (~$5.30), DRCD (~$0.40).

---

## 9. Open items for the team

- Open the PR `baseline/setup -> main`, get a teammate review, merge;
  close Issue 2 (all acceptance criteria met).
- Open Issue 3 (Extension) with the drafted motivation text.
- Decide later: DRCD; the optional GPT-4o-mini top-3 control run.

## 10. Next step

**Step 2.9 (RRF baseline, free):** add reciprocal rank fusion
(`sum of 1 / (60 + rank)` over both retrievers), a strong zero-judge
baseline that does not depend on score normalization, and compare it with
fixed 0.6, DAT, J1 and J3 on the test set. Answers part of RQ3 (are the
judges worth their cost against zero-call baselines?).
