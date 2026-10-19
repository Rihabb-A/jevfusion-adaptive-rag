# JevFusion: project progress log

A detailed, cumulative record of everything done in this project: what we
did, how, why, what it cost and what came out. Updated after every
completed step until the project ends.

**Last updated:** 19 October 2026, after Step 2.5.

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
| 2 | 2.6 | Jev J1 on test set, compare with DAT | **next** | ~$0.14 |
| 2 | 2.7 | J3: choose aggregation on tune, run on test | to do | ~$0.67 |
| 2 | 2.8 | Control: top-1 with J3's formula | to do | free (reuses J1) |
| 2 | 2.9 | RRF baseline | to do | free |
| 2 | 2.10 | Full comparison: quality, CIs, cost, fair latency | to do | small |
| 2 | 2.11 | `extension/README.md`, PR, close Issue 3 | to do | free |
| 3 | 3.1-3.3 | Report, slides (max 8), defense | to do | free |

**Spend so far: $0.498 of $20** (see §8).

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

`extension/jev` starts from `baseline/setup` (the baseline PR is not
merged yet) and is not pushed yet. **55 unit tests** pass (`pytest`).

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
| **Total** | | **$0.498 of $20** |

Remaining planned spend: J1 test ~$0.14, J3 tune + test ~$0.67, latency
measurement small. Optional: GPT-4o-mini top-3 control (~$0.70), DAT with
GPT-4o (~$5.30), DRCD (~$0.40).

---

## 9. Open items for the team

- Open the PR `baseline/setup -> main`, get a teammate review, merge;
  close Issue 2 (all acceptance criteria met).
- Open Issue 3 (Extension) with the drafted motivation text.
- Decide later: DRCD; the optional GPT-4o-mini top-3 control run.

## 10. Next step

**Step 2.6:** run Jev J1 (mapping `top_level`) on the 2,996 test
questions (~$0.14, ~25 min) and compare with DAT (GPT-4o-mini) and fixed
0.6 on the same questions, with a paired bootstrap test.
