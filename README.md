# JevFusion: a cheaper relevance judge for DAT's adaptive hybrid retrieval

Team project for **Data Science for Conversational AI** (topic: grounding /
retrieval-augmented generation), built on the course's team template.

**Paper reproduced:** Hsin-Ling Hsu and Jengnan Tzeng, *DAT: Dynamic Alpha
Tuning for Hybrid Retrieval in Retrieval-Augmented Generation*, 2025,
[arXiv:2503.23013](https://arxiv.org/abs/2503.23013). DAT combines BM25
and dense retrieval with a per-question weight `alpha`, set by an LLM
(GPT-4o-mini) that scores the top-1 result of each retriever.

**Our extension, JevFusion:** replace DAT's general-purpose LLM judge with
**Jev** (TypeSafe's decision model), and test whether judging the **top-3**
results per retriever beats judging the top-1. Evaluated on SQuAD (DAT's
own dataset) and on BEIR NFCorpus (a second, multi-relevance dataset).

## Results at a glance

**SQuAD** (2,996 test questions, one gold paragraph each; metric P@1):

| Method | P@1 | Judge cost / 1,000 questions |
|---|---:|---:|
| Fixed alpha 0.6 (best fixed weight) | 0.8571 | $0 |
| DAT, GPT-4o-mini (our reproduction; paper 0.8676) | 0.8735 | $0.110 |
| **JevFusion J1 (Jev, top-1)** | **0.8899** | **$0.047** |
| JevFusion J3 (Jev, top-3) | 0.8708 | $0.118 |

**BEIR NFCorpus** (323 test queries, many graded relevant documents each;
metric nDCG@10):

| Method | nDCG@10 | Judge cost / 1,000 queries |
|---|---:|---:|
| **Fixed alpha 0.8 (tuned on NFCorpus dev)** | **0.4252** | $0 |
| Dense only | 0.4220 | $0 |
| JevFusion J1 | 0.4173 | **$0.062** |
| JevFusion J3 | 0.4158 | $0.164 |
| DAT, GPT-4o-mini | 0.4120 | $0.158 |

**In one sentence each:**
- **Reproduction:** our DAT matches the paper within 1.2 points on P@1
  and MRR@20 and within 2.0 points on hybrid-subset alpha accuracy, and
  its gain over fixed fusion is significant.
- **RQ1, Jev as the judge:** on SQuAD Jev beats GPT-4o-mini (+1.64 P@1,
  significant); on NFCorpus the difference (+0.53 nDCG@10) is not
  significant and inconclusive at 323 queries.
  On both, Jev is **2.3-2.6x cheaper** and about **0.5 s faster** per call.
- **RQ2, top-3 judging:** worse than top-1 on SQuAD (-1.90 P@1,
  significant), no difference on NFCorpus; never worth its extra cost.
- **RQ3, judges vs fixed fusion:** judges beat fixed fusion on SQuAD, but
  on NFCorpus no judge beats a fixed weight tuned on its dev split: the
  benefit of adaptive weighting is dataset-dependent.

All comparisons use paired bootstrap confidence intervals on the same
queries; total API spend for the whole project: **$1.71 of the $20 cap**.

## Where to read more

| Document | Contents |
|---|---|
| [`baseline/README.md`](baseline/README.md) | DAT reproduction on SQuAD: paper vs our numbers, commands, deviations |
| [`extension/README.md`](extension/README.md) | JevFusion on SQuAD: method, results, why J1 wins and J3 loses, reproduce |
| [`extension/NFCORPUS.md`](extension/NFCORPUS.md) | The NFCorpus experiment: setup, results, analysis, limitations, reproduce |
| [`PROGRESS.md`](PROGRESS.md) | Full working log: every step, decision, problem, cost and commit |
| [`report/paper/JevFusion__Evaluating_a_Specialized_Decision_Model_forCost_Efficient_Adaptive_Hybrid_Retrieval.pdf`](report/paper/JevFusion__Evaluating_a_Specialized_Decision_Model_forCost_Efficient_Adaptive_Hybrid_Retrieval.pdf) | The research paper (ACL format, PDF): *JevFusion: Evaluating a Specialized Decision Model for Cost-Efficient Adaptive Hybrid Retrieval* |
| [`report/REPORT.md`](report/REPORT.md) | The paper-style report (in preparation) |
| [`slides/`](slides/README.md) | Defense slides (in preparation) |
| [`data/README.md`](data/README.md) | Datasets, how to obtain them, what is cached |

## Repository layout

```
baseline/        DAT reproduction (SQuAD): sampling, BM25 and dense retrieval,
                 GPT-4o-mini judge, fusion, evaluation, budget ledger
extension/       JevFusion: Jev judge (J1/J3), fusion and analysis, RRF,
                 latency, final tables; NFCorpus data, metrics, judges, analysis
results/         Committed result tables (SQuAD) and results/nfcorpus/ (NFCorpus);
                 budget_ledger.csv records every paid API call
data/            Dataset manifests (committed); datasets and caches (gitignored)
tests/           105 unit tests (pytest), including fake-API tests of the judges
scripts/         verify_setup.py (environment check)
report/, slides/ Course deliverables
```

## Setup

Python 3.10+ (developed on 3.13). On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env              # then fill in the keys below
python scripts/verify_setup.py      # must report no FAIL
```

`.env` (never committed) needs `OPENROUTER_API_KEY` (embeddings and the
GPT-4o-mini judge, via OpenRouter) and `JEV_API_KEY` (Jev, via
`api.typesafe.ai`). `.env.example` documents every variable, including the
pinned models (`openai/gpt-4o-mini`, `openai/text-embedding-3-large`,
`jev-1.13.0`) and `BUDGET_CAP_USD=20`.

## Reproduce

Each document lists its exact commands with time and cost:

| Part | Commands | API cost |
|---|---|---:|
| DAT on SQuAD | [`baseline/README.md`](baseline/README.md#how-to-reproduce), 7 commands | ~$0.37 |
| JevFusion on SQuAD | [`extension/README.md`](extension/README.md#8-reproduce), 11 commands | ~$1.02 |
| NFCorpus | [`extension/NFCORPUS.md`](extension/NFCORPUS.md#8-reproduce), 12 commands | ~$0.33 |

Every paid reply is cached by its exact request (re-runs cost nothing) and
logged with its billed cost in `results/budget_ledger.csv`; runs stop
before exceeding the budget cap. Run `pytest` for the unit tests.

## How we worked

Course workflow from the template (see [`CONTRIBUTING.md`](CONTRIBUTING.md)):
the four milestone issues (idea pitch, baseline reproduction, extension,
report and slides), one branch per piece of work, and pull requests
reviewed by a teammate before merging into `main`
(PR #1 baseline, PR #2 SQuAD extension; NFCorpus on `extension/nfcorpus`).
Decisions that could bias results (score mappings, aggregation formulas,
fusion weights, significance margins) were fixed on tuning or dev data
before any test result was seen; all of them are recorded in `PROGRESS.md`.

## Data

Public datasets only (SQuAD v1.1 and BEIR NFCorpus); no personal data, no
client data, no real users' information. See
[`data/README.md`](data/README.md).
