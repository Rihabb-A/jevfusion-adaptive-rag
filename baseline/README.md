# Baseline: DAT (Dynamic Alpha Tuning) reproduction

**Paper:** Hsin-Ling Hsu and Jengnan Tzeng, *DAT: Dynamic Alpha Tuning for
Hybrid Retrieval in Retrieval-Augmented Generation*, 2025,
[arXiv:2503.23013v1](https://arxiv.org/abs/2503.23013).

**How this folder was filled:** option 2 of the course template,
*reimplemented from the paper*. The paper links no code, and no official
release had been found when the project was planned, so everything here
is our own implementation of the method as the paper specifies it
(§3-5, Appendix A). Nothing in this folder is our extension;
that lives in `extension/`.

**Status:** full run on SQuAD (not a smoke test). DAT with GPT-4o-mini
reproduces the paper within 1.2 points on P@1 and MRR@20 and within 2.0
points on hybrid-subset alpha accuracy (all metrics in the tables below),
and its gain over fixed-weight fusion is statistically significant. We
label it a **DAT-style reproduction** because our random article sample
differs from the authors' (they published no seed); see
[Deviations](#deviations-from-the-paper).

## What DAT does

For each question, BM25 and dense retrieval each rank the paragraphs. An
LLM sees only the **top-1 paragraph from each** and scores both 0-5 with
the paper's rubric. The two scores set a per-question weight `alpha`
(paper Eq. 6), and paragraphs are ranked by
`alpha * dense + (1 - alpha) * BM25` over min-max normalized scores. The
baseline it beats is the same fusion with one fixed `alpha` for every
question.

## Results: paper vs ours (SQuAD)

Test set: 13 SQuAD v1.1 dev articles, 621 paragraphs, **2,996 questions**
(paper: 13 articles, 585 paragraphs, 2,976 questions).

**All questions** (paper Tables 2 and 3)

| Method | P@1 paper | P@1 ours | MRR@20 paper | MRR@20 ours | Alpha acc. paper | Alpha acc. ours |
|---|---:|---:|---:|---:|---:|---:|
| BM25 only (alpha = 0) | 0.7594 | 0.7610 | 0.8223 | 0.8289 | 0.7981 | 0.7894 |
| Dense only (alpha = 1) | 0.7396 | 0.7730 | 0.8119 | 0.8489 | 0.7789 | 0.8194 |
| Fixed hybrid (alpha = 0.6) | 0.8461 | 0.8571 | 0.8997 | 0.9087 | 0.8975 | 0.8975 |
| **DAT (GPT-4o-mini)** | **0.8676** | **0.8735** | **0.9093** | **0.9167** | **0.9194** | **0.9169** |

**Hybrid-sensitive questions** (paper Tables 4 and 5; paper n = 1,111, ours n = 1,056)

| Method | P@1 paper | P@1 ours | MRR@20 paper | MRR@20 ours | Alpha acc. paper | Alpha acc. ours |
|---|---:|---:|---:|---:|---:|---:|
| BM25 only (alpha = 0) | 0.3906 | 0.3494 | 0.5420 | 0.5288 | 0.4590 | 0.4025 |
| Dense only (alpha = 1) | 0.3375 | 0.3835 | 0.5143 | 0.5855 | 0.4077 | 0.4877 |
| Fixed hybrid (alpha = 0.6) | 0.6229 | 0.6222 | 0.7493 | 0.7554 | 0.7255 | 0.7093 |
| **DAT (GPT-4o-mini)** | **0.6805** | **0.6686** | **0.7750** | **0.7779** | **0.7840** | **0.7642** |

**DAT's gain over fixed alpha = 0.6** (paired, same 2,996 questions, 10,000
bootstrap resamples; the paper reports no significance test):

| | Paper | Ours | Our 95% CI |
|---|---:|---:|---|
| P@1, all questions | +0.0215 | **+0.0164** | [+0.0080, +0.0247] |
| MRR@20, all questions | +0.0096 | **+0.0079** | [+0.0029, +0.0130] |
| P@1, hybrid-sensitive | +0.0576 | **+0.0464** | |

Versus fixed 0.6, DAT moves the gold paragraph to rank 1 for 107 questions
and off rank 1 for 58 (net +49). Our gain is somewhat smaller than the
paper's, consistent with our dense retriever being about 3 points stronger
(P@1 0.7730 vs 0.7396), which raises the fixed-alpha baseline and leaves
less room for per-question weighting.

Our own grid search on a separate tuning set independently selected
**alpha\* = 0.6**, the same fixed value the paper reports.

Machine-readable tables: `results/dat_reproduction.csv` (all rows above),
`results/dat_alphas_test.csv` (judge scores and alpha for every question),
`results/baseline_checkpoint.csv` (baselines only).

## How to reproduce

Setup (Windows PowerShell; on macOS/Linux use `source .venv/bin/activate`):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then set OPENROUTER_API_KEY
python scripts/verify_setup.py
```

Then, from the repository root:

| # | Command | Produces | Time | Cost |
|---|---|---|---|---|
| 1 | `python -m baseline.prepare_squad` | test and tune samples, `data/squad_sample_manifest.json` | ~1 min | free |
| 2 | `python -m baseline.retrieve --set test` | BM25 and dense score matrices for the test set | ~4 min | ~$0.017 |
| 3 | `python -m baseline.retrieve --set tune` | same for the tuning set | ~4 min | ~$0.017 |
| 4 | `python -m baseline.evaluate --tune-alpha` | alpha grid search on tune; BM25, dense, fixed rows | seconds | free |
| 5 | `python -m baseline.dat_judge --set test` | GPT-4o-mini judgment for every test question | ~45 min | ~$0.33 |
| 6 | `python -m baseline.dat_judge --set test --alphas` | `results/dat_alphas_test.csv` | seconds | free |
| 7 | `python -m baseline.evaluate --dat` | full tables above, `results/dat_reproduction.csv`, paired test | seconds | free |

Every paid call is cached (embeddings by text hash, judge replies by exact
request) and logged with its billed cost in `results/budget_ledger.csv`,
so re-running any step costs nothing. Scripts refuse to start a run whose
estimate would exceed `BUDGET_CAP_USD` (default 20). Our total spend for
this reproduction: **$0.37**.

Optional pilot before step 5: `python -m baseline.dat_judge --set test --limit 20`
(about $0.002; its replies are reused by step 5).

Unit tests: `pytest` (38 tests: fusion, `dat_alpha` cases, metrics,
sampling, prompt, parser, cache).

## Deviations from the paper

Things that differ from the paper, or that the paper does not specify and
we had to choose:

1. **Article sample.** The paper names neither the SQuAD split nor the seed.
   We sample with DAT's rule (§5.1: random whole articles, stop before
   exceeding 3,000 questions), seed 42, from the **v1.1 dev** split, whose
   question density (5.1 per paragraph) matches the paper's sample better
   than train (4.6). Result: 13 / 621 / 2,996 vs the paper's 13 / 585 / 2,976.
   Exact numerical replication would need the authors' article list.
2. **Tuning set.** The paper tunes fixed alpha on "validation sets" it does
   not describe. Ours is a disjoint sample of 13 train-split articles
   (2,865 questions), used only for the alpha grid search.
3. **BM25.** The paper says "standard word tokenization". We use
   `rank_bm25.BM25Okapi` (k1 = 1.5, b = 0.75) over lowercase `\w+` tokens,
   no stemming or stop-word removal.
4. **Dense retrieval.** Same model as the paper (`text-embedding-3-large`),
   called through OpenRouter, embedding raw paragraph text without the
   article title (the paper does not say). Our dense retriever scores about
   3 points P@1 higher than the paper's; the cause (sample or formatting)
   is unknown.
5. **Candidate pool and normalization.** The paper normalizes over each
   method's retrieved list without giving its size. Our corpora are small
   (~620 paragraphs), so both retrievers score the full corpus and min-max
   normalization runs over all paragraphs. Ties in ranking are broken by
   corpus order.
6. **Judge.** Same model (GPT-4o-mini) and prompt (Appendix A) as the paper.
   Through OpenRouter, pinned to OpenAI's own endpoint with no fallback;
   temperature 0, seed 42, max 10 output tokens (the paper states no
   generation settings). The model was called in October 2026, so the
   served GPT-4o-mini snapshot may differ from the one the authors used in
   early 2025. The prompt's punctuation spacing was restored from the PDF's
   code font; no words were changed (`baseline/dat_judge.py`).
7. **Judge failures.** All 2,996 replies were in the exact `d b` format, all
   served by OpenAI; no retries or fallback alphas were needed.
8. **Alpha rounding.** The paper rounds alpha to one decimal without saying
   how halves round. We use Python's `round` (half to even), so 0.25 becomes
   0.2 and 0.75 becomes 0.8.
9. **Hybrid-sensitive subset and alpha accuracy.** The paper's prose
   definitions are loose. We use: a question is hybrid-sensitive if its gold
   paragraph's rank changes anywhere on the alpha grid 0.0 to 1.0 (step 0.1);
   a method's alpha is correct if it achieves the best gold rank on that
   grid. This is the definition consistent with the paper's tables: with it,
   the questions outside the subset contribute identically to every
   method's P@1 (1,826 hits) and alpha accuracy (1,865) in Tables 2-5.

## Not reproduced, and what it would take

- **DAT with GPT-4o** (paper: P@1 0.8740): same pipeline with
  `DAT_JUDGE_MODEL=openai/gpt-4o`; about $5.30 at current OpenRouter prices.
  Skipped to keep budget for the extension.
- **DAT with DeepSeek-R1-Distill-Qwen-14B** (paper: 0.8663): not offered on
  OpenRouter when we checked; would need local inference on a GPU with
  roughly 30 GB of memory (or a quantized build).
- **DRCD** (Traditional Chinese, paper's second dataset): needs the
  `ckiplab/albert-base-chinese` tokenizer for BM25 and a DRCD sampler; API
  cost similar to SQuAD (under $0.50 with GPT-4o-mini).

## Files

| File | Role |
|---|---|
| `prepare_squad.py` | SQuAD v1.1 download, DAT article sampling, test/tune sets |
| `retrieve.py` | BM25 and dense score matrices, embedding cache |
| `dat_fusion.py` | min-max normalization, linear fusion, `dat_alpha` (Eq. 6) |
| `dat_judge.py` | Appendix A prompt, reply parser, cached parallel judge calls, per-question alphas |
| `evaluate.py` | P@1, MRR@20, alpha accuracy, hybrid subset, alpha grid search, paired bootstrap, paper reference values |
| `budget.py` | spend ledger and USD 20 cap check |
