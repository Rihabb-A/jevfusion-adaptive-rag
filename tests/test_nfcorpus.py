"""NFCorpus experiment module: namespacing, data loading, budget guard."""

import json

import numpy as np
import pytest

from baseline import retrieve
from extension import nfcorpus

HAVE_DATA = (nfcorpus.ROOT / "data" / "nfcorpus" / "raw" / "nfcorpus" / "corpus.jsonl").exists()


def test_paths_are_namespaced_away_from_squad():
    for set_name in nfcorpus.SETS:
        path = nfcorpus.score_path(set_name, "bm25")
        assert path.parent == nfcorpus.CACHE and path.name.startswith("nfcorpus-")
        assert path != retrieve.score_path("test", "bm25") and path != retrieve.score_path("tune", "bm25")


def test_embedding_cache_file_is_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(nfcorpus, "CACHE", tmp_path)
    cache = nfcorpus.NFCorpusEmbeddingCache("openai/text-embedding-3-large")
    squad_file = retrieve.CACHE_DIR / "embeddings_openai_text-embedding-3-large.npz"
    assert cache.path.parent == tmp_path and cache.path != squad_file
    assert cache.vectors == {}  # did not read the SQuAD cache


def test_unknown_set_is_rejected():
    with pytest.raises(ValueError):
        nfcorpus.load_split("test")  # the SQuAD name must never resolve to NFCorpus


def test_choose_alpha_rule():
    grid = [(0.6, {"ndcg@10": 0.40, "recall@10": 0.20}), (0.8, {"ndcg@10": 0.42, "recall@10": 0.19}),
            (0.9, {"ndcg@10": 0.42, "recall@10": 0.21})]
    assert nfcorpus.choose_alpha(grid) == 0.9  # nDCG tie -> higher recall
    grid[2][1]["recall@10"] = 0.19
    assert nfcorpus.choose_alpha(grid) == 0.8  # full tie -> closest to 0.6


def test_alpha_can_only_be_tuned_on_dev():
    with pytest.raises(ValueError):
        nfcorpus.tune_alpha("nfcorpus-test")


def test_budget_guard_refuses_above_nfcorpus_cap(monkeypatch):
    monkeypatch.setattr(nfcorpus, "nfcorpus_spent", lambda: 1.9)
    with pytest.raises(SystemExit):
        nfcorpus.ensure_nfcorpus_budget(0.2)


def test_load_scores_rejects_mismatched_order(tmp_path, monkeypatch):
    monkeypatch.setattr(nfcorpus, "CACHE", tmp_path)
    monkeypatch.setattr(nfcorpus, "load_split", lambda s: ([{"id": "d1"}], [{"id": "q1"}], {}))
    np.save(tmp_path / "nfcorpus-test_bm25.npy", np.zeros((1, 1)))
    np.save(tmp_path / "nfcorpus-test_dense.npy", np.zeros((1, 1)))
    (tmp_path / "nfcorpus-test_meta.json").write_text(json.dumps({"query_ids_sha": "x", "doc_ids_sha": "y"}))
    with pytest.raises(SystemExit):
        nfcorpus.load_scores("nfcorpus-test")


@pytest.mark.skipif(not HAVE_DATA, reason="NFCorpus not downloaded (python -m extension.beir_data)")
def test_load_split_uses_title_plus_text_and_split_queries():
    docs, qs, qrels = nfcorpus.load_split("nfcorpus-test")
    assert len(docs) == 3633 and len(qs) == 323 == len(qrels)
    assert [q["id"] for q in qs] == sorted(qrels)
    assert all(d["text"].startswith(d["title"]) for d in docs)
    dev_ids = {q["id"] for q in nfcorpus.load_split("nfcorpus-dev")[1]}
    assert not dev_ids & {q["id"] for q in qs}
