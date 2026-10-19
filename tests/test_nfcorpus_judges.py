"""NFCorpus judge pipeline: candidates, cache separation, backward
compatibility of the shared judge code. No network: API calls are faked."""

import numpy as np
import pytest

from baseline import budget, dat_judge
from extension import jev_fusion, jev_judge
from extension.nfcorpus_judges import build_dat_candidates, build_jev_candidates

DOCS = [{"id": f"MED-{i}", "text": f"title {i} text {i}"} for i in range(5)]
QUERIES = [{"id": "PLAIN-1", "question": "vitamin d"}, {"id": "PLAIN-2", "question": "statins"}]
BM25 = np.array([[0.0, 3.0, 3.0, 1.0, 0.0],   # top: MED-1 (tie with MED-2 -> corpus order), MED-2, MED-3
                 [0.0, 0.0, 0.0, 0.0, 0.0]])  # matches nothing -> corpus order: MED-0 first
DENSE = np.array([[0.1, 0.9, 0.2, 0.8, 0.3],  # top: MED-1, MED-3, MED-4
                  [0.5, 0.1, 0.7, 0.2, 0.6]])  # top: MED-2, MED-4, MED-0


def test_dat_candidates_inputs_and_positions():
    rows = build_dat_candidates(DOCS, QUERIES, BM25, DENSE)
    assert [r["query_id"] for r in rows] == ["PLAIN-1", "PLAIN-2"]
    assert rows[0]["question"] == "vitamin d"
    assert (rows[0]["dense_top1"], rows[0]["bm25_top1"]) == ("MED-1", "MED-1") and rows[0]["same_top1"]
    assert rows[0]["dense_text"] == rows[0]["bm25_text"] == "title 1 text 1"
    assert (rows[1]["dense_top1"], rows[1]["bm25_top1"]) == ("MED-2", "MED-0")  # all-zero BM25 -> first doc


def test_jev_candidates_order_ranks_and_duplicates():
    rows = build_jev_candidates(DOCS, QUERIES, BM25, DENSE, k=3)
    passages = rows[0]["passages"]
    assert [p["slot"] for p in passages] == ["dense_1", "dense_2", "dense_3", "bm25_1", "bm25_2", "bm25_3"]
    assert [p["id"] for p in passages] == ["MED-1", "MED-3", "MED-4", "MED-1", "MED-2", "MED-3"]
    assert [p["rank"] for p in passages] == [1, 2, 3, 1, 2, 3]
    # MED-1 and MED-3 appear in both lists and are kept under both labels
    assert {p["slot"] for p in passages if p["id"] == "MED-3"} == {"dense_2", "bm25_3"}


def test_dat_and_jev_top1_agree():
    rng = np.random.default_rng(0)
    bm25, dense = rng.random((20, 5)), rng.random((20, 5))
    dat = build_dat_candidates(DOCS, [{"id": f"q{i}", "question": "x"} for i in range(20)], bm25, dense)
    jev = build_jev_candidates(DOCS, [{"id": f"q{i}", "question": "x"} for i in range(20)], bm25, dense, k=1)
    for d, j in zip(dat, jev):
        assert d["dense_top1"] == j["passages"][0]["id"] and d["bm25_top1"] == j["passages"][1]["id"]


def test_judge_caches_are_separated_by_dataset():
    assert dat_judge.cache_path("openai/gpt-4o-mini", "nfcorpus-test") != dat_judge.cache_path("openai/gpt-4o-mini", "test")
    assert dat_judge.cache_path("m", "nfcorpus-test").name == "nfcorpus-test.jsonl"
    assert jev_judge.cache_path("jev-1.13.0", "nfcorpus-test", 3).name == "nfcorpus-test_k3.jsonl"
    assert jev_judge.cache_path("jev-1.13.0", "nfcorpus-test", 3) != jev_judge.cache_path("jev-1.13.0", "test", 3)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """No network, no real ledger, no real caches."""
    monkeypatch.setattr(budget, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(dat_judge, "CACHE_DIR", tmp_path / "judge")
    monkeypatch.setattr(jev_judge, "CACHE_DIR", tmp_path / "jev")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setenv("DAT_JUDGE_MODEL", "openai/gpt-4o-mini")
    return tmp_path


def test_dat_judge_set_uses_given_candidates(sandbox, monkeypatch):
    monkeypatch.setattr(dat_judge, "top1_candidates", lambda s: pytest.fail("SQuAD candidates must not be built"))
    monkeypatch.setattr(dat_judge, "call_judge", lambda *a, **kw: {
        "http_status": 200, "raw": "4 1", "parsed": (4, 1), "served_model": "openai/gpt-4o-mini",
        "provider": "OpenAI", "prompt_tokens": 10, "completion_tokens": 3, "cost_usd": 1e-6, "latency_ms": 1.0})
    rows = build_dat_candidates(DOCS, QUERIES, BM25, DENSE)
    results = dat_judge.judge_set("nfcorpus-test", workers=1, candidates=rows)
    assert [r["query_id"] for _, r in results] == ["PLAIN-1", "PLAIN-2"]
    assert (sandbox / "judge" / "openai_gpt-4o-mini" / "nfcorpus-test.jsonl").exists()
    assert "nfcorpus-test" in (sandbox / "ledger.csv").read_text(encoding="utf-8")


def test_dat_judge_set_default_still_builds_squad_candidates(sandbox, monkeypatch):
    called = []
    monkeypatch.setattr(dat_judge, "top1_candidates", lambda s: called.append(s) or [])
    dat_judge.judge_set("test", workers=1)
    assert called == ["test"]


def test_jev_judge_set_uses_given_candidates_and_loads_back(sandbox, monkeypatch):
    monkeypatch.setattr(jev_judge, "jev_config", lambda: ("fake", "https://example.invalid/v1", "jev-1.13.0"))
    monkeypatch.setattr(jev_fusion, "jev_config", lambda: ("fake", "https://example.invalid/v1", "jev-1.13.0"))
    monkeypatch.setattr(jev_judge, "topk_candidates", lambda s, k: pytest.fail("SQuAD candidates must not be built"))
    levels = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 1.0}

    def fake_call(body, *a, **kw):
        answers = {slot: {"type": "score", "score": 5.0, "confidence": 1.0, "probabilities": levels}
                   for slot in body["questions"]}
        return {"http_status": 200, "latency_ms": 1.0,
                "payload": {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 100}}}

    monkeypatch.setattr(jev_judge, "call_jev", fake_call)
    rows = build_jev_candidates(DOCS, QUERIES, BM25, DENSE, k=1)
    results = jev_judge.judge_set("nfcorpus-test", k=1, workers=1, candidates=rows)
    assert all(r["judgments"] for _, r in results)
    judgments = jev_fusion.load_judgments("nfcorpus-test", 1, candidates=rows)
    assert jev_fusion.j1_alphas(judgments, "top_level").tolist() == [0.5, 0.5]  # 5 vs 5 -> 0.5
