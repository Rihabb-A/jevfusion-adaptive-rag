"""Jev judge tests against fake API responses; no network calls."""

import numpy as np
import pytest

from extension import jev_judge
from extension.jev_judge import DAT_RUBRIC, build_request, call_jev, parse_jev, topk_indices

PASSAGES = [
    {"slot": "dense_1", "method": "dense", "rank": 1, "id": "p1", "text": "SAME TEXT"},
    {"slot": "bm25_1", "method": "bm25", "rank": 1, "id": "p1", "text": "SAME TEXT"},
]


def answer(score, probabilities, confidence=0.9):
    return {"type": "score", "score": score, "confidence": confidence,
            "probabilities": {str(level): p for level, p in enumerate(probabilities)}}


def payload(**answers):
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 900, "output_tokens": 40}}


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body or {}
        self.text = str(body)
        self.headers = headers or {}

    def json(self):
        return self._body


def test_topk_indices_best_first_with_corpus_order_ties():
    scores = np.array([[0.2, 0.9, 0.9, 0.5]])
    assert topk_indices(scores, 3).tolist() == [[1, 2, 3]]
    assert topk_indices(scores, 1).tolist() == [[int(scores.argmax())]]  # same as baseline top-1


def test_build_request_one_score_question_per_paragraph():
    body = build_request("jev-1.13.0", "Who?", PASSAGES)
    assert body["model"] == "jev-1.13.0"
    assert body["state"] == {"question": "Who?", "passages": {"dense_1": "SAME TEXT", "bm25_1": "SAME TEXT"}}
    assert set(body["questions"]) == {"dense_1", "bm25_1"}  # same paragraph still judged under both labels
    for slot, question in body["questions"].items():
        assert question["type"] == "score"
        assert question["criteria"] == DAT_RUBRIC and len(DAT_RUBRIC) == 6
        assert slot in question["instructions"]
    assert "BM25 retrieval" in body["questions"]["bm25_1"]["instructions"]


def test_parse_valid_reply_keeps_score_level_and_probabilities():
    reply = payload(dense_1=answer(4.88, [0, 0, 0.01, 0.01, 0.05, 0.93]),
                    bm25_1=answer(1.5, [0, 0.5, 0.5, 0, 0, 0]))
    judgments, error = parse_jev(reply, ["dense_1", "bm25_1"], "jev-1.13.0")
    assert error is None
    assert judgments["dense_1"]["score"] == 4.88 and judgments["dense_1"]["top_level"] == 5
    assert judgments["bm25_1"]["top_level"] == 1  # tie between levels 1 and 2 -> lower level
    assert judgments["dense_1"]["probabilities"] == [0, 0, 0.01, 0.01, 0.05, 0.93]


@pytest.mark.parametrize(
    "reply, reason",
    [
        (payload(dense_1=answer(3.0, [0, 0, 0, 1, 0, 0])), "missing score answer for bm25_1"),
        (payload(dense_1=answer(5.4, [0] * 5 + [1]), bm25_1=answer(1.0, [0, 1, 0, 0, 0, 0])), "outside 0-5"),
        (payload(dense_1=answer(3.0, [0, 0, 0, 1, 0]), bm25_1=answer(1.0, [0, 1, 0, 0, 0, 0])), "expected 0-5"),
        ({**payload(), "model": "jev-1.14.0"}, "expected jev-1.13.0"),
    ],
)
def test_parse_rejects_invalid_replies(reply, reason):
    judgments, error = parse_jev(reply, ["dense_1", "bm25_1"], "jev-1.13.0")
    assert judgments is None and reason in error


def test_call_retries_rate_limit_then_succeeds(monkeypatch):
    responses = [FakeResponse(429, {"error": "slow down"}, {"retry-after": "0"}), FakeResponse(200, {"ok": 1})]
    calls = []
    monkeypatch.setattr(jev_judge.requests, "post", lambda *a, **kw: calls.append(1) or responses.pop(0))
    monkeypatch.setattr(jev_judge.time, "sleep", lambda seconds: None)
    result = call_jev({}, "https://example.invalid/v1", "key")
    assert result["http_status"] == 200 and result["payload"] == {"ok": 1} and len(calls) == 2


@pytest.mark.parametrize("status", [401, 422])
def test_call_does_not_retry_auth_or_validation_errors(monkeypatch, status):
    calls = []
    monkeypatch.setattr(jev_judge.requests, "post",
                        lambda *a, **kw: calls.append(1) or FakeResponse(status, {"error": "no"}))
    monkeypatch.setattr(jev_judge.time, "sleep", lambda seconds: None)
    result = call_jev({}, "https://example.invalid/v1", "key")
    assert result["http_status"] == status and len(calls) == 1
