import numpy as np
import pytest

from baseline.dat_judge import (
    JUDGE_SETTINGS,
    JudgeCache,
    build_prompt,
    parse_scores,
    request_key,
    top1_indices,
)


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("3 4", (3, 4)),
        (" 5 0\n", (5, 0)),
        ("0  0", (0, 0)),
        ("3 4\n(Vector: 3, BM25: 4)", (3, 4)),  # echoed the prompt's example line
        ("3, 4", (3, 4)),
        ("6 2", None),  # out of DAT's 0-5 range
        ("3 4 5", None),
        ("34", None),
        ("Vector: 3, BM25: 4", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_scores(reply, expected):
    assert parse_scores(reply) == expected


def test_build_prompt_fills_all_three_slots_in_paper_order():
    prompt = build_prompt("Who?", "DENSE PARAGRAPH {braces}", "BM25 PARAGRAPH")
    assert '**Question:** "Who?"' in prompt
    assert '**dense retrieval Top1 Result:** "DENSE PARAGRAPH {braces}"' in prompt
    assert '**BM25 retrieval Top1 Result:** "BM25 PARAGRAPH"' in prompt
    assert prompt.index("DENSE PARAGRAPH") < prompt.index("BM25 PARAGRAPH")
    assert "{" not in prompt.replace("{braces}", "")  # no unfilled placeholders


def test_request_key_changes_with_any_input():
    base = request_key("m", "p", {"temperature": 0})
    assert base == request_key("m", "p", {"temperature": 0})
    assert base != request_key("m2", "p", {"temperature": 0})
    assert base != request_key("m", "p2", {"temperature": 0})
    assert base != request_key("m", "p", {"temperature": 1})


def test_judge_settings_pin_one_host_deterministically():
    assert JUDGE_SETTINGS["temperature"] == 0
    assert JUDGE_SETTINGS["provider"] == {"order": ["openai"], "allow_fallbacks": False}
    request_key("m", "p", JUDGE_SETTINGS)  # settings must be JSON-serializable for the cache key


def test_judge_cache_persists_and_skips_truncated_line(tmp_path):
    path = tmp_path / "judge.jsonl"
    cache = JudgeCache(path)
    cache.put({"key": "a", "raw": "3 4"})
    cache.put({"key": "b", "raw": "5 0"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "c", "raw": "1')  # simulated crash mid-write

    reloaded = JudgeCache(path)
    assert len(reloaded) == 2
    assert reloaded.get("a")["raw"] == "3 4"
    assert reloaded.get("c") is None


def test_top1_indices_breaks_ties_by_corpus_order():
    assert top1_indices(np.array([[0.2, 0.9, 0.9], [0.5, 0.1, 0.0]])).tolist() == [1, 0]
