import zipfile

import pytest

from extension.beir_data import doc_text, read_qrels, safe_extract, verify


def test_read_qrels_graded_multi_relevance(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\nq2\td3\t1\n", encoding="utf-8")
    assert read_qrels(path) == {"q1": {"d1": 2, "d2": 1}, "q2": {"d3": 1}}


def test_read_qrels_rejects_unexpected_header(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("a\tb\tc\nq1\td1\t1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        read_qrels(path)


def test_safe_extract_refuses_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.txt", "x")
    with pytest.raises(SystemExit):
        safe_extract(archive, tmp_path / "target")
    assert not (tmp_path / "outside.txt").exists()


def test_doc_text_is_title_plus_text():
    assert doc_text({"title": "Vitamin D", "text": "and bones."}) == "Vitamin D and bones."
    assert doc_text({"title": "", "text": "only text"}) == "only text"


def test_verify_flags_missing_documents_and_counts_relevance():
    corpus = [{"_id": "d1", "title": "t", "text": "x"}, {"_id": "d2", "title": "", "text": "y"}]
    queries = {"q1": "vitamin d", "q2": "bones"}
    splits = {"test": {"q1": {"d1": 2, "d2": 1}, "q2": {"d9": 1}}}
    report, problems = verify(corpus, queries, splits)
    assert report["splits"]["test"]["relevant_per_query_grade_ge_1"]["max"] == 2
    assert report["splits"]["test"]["grade_counts"] == {"1": 2, "2": 1}
    assert any("not in corpus" in p for p in problems)
