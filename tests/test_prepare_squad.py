from baseline.prepare_squad import build_set, sample_articles


def article(title, questions_per_paragraph):
    return {
        "title": title,
        "paragraphs": [
            {"context": f"{title} p{i}", "qas": [{"id": f"{title}-{i}-{j}", "question": "q"} for j in range(n)]}
            for i, n in enumerate(questions_per_paragraph)
        ],
    }


def test_sampling_stops_before_exceeding_the_question_cap():
    articles = [article(f"a{i}", [5, 5]) for i in range(10)]  # 10 questions each
    chosen = sample_articles(articles, seed=0, max_questions=35)
    assert len(chosen) == 3  # a 4th article would reach 40 > 35
    assert chosen == sample_articles(articles, seed=0, max_questions=35)  # deterministic


def test_build_set_points_each_question_at_its_paragraph():
    corpus, queries = build_set([article("x", [2, 1])], [0], "test")
    assert [d["id"] for d in corpus] == ["test/0/0", "test/0/1"]
    assert [q["gold"] for q in queries] == ["test/0/0", "test/0/0", "test/0/1"]
