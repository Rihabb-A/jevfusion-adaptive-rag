"""
Step 1.1 of the DAT reproduction: build the SQuAD evaluation sets.

DAT (paper §5.1) randomly sampled whole SQuAD articles, kept all their
paragraphs and questions, and stopped "before the next sampled article
would exceed" ~3000 questions. The answer-containing paragraph is each
question's single gold document. We apply the same rule twice, on
disjoint SQuAD v1.1 splits:

- test: sampled from the dev split. Every reported number comes from here.
- tune: sampled from the train split. Used only to pick a fixed alpha (and
  later any extension constants), never touching test questions.

The paper does not name its split or seed. Its sample (585 paragraphs,
2976 questions, 5.1 questions per paragraph) matches the dev split's
density (5.1) better than train's (4.6), so dev is our test source.

    python -m baseline.prepare_squad

Writes data/squad/<set>/{corpus,queries}.jsonl (gitignored) and
data/squad_sample_manifest.json (committed: seed, checksums, titles,
counts), enough to rebuild the identical sample.
"""

import hashlib
import json
import random
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "squad" / "raw"
SETS_DIR = ROOT / "data" / "squad"
MANIFEST = ROOT / "data" / "squad_sample_manifest.json"

SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}
SETS = {"test": "dev", "tune": "train"}  # evaluation set -> SQuAD v1.1 split
SEED = 42
MAX_QUESTIONS = 3000  # DAT §5.1


def download(split):
    """Fetch a SQuAD v1.1 split once; return its path and sha256."""
    path = RAW_DIR / f"{split}-v1.1.json"
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        response = requests.get(SQUAD_URLS[split], timeout=300)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def question_count(article):
    return sum(len(paragraph["qas"]) for paragraph in article["paragraphs"])


def sample_articles(articles, seed=SEED, max_questions=MAX_QUESTIONS):
    """Shuffle article order with `seed` and take articles until the next one
    would push the question count past `max_questions` (DAT §5.1)."""
    order = list(range(len(articles)))
    random.Random(seed).shuffle(order)
    chosen, total = [], 0
    for index in order:
        count = question_count(articles[index])
        if total + count > max_questions:
            break
        chosen.append(index)
        total += count
    return chosen


def build_set(articles, chosen, set_name):
    """Flatten the chosen articles into a paragraph corpus and a query list
    whose `gold` field is the id of the answer-containing paragraph."""
    corpus, queries = [], []
    for article_index in chosen:
        article = articles[article_index]
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            pid = f"{set_name}/{article_index}/{paragraph_index}"
            corpus.append({"id": pid, "title": article["title"], "text": paragraph["context"]})
            for qa in paragraph["qas"]:
                queries.append({"id": qa["id"], "question": qa["question"], "gold": pid})
    return corpus, queries


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_set(set_name):
    """Return (corpus, queries) for 'test' or 'tune', as written by main()."""
    folder = SETS_DIR / set_name
    if not (folder / "corpus.jsonl").exists():
        raise SystemExit(f"{folder} missing, run: python -m baseline.prepare_squad")
    return read_jsonl(folder / "corpus.jsonl"), read_jsonl(folder / "queries.jsonl")


def main():
    manifest = {"seed": SEED, "max_questions": MAX_QUESTIONS, "squad_version": "1.1", "sets": {}}
    for set_name, split in SETS.items():
        path, sha256 = download(split)
        articles = json.loads(path.read_text(encoding="utf-8"))["data"]
        chosen = sample_articles(articles)
        corpus, queries = build_set(articles, chosen, set_name)
        write_jsonl(SETS_DIR / set_name / "corpus.jsonl", corpus)
        write_jsonl(SETS_DIR / set_name / "queries.jsonl", queries)
        manifest["sets"][set_name] = {
            "source_split": split,
            "source_url": SQUAD_URLS[split],
            "source_sha256": sha256,
            "articles": len(chosen),
            "paragraphs": len(corpus),
            "questions": len(queries),
            "article_indices": chosen,
            "article_titles": [articles[i]["title"] for i in chosen],
        }
        print(f"{set_name:>4} (SQuAD {split}): {len(chosen)} articles, "
              f"{len(corpus)} paragraphs, {len(queries)} questions")
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("DAT paper SQuAD sample: 13 articles, 585 paragraphs, 2976 questions")
    print(f"Manifest written to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
