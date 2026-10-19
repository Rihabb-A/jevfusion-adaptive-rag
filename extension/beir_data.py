"""
Phase 4, Step N1: download and verify a BEIR dataset (NFCorpus).

Source: the official BEIR download link and MD5 listed in the BEIR README
(github.com/beir-cellar/beir). The archive is checked against that MD5
before use, and extracted only inside data/<dataset>/raw (entries that
would land outside it are refused).

BEIR format: corpus.jsonl (_id, title, text), queries.jsonl (_id, text),
qrels/<split>.tsv (query-id, corpus-id, score). NFCorpus labels are
graded, and a query can have many relevant documents, unlike SQuAD.

Nothing here touches SQuAD files: everything is written under
data/nfcorpus/, and the committed manifest is data/nfcorpus_manifest.json.

    python -m extension.beir_data            # download (once), verify, write manifest
"""

import argparse
import csv
import hashlib
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    # From the BEIR README dataset table (checked 2026-10-20).
    "nfcorpus": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "md5": "a89dba18a62ef92f7d323ec890a0d38d",
        "readme_test_queries": 323,
        "readme_corpus": "3.6K",
        "readme_avg_relevant_per_query": 38.2,
    },
}


def data_dir(dataset):
    return ROOT / "data" / dataset


def file_hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def download(dataset):
    """Fetch the archive once into data/<dataset>/raw and check its MD5."""
    spec = DATASETS[dataset]
    raw = data_dir(dataset) / "raw"
    archive = raw / f"{dataset}.zip"
    if not archive.exists():
        raw.mkdir(parents=True, exist_ok=True)
        with requests.get(spec["url"], stream=True, timeout=300) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                for block in response.iter_content(1 << 20):
                    handle.write(block)
    md5, sha256 = file_hashes(archive)
    if md5 != spec["md5"]:
        raise SystemExit(f"MD5 mismatch for {archive.name}: got {md5}, BEIR lists {spec['md5']}")
    return archive, md5, sha256


def safe_extract(archive, target):
    """Extract a zip, refusing absolute paths or entries escaping `target`."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            destination = (target / member.filename).resolve()
            if member.filename.startswith(("/", "\\")) or not destination.is_relative_to(target):
                raise SystemExit(f"Refusing unsafe archive entry: {member.filename!r}")
        zf.extractall(target)


def dataset_root(dataset):
    """Folder holding corpus.jsonl / queries.jsonl / qrels after extraction."""
    return data_dir(dataset) / "raw" / dataset


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_qrels(path):
    """{query_id: {doc_id: grade}} from a BEIR qrels TSV (with header)."""
    qrels = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if [h.strip().lower() for h in header[:3]] != ["query-id", "corpus-id", "score"]:
            raise SystemExit(f"Unexpected qrels header in {path.name}: {header}")
        for row in reader:
            if not row:
                continue
            query_id, doc_id, grade = row[0], row[1], int(row[2])
            qrels.setdefault(query_id, {})[doc_id] = grade
    return qrels


def load_beir(dataset):
    """(corpus, queries, {split: qrels}) as written by the BEIR archive."""
    root = dataset_root(dataset)
    corpus = read_jsonl(root / "corpus.jsonl")
    queries = {q["_id"]: q["text"] for q in read_jsonl(root / "queries.jsonl")}
    splits = {p.stem: read_qrels(p) for p in sorted((root / "qrels").glob("*.tsv"))}
    return corpus, queries, splits


def doc_text(doc):
    """Indexed text for both retrievers: title + text (BEIR convention)."""
    return f"{doc.get('title', '')} {doc.get('text', '')}".strip()


def describe(values):
    return {"mean": round(statistics.mean(values), 2), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def verify(corpus, queries, splits):
    """Integrity checks and descriptive statistics. Returns (report, problems)."""
    problems = []
    doc_ids = [d["_id"] for d in corpus]
    if len(set(doc_ids)) != len(doc_ids):
        problems.append(f"{len(doc_ids) - len(set(doc_ids))} duplicate document ids")
    empty = sum(not doc_text(d) for d in corpus)
    if empty:
        problems.append(f"{empty} documents with empty title and text")
    doc_set = set(doc_ids)
    report = {
        "corpus_documents": len(corpus),
        "documents_with_title": sum(bool(d.get("title")) for d in corpus),
        "doc_words_title_plus_text": describe([len(doc_text(d).split()) for d in corpus]),
        "doc_chars_title_plus_text": describe([len(doc_text(d)) for d in corpus]),
        "queries_total": len(queries),
        "splits": {},
    }
    for split, qrels in splits.items():
        missing_queries = [q for q in qrels if q not in queries]
        missing_docs = sorted({d for rel in qrels.values() for d in rel if d not in doc_set})
        if missing_queries:
            problems.append(f"{split}: {len(missing_queries)} judged queries not in queries.jsonl")
        if missing_docs:
            problems.append(f"{split}: {len(missing_docs)} judged documents not in corpus")
        grades = Counter(g for rel in qrels.values() for g in rel.values())
        relevant = [sum(g > 0 for g in rel.values()) for rel in qrels.values()]
        highly = [sum(g >= 2 for g in rel.values()) for rel in qrels.values()]
        report["splits"][split] = {
            "queries_with_judgments": len(qrels),
            "judgments": sum(len(rel) for rel in qrels.values()),
            "grade_counts": {str(g): c for g, c in sorted(grades.items())},
            "relevant_per_query_grade_ge_1": describe(relevant),
            "highly_relevant_per_query_grade_ge_2": describe(highly),
            "queries_without_any_relevant": sum(r == 0 for r in relevant),
            "query_words": describe([len(queries[q].split()) for q in qrels if q in queries]),
        }
    overlaps = {}
    names = sorted(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]) & set(splits[b])
            overlaps[f"{a}&{b}"] = len(shared)
            if shared and {a, b} & {"test"}:
                problems.append(f"{len(shared)} queries shared between {a} and {b}")
    report["query_overlap_between_splits"] = overlaps
    return report, problems


def main():
    parser = argparse.ArgumentParser(description="Download and verify a BEIR dataset")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="nfcorpus")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    archive, md5, sha256 = download(args.dataset)
    print(f"Archive: {archive.relative_to(ROOT)}  MD5 {md5} (matches BEIR README)  SHA256 {sha256}")
    if not (dataset_root(args.dataset) / "corpus.jsonl").exists():
        safe_extract(archive, archive.parent)
    corpus, queries, splits = load_beir(args.dataset)
    report, problems = verify(corpus, queries, splits)

    manifest = {
        "dataset": args.dataset, "source_url": spec["url"], "md5": md5, "sha256": sha256,
        "beir_readme": {k.removeprefix("readme_"): v for k, v in spec.items() if k.startswith("readme_")},
        "indexed_text": "title + ' ' + text, for both BM25 and dense",
        "files": {p.relative_to(dataset_root(args.dataset)).as_posix(): file_hashes(p)[1]
                  for p in sorted(dataset_root(args.dataset).rglob("*")) if p.is_file()},
        "verification": report, "problems": problems,
    }
    out = ROOT / "data" / f"{args.dataset}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Problems:", problems or "none")
    print(f"Manifest written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
