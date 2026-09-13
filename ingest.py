"""
Ingest the HR Q&A dataset into per-department ChromaDB collections.

Each row of hr_faq.csv (question, answer, department) goes into that
department's own collection (hr_leave, hr_conduct, hr_compliance, hr_general),
so departments retrieve only their own documents.

Usage:
    python fetch_hr_data.py     # build hr_faq.csv from the real HR dataset first
    python ingest.py            # ingest hr_faq.csv into per-department collections
    python ingest.py other.csv --q question --a answer --cat department

Re-running is safe: rows already present (by id) are skipped.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

from rag import DEPARTMENTS, get_collection

DEFAULT_FILE = "hr_faq.csv"


def _guess_columns(headers: list[str]) -> tuple[str | None, str | None]:
    q_candidates = ["question", "instruction", "query", "q", "input", "utterance"]
    a_candidates = ["answer", "response", "reply", "a", "output", "text"]
    q_col = next((h for c in q_candidates for h in headers if c in h.lower()), None)
    a_col = next((h for c in a_candidates for h in headers if c in h.lower()), None)
    return q_col, a_col


def ingest(
    filepath: str,
    question_col: str | None = None,
    answer_col: str | None = None,
    category_col: str | None = "department",
    batch_size: int = 16,
) -> int:
    path = Path(filepath)
    if not path.exists():
        logger.error(
            f"File not found: {filepath}. Run `python fetch_hr_data.py` first to "
            "build hr_faq.csv."
        )
        sys.exit(1)

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not question_col or not answer_col:
            q_col, a_col = _guess_columns(headers)
            question_col = question_col or q_col
            answer_col = answer_col or a_col
        if not question_col or not answer_col:
            logger.error(f"Could not detect question/answer columns. Headers: {headers}")
            sys.exit(1)
        has_dept = category_col in headers
        rows = list(reader)

    logger.info(
        f"Columns: question='{question_col}', answer='{answer_col}', "
        f"department='{category_col if has_dept else '(none → all to general)'}'"
    )

    # Group rows by department.
    by_dept: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for i, row in enumerate(rows):
        q = str(row.get(question_col, "")).strip()
        a = str(row.get(answer_col, "")).strip()
        if not q or not a:
            continue
        dept = str(row.get(category_col, "general")).strip().lower() if has_dept else "general"
        if dept not in DEPARTMENTS:
            dept = "general"
        by_dept[dept].append((i, q, a))

    total_added = 0
    for dept in sorted(by_dept):
        collection = get_collection(dept)
        existing = set(collection.get(include=[])["ids"])
        docs, metas, ids = [], [], []
        for i, q, a in by_dept[dept]:
            doc_id = f"{dept}_{i}"
            if doc_id in existing:
                continue
            docs.append(f"Q: {q}\nA: {a}")
            metas.append({"department": dept, "question": q})
            ids.append(doc_id)

        if not docs:
            logger.info(f"  {dept}: nothing new ({collection.count()} already present)")
            continue

        for start in range(0, len(docs), batch_size):
            end = start + batch_size
            collection.add(
                documents=docs[start:end],
                metadatas=metas[start:end],
                ids=ids[start:end],
            )
        total_added += len(docs)
        logger.info(f"  {dept}: added {len(docs)} (collection now {collection.count()})")

    logger.info(f"Done. Added {total_added} documents across {len(by_dept)} departments.")
    return total_added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest HR Q&A into per-department collections.")
    parser.add_argument("file", nargs="?", default=DEFAULT_FILE)
    parser.add_argument("--q", dest="question_col", help="Question column name")
    parser.add_argument("--a", dest="answer_col", help="Answer column name")
    parser.add_argument("--cat", dest="category_col", default="department", help="Department column")
    args = parser.parse_args()

    ingest(
        filepath=args.file,
        question_col=args.question_col,
        answer_col=args.answer_col,
        category_col=args.category_col,
    )
