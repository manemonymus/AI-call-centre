"""
Ingest a CSV file into the ChromaDB knowledge base.

Usage:
    python ingest.py                                   # uses home_services_faq.csv
    python ingest.py path/to/my_data.csv               # custom file, auto-detect columns
    python ingest.py data.csv --q question --a answer  # explicit column names

The script treats each row as one document: "Q: <question>\\nA: <answer>".
It skips rows already in the collection so you can safely re-run on the same file.

For Kaggle datasets:
    1. Download a customer-service or FAQ dataset as CSV from Kaggle.
    2. Run:  python ingest.py kaggle_file.csv --q <question_col> --a <answer_col>
    3. Check column names with:  python -c "import csv; print(open('file.csv').readline())"

Recommended Kaggle datasets:
    - "Bitext Customer Support LLM Chatbot Training Dataset"
      bitext-team/bitext-customer-support-llm-chatbot-training-dataset
      columns: instruction (question), response (answer)
    - "FAQ Questions and Answers for Chatbot Training"
      Search Kaggle for 'FAQ chatbot training'
"""

import argparse
import csv
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from loguru import logger

# Keep in sync with rag.py
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "home_services_kb"
DB_PATH = "./chroma_db"


def _guess_columns(headers: list[str]) -> tuple[str, str]:
    """Try to auto-detect question and answer column names."""
    q_candidates = ["question", "instruction", "query", "q", "input", "utterance"]
    a_candidates = ["answer", "response", "reply", "a", "output", "text"]
    q_col = next((h for c in q_candidates for h in headers if c in h.lower()), None)
    a_col = next((h for c in a_candidates for h in headers if c in h.lower()), None)
    return q_col, a_col


def ingest(
    filepath: str,
    question_col: str | None = None,
    answer_col: str | None = None,
    category_col: str | None = None,
    source_name: str | None = None,
    batch_size: int = 100,
) -> int:
    path = Path(filepath)
    if not path.exists():
        logger.error(f"File not found: {filepath}")
        sys.exit(1)

    source_name = source_name or path.stem

    logger.info(f"Connecting to ChromaDB at {DB_PATH} ...")
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"Reading {filepath} ...")
    rows = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        if not question_col or not answer_col:
            q_col, a_col = _guess_columns(headers)
            question_col = question_col or q_col
            answer_col = answer_col or a_col

        if not question_col or not answer_col:
            logger.error(
                f"Could not auto-detect columns. Headers found: {headers}\n"
                "Use --q and --a to specify them explicitly."
            )
            sys.exit(1)

        logger.info(f"Using columns: question='{question_col}', answer='{answer_col}'")
        for row in reader:
            rows.append(row)

    # Build documents
    docs, metadatas, ids = [], [], []
    existing_ids = set(collection.get(include=[])["ids"])

    for i, row in enumerate(rows):
        q = str(row.get(question_col, "")).strip()
        a = str(row.get(answer_col, "")).strip()
        cat = str(row.get(category_col, "")) if category_col else ""

        if not q or not a:
            continue

        doc_id = f"{source_name}_{i}"
        if doc_id in existing_ids:
            continue  # skip duplicates

        docs.append(f"Q: {q}\nA: {a}")
        metadatas.append({"source": source_name, "question": q, "category": cat})
        ids.append(doc_id)

    if not docs:
        logger.info("No new documents to add (all already ingested).")
        return 0

    # Add in batches
    total = 0
    for start in range(0, len(docs), batch_size):
        end = start + batch_size
        collection.add(
            documents=docs[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        total += len(docs[start:end])
        logger.info(f"  Embedded {total}/{len(docs)} documents ...")

    logger.info(
        f"Done. Added {total} new documents. "
        f"Collection now has {collection.count()} total documents."
    )
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a CSV into the knowledge base.")
    parser.add_argument("file", nargs="?", default="home_services_faq.csv")
    parser.add_argument("--q", dest="question_col", help="Question column name")
    parser.add_argument("--a", dest="answer_col", help="Answer column name")
    parser.add_argument("--cat", dest="category_col", help="Category column name (optional)")
    parser.add_argument("--source", help="Source label for metadata (default: filename)")
    args = parser.parse_args()

    ingest(
        filepath=args.file,
        question_col=args.question_col,
        answer_col=args.answer_col,
        category_col=args.category_col,
        source_name=args.source,
    )
