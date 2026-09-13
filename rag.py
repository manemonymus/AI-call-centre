"""
Per-department knowledge-base search (ChromaDB + Ollama embeddings).

Each HR department has its OWN ChromaDB collection (hr_leave, hr_conduct,
hr_compliance, hr_general), so a department only ever retrieves its own
documents — "each department trained on its own data." make_search_tool(dept)
returns a Pipecat Flows tool bound to one department's collection; each
department node gets its own instance.

Setup (one time):
    ollama pull nomic-embed-text
    pip install chromadb
    python fetch_hr_data.py     # builds hr_faq.csv from a real HR dataset
    python ingest.py            # loads it into per-department collections

Embeddings run locally via Ollama (free); this is independent of whichever
chat LLM / STT you choose, but Ollama must be running for RAG to work.
"""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from loguru import logger

from pipecat_flows import FlowArgs, FlowManager, FlowsFunctionSchema

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL = "nomic-embed-text"   # pull with: ollama pull nomic-embed-text
OLLAMA_URL = "http://localhost:11434"
DB_PATH = "./chroma_db"            # created in the project directory
COLLECTION_PREFIX = "hr_"

DEPARTMENTS = ("leave", "conduct", "compliance", "general")

# Human-readable label per department, used in tool descriptions.
DEPARTMENT_LABEL = {
    "leave": "leave, time-off, PTO, compensatory off, overtime, and attendance",
    "conduct": "code of conduct, gifts, anti-bribery, harassment, and ethics",
    "compliance": "reporting violations, disciplinary action, and compliance",
    "general": "general HR policy, how policies are reviewed, updated, and communicated",
}


def collection_name(department: str) -> str:
    return f"{COLLECTION_PREFIX}{department}"


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------
def get_collection(department: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name=collection_name(department),
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def search_kb(query: str, department: str, n_results: int = 3) -> list[dict]:
    """Top-k most relevant chunks for the query, from ONE department's collection."""
    try:
        collection = get_collection(department)
        if collection.count() == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )
        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({"content": doc, "category": meta.get("department", department)})
        return docs
    except Exception as e:
        logger.warning(f"KB search failed for '{department}': {e}")
        return []


# ---------------------------------------------------------------------------
# Pipecat Flows tool factory — one tool per department.
# ---------------------------------------------------------------------------
def make_search_tool(department: str) -> FlowsFunctionSchema:
    """Build a search_knowledge_base tool scoped to a single department's KB."""

    async def _search_handler(args: FlowArgs, flow_manager: FlowManager):
        query = str(args.get("query", "")).strip()
        if not query:
            return {"found": False, "note": "Empty query."}, None

        results = search_kb(query, department)
        log = flow_manager.state.get("log")
        if log:
            log.event("kb_search", department=department, query=query, hits=len(results))

        if not results:
            return {
                "found": False,
                "note": "Nothing in this department's knowledge base covers that.",
            }, None

        context = "\n\n".join(f"- {r['content']}" for r in results)
        logger.info(f"KB[{department}] '{query}' → {len(results)} result(s)")
        return {"found": True, "context": context}, None

    return FlowsFunctionSchema(
        name="search_knowledge_base",
        description=(
            f"Search the {department} knowledge base, which covers "
            f"{DEPARTMENT_LABEL.get(department, department)}. Use this whenever the "
            "caller asks a specific question — answer ONLY from what it returns, "
            "and never invent policies, numbers, or rules it doesn't contain."
        ),
        properties={
            "query": {
                "type": "string",
                "description": "A concise search query describing what the caller wants to know.",
            }
        },
        required=["query"],
        handler=_search_handler,
    )
