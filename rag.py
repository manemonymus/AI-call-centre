"""
Knowledge-base search using ChromaDB and Ollama embeddings.

Setup (one time):
    ollama pull nomic-embed-text
    pip install chromadb
    python ingest.py                     # loads home_services_faq.csv
    python ingest.py my_kaggle_data.csv  # or any CSV with question/answer columns

The SEARCH_KB_TOOL can then be added to any FlowManager node.
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
COLLECTION_NAME = "home_services_kb"
DB_PATH = "./chroma_db"            # created in the project directory


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------
def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def search_kb(query: str, n_results: int = 3) -> list[dict]:
    """Return the top-k most relevant chunks for the query."""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )
        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({"content": doc, "category": meta.get("category", "")})
        return docs
    except Exception as e:
        logger.warning(f"Knowledge base search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Pipecat Flows tool handler
# ---------------------------------------------------------------------------
async def _search_handler(args: FlowArgs, flow_manager: FlowManager):
    query = str(args.get("query", "")).strip()
    if not query:
        return {"found": False, "note": "Empty query."}, None

    results = search_kb(query)
    if not results:
        return {
            "found": False,
            "note": "No relevant information found in the knowledge base.",
        }, None

    context = "\n\n".join(
        f"[{r['category'].title() or 'Info'}] {r['content']}" for r in results
    )
    logger.info(f"KB search '{query}' → {len(results)} result(s)")
    return {"found": True, "context": context}, None


SEARCH_KB_TOOL = FlowsFunctionSchema(
    name="search_knowledge_base",
    description=(
        "Search the company knowledge base for information about services, "
        "pricing, policies, warranties, scheduling, or any FAQ the caller asks "
        "about. Use this whenever the caller asks a specific question you don't "
        "know the answer to from memory."
    ),
    properties={
        "query": {
            "type": "string",
            "description": (
                "A concise search query describing what the caller wants to know, "
                "e.g. 'HVAC tune-up cost' or 'cancellation policy'."
            ),
        }
    },
    required=["query"],
    handler=_search_handler,
)
