from __future__ import annotations

from typing import Any

from .chroma_store import get_chroma_client, get_embedding_function

RISK_POLICIES_COLLECTION = "risk_policies"
HISTORICAL_CASES_COLLECTION = "historical_cases"


def _distance_to_similarity(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + float(distance))


def _search_collection(collection_name: str, query: str, top_k: int) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    client = get_chroma_client()
    embedding_function = get_embedding_function(model_name="sentence-transformers/all-MiniLM-L6-v2")
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output: list[dict[str, Any]] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        output.append(
            {
                "text": text,
                "metadata": metadata or {},
                "similarity_score": _distance_to_similarity(distance),
            }
        )
    return output


def search_policies(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return _search_collection(RISK_POLICIES_COLLECTION, query=query, top_k=top_k)


def search_cases(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return _search_collection(HISTORICAL_CASES_COLLECTION, query=query, top_k=top_k)
