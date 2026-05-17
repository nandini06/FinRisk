from __future__ import annotations

from typing import Any

from ..rag.reranker import rerank_documents
from ..rag.retriever import search_cases


def retrieve_similar_cases(case_query: str, top_k: int = 3) -> list[dict[str, Any]]:
    query = case_query.strip()
    if not query:
        return []

    retrieved_documents = search_cases(query=query, top_k=max(top_k * 2, 5))
    reranked_documents = rerank_documents(query=query, retrieved_documents=retrieved_documents, top_k=top_k)

    cases: list[dict[str, Any]] = []
    for document in reranked_documents[:top_k]:
        cases.append(
            {
                "text": document.get("text", ""),
                "metadata": document.get("metadata", {}),
                "similarity_score": document.get("similarity_score", 0.0),
            }
        )
    return cases
