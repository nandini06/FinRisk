from __future__ import annotations

from typing import Any

from ..rag.reranker import rerank_documents
from ..rag.retriever import search_policies
from .evidence import normalize_retrieval_rows


def retrieve_policy_evidence(policy_query: str, top_k: int = 3) -> list[dict[str, Any]]:
    query = policy_query.strip()
    if not query:
        return []

    retrieved_documents = search_policies(query=query, top_k=max(top_k * 2, 5))
    reranked_documents = rerank_documents(query=query, retrieved_documents=retrieved_documents, top_k=top_k)

    evidence: list[dict[str, Any]] = []
    for document in reranked_documents[:top_k]:
        evidence.append(
            {
                "text": document.get("text", ""),
                "metadata": document.get("metadata", {}),
                "similarity_score": document.get("similarity_score", 0.0),
            }
        )
    return normalize_retrieval_rows(evidence, kind="policy", metadata_key="policy_id")
