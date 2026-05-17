from __future__ import annotations

import re
from typing import Any

TRANSACTION_TYPES = {
    "wire",
    "card",
    "cash_payment",
    "cash",
    "crypto_transfer",
    "crypto",
    "ach",
    "check",
    "international_remittance",
}

COUNTRY_ALIASES = {
    "usa": "united states",
    "uk": "united kingdom",
    "uae": "united arab emirates",
}

AMOUNT_THRESHOLD_TERMS = {
    "5m",
    "5 million",
    "high value",
    "high-value",
    "cash",
    "large transfer",
    "large payment",
    "threshold",
}

POLICY_CATEGORY_TERMS = {
    "jurisdiction",
    "behavior",
    "crypto",
    "merchant",
    "customer_history",
    "hni",
    "velocity",
    "wire",
    "high_value_cash",
}

WORD_PATTERN = re.compile(r"[a-z0-9_]+")


def _to_text(document: dict[str, Any]) -> str:
    text = str(document.get("text", "")).lower()
    metadata = document.get("metadata") or {}
    metadata_text = " ".join(str(value) for value in metadata.values()).lower()
    return f"{text} {metadata_text}".strip()


def _normalize_query(query: str) -> str:
    query_text = query.lower().strip()
    for alias, canonical in COUNTRY_ALIASES.items():
        query_text = re.sub(rf"\b{re.escape(alias)}\b", canonical, query_text)
    return query_text


def _tokens(text: str) -> set[str]:
    return set(WORD_PATTERN.findall(text.lower()))


def _phrase_matches(query_text: str, doc_text: str, terms: set[str]) -> int:
    matches = 0
    for term in terms:
        if term in query_text and term in doc_text:
            matches += 1
    return matches


def rerank_documents(
    query: str, retrieved_documents: list[dict[str, Any]], top_k: int = 5
) -> list[dict[str, Any]]:
    query_text = _normalize_query(query)
    query_tokens = _tokens(query_text)

    scored_documents: list[dict[str, Any]] = []
    for doc in retrieved_documents:
        base_score = float(doc.get("similarity_score", 0.0))
        doc_text = _normalize_query(_to_text(doc))
        doc_tokens = _tokens(doc_text)

        exact_keyword_matches = len(query_tokens.intersection(doc_tokens))
        transaction_type_matches = _phrase_matches(query_text, doc_text, TRANSACTION_TYPES)
        country_matches = _phrase_matches(query_text, doc_text, set(COUNTRY_ALIASES.values()))
        amount_threshold_matches = _phrase_matches(query_text, doc_text, AMOUNT_THRESHOLD_TERMS)
        policy_category_matches = _phrase_matches(query_text, doc_text, POLICY_CATEGORY_TERMS)

        keyword_boost = min(exact_keyword_matches, 8) * 0.01
        transaction_boost = transaction_type_matches * 0.04
        country_boost = country_matches * 0.03
        amount_boost = amount_threshold_matches * 0.05
        policy_boost = policy_category_matches * 0.035

        final_score = base_score + keyword_boost + transaction_boost + country_boost + amount_boost + policy_boost

        reranked_doc = dict(doc)
        reranked_doc["similarity_score"] = round(final_score, 6)
        reranked_doc["score_breakdown"] = {
            "base_score": round(base_score, 6),
            "keyword_boost": round(keyword_boost, 6),
            "transaction_boost": round(transaction_boost, 6),
            "country_boost": round(country_boost, 6),
            "amount_boost": round(amount_boost, 6),
            "policy_boost": round(policy_boost, 6),
        }
        scored_documents.append(reranked_doc)

    scored_documents.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
    return scored_documents[:top_k]
