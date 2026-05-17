from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.rag.chroma_store import get_chroma_client, get_embedding_function

DATA_DIR = ROOT_DIR / "data"
RISK_POLICIES_COLLECTION = "risk_policies"
HISTORICAL_CASES_COLLECTION = "historical_cases"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def reset_collection(client, name: str, embedding_function):
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name, embedding_function=embedding_function)


def seed_risk_policies(client, embedding_function, df: pd.DataFrame) -> int:
    collection = reset_collection(client, RISK_POLICIES_COLLECTION, embedding_function)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for row in df.to_dict(orient="records"):
        policy_id = safe_str(row.get("policy_id"))
        title = safe_str(row.get("title"))
        policy_text = safe_str(row.get("policy_text"))
        risk_category = safe_str(row.get("risk_category"))
        severity = safe_str(row.get("severity"))

        ids.append(policy_id)
        documents.append(f"{title}. {policy_text}".strip())
        metadatas.append(
            {
                "policy_id": policy_id,
                "title": title,
                "risk_category": risk_category,
                "severity": severity,
            }
        )

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def seed_historical_cases(client, embedding_function, df: pd.DataFrame) -> int:
    collection = reset_collection(client, HISTORICAL_CASES_COLLECTION, embedding_function)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for row in df.to_dict(orient="records"):
        case_id = safe_str(row.get("case_id"))
        summary = safe_str(row.get("summary"))
        notes = safe_str(row.get("investigator_notes"))
        linked_pattern = safe_str(row.get("linked_transaction_pattern"))
        customer_segment = safe_str(row.get("customer_segment"))
        amount_bucket = safe_str(row.get("amount_bucket"))
        destination_country = safe_str(row.get("destination_country"))
        risk_factors = safe_str(row.get("risk_factors"))
        decision = safe_str(row.get("decision"))

        case_text = summary
        if notes:
            case_text = f"{summary} Investigator notes: {notes}"

        ids.append(case_id)
        documents.append(case_text)
        metadatas.append(
            {
                "case_id": case_id,
                "linked_transaction_pattern": linked_pattern,
                "customer_segment": customer_segment,
                "amount_bucket": amount_bucket,
                "destination_country": destination_country,
                "risk_factors": risk_factors,
                "decision": decision,
            }
        )

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def main() -> None:
    client = get_chroma_client()
    embedding_function = get_embedding_function(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    risk_policies_df = read_csv("risk_policies.csv")
    historical_cases_df = read_csv("historical_cases.csv")

    policy_count = seed_risk_policies(client, embedding_function, risk_policies_df)
    case_count = seed_historical_cases(client, embedding_function, historical_cases_df)

    print("Inserted documents:")
    print(f"risk_policies: {policy_count}")
    print(f"historical_cases: {case_count}")


if __name__ == "__main__":
    main()
