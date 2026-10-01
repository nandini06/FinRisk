from __future__ import annotations

from copy import deepcopy
from typing import Any

ALLOWED_SOURCE_PREFIXES = {"policy", "case", "feature", "fact"}


def evidence_source_id(kind: str, identifier: str) -> str:
    return f"{kind}:{identifier}"


def _retrieval_id(row: dict[str, Any], kind: str, metadata_key: str) -> str:
    metadata = row.get("metadata") or {}
    identifier = str(metadata.get(metadata_key) or "").strip()
    if not identifier:
        raise RuntimeError(f"Retrieved {kind} evidence is missing {metadata_key}.")
    return evidence_source_id(kind, identifier)


def normalize_retrieval_rows(
    rows: list[dict[str, Any]], kind: str, metadata_key: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = deepcopy(row)
        item["source_id"] = _retrieval_id(item, kind, metadata_key)
        normalized.append(item)
    return normalized


def merge_evidence_rows(
    initial: list[dict[str, Any]], additional: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Preserve retrieval order and the union; scores from separate queries are not compared."""
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in initial + additional:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        output.append(row)
    return output


def build_evidence_registry(
    transaction: dict[str, Any],
    customer: dict[str, Any],
    features: dict[str, Any],
    anomaly_score: float,
    policy_evidence: list[dict[str, Any]],
    case_evidence: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}

    for row in policy_evidence:
        source_id = str(row["source_id"])
        registry[source_id] = {
            "source_id": source_id,
            "source_type": "policy",
            "text": str(row.get("text") or ""),
            "metadata": deepcopy(row.get("metadata") or {}),
            "provenance": "chroma:risk_policies",
        }
    for row in case_evidence:
        source_id = str(row["source_id"])
        registry[source_id] = {
            "source_id": source_id,
            "source_type": "case",
            "text": str(row.get("text") or ""),
            "metadata": deepcopy(row.get("metadata") or {}),
            "provenance": "chroma:historical_cases",
        }
    for name, value in features.items():
        source_id = evidence_source_id("feature", name)
        registry[source_id] = {
            "source_id": source_id,
            "source_type": "feature",
            "value": deepcopy(value),
            "provenance": "computed_by_application",
        }
        if name == "amount_vs_customer_avg":
            registry[source_id]["definition"] = (
                "Transaction amount divided by the dataset-provided customer "
                "monthly-average amount baseline."
            )
    registry["feature:anomaly_score"] = {
        "source_id": "feature:anomaly_score",
        "source_type": "feature",
        "value": anomaly_score,
        "provenance": "rule_based_application_score",
    }

    for scope, values in (("transaction", transaction), ("customer", customer)):
        for name, value in values.items():
            if name == "metadata_json" and isinstance(value, dict):
                for meta_name, meta_value in value.items():
                    source_id = evidence_source_id("fact", f"{scope}.{meta_name}")
                    registry[source_id] = {
                        "source_id": source_id,
                        "source_type": "fact",
                        "value": deepcopy(meta_value),
                        "provenance": f"{scope}.metadata_json.{meta_name}",
                    }
                continue
            source_id = evidence_source_id("fact", f"{scope}.{name}")
            registry[source_id] = {
                "source_id": source_id,
                "source_type": "fact",
                "value": deepcopy(value),
                "provenance": f"{scope}.{name}",
            }
            if scope == "customer" and name == "avg_monthly_transaction_amount":
                registry[source_id]["provenance"] = (
                    "data/customers.csv.avg_monthly_transaction_amount "
                    "(stored in customers.avg_transaction_amount)"
                )
                registry[source_id]["definition"] = (
                    "Precomputed synthetic customer monthly-average amount baseline; "
                    "the source generation window and aggregation code are not included."
                )
    return registry


def validate_report_citations(
    report: dict[str, Any], registry: dict[str, dict[str, Any]]
) -> None:
    def validate_source_ids(source_ids: Any, field_name: str) -> None:
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError(f"{field_name} must be a nonempty list")
        for source_id in source_ids:
            prefix = str(source_id).partition(":")[0]
            if prefix not in ALLOWED_SOURCE_PREFIXES:
                raise ValueError(f"{field_name} has invalid citation type: {source_id}")
            if source_id not in registry:
                raise ValueError(f"{field_name} cites unavailable source: {source_id}")

    validate_source_ids(report.get("risk_level_source_ids"), "risk_level_source_ids")
    validate_source_ids(
        report.get("recommended_action_source_ids"), "recommended_action_source_ids"
    )

    for index, factor in enumerate(report.get("risk_factors", [])):
        source_ids = factor.get("source_ids")
        validate_source_ids(source_ids, f"risk_factors[{index}].source_ids")

        cited_policy_ids = {
            str(source_id).partition(":")[2]
            for source_id in source_ids
            if str(source_id).startswith("policy:")
        }
        declared = factor.get("policy_severities") or []
        declared_policy_ids = {str(item.get("policy_id") or "") for item in declared}
        if declared_policy_ids != cited_policy_ids:
            raise ValueError(
                f"risk_factors[{index}].policy_severities must exactly match cited policies"
            )
        for item in declared:
            policy_id = str(item["policy_id"])
            actual = str(
                (registry[f"policy:{policy_id}"].get("metadata") or {}).get("severity") or ""
            ).upper()
            if item["severity"] != actual:
                raise ValueError(
                    f"risk_factors[{index}] reports {policy_id} severity as "
                    f"{item['severity']}, but evidence defines {actual or 'no severity'}"
                )

        if declared:
            rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            highest_policy_rank = max(rank[item["severity"]] for item in declared)
            if rank[factor["severity"]] > highest_policy_rank and not any(
                str(source_id).startswith(("fact:", "feature:")) for source_id in source_ids
            ):
                raise ValueError(
                    f"risk_factors[{index}] exceeds cited policy severity without fact/feature evidence"
                )

    for index, similar_case in enumerate(report.get("similar_cases", [])):
        source_id = evidence_source_id("case", str(similar_case.get("case_id") or ""))
        if source_id not in registry:
            raise ValueError(f"similar_cases[{index}] cites unavailable case: {source_id}")

    action = str(report.get("recommended_action") or "")
    action_basis = report.get("recommended_action_basis")
    action_source_ids = report.get("recommended_action_source_ids") or []
    if action_basis == "POLICY_REQUIRED":
        if not action.startswith("Policy-required:"):
            raise ValueError("A policy-required action must start with 'Policy-required:'")
        if not any(str(source_id).startswith("policy:") for source_id in action_source_ids):
            raise ValueError("A policy-required action must cite policy evidence")
    elif action_basis == "DISCRETIONARY_HUMAN_REVIEW" and not action.startswith(
        "Suggested for human review:"
    ):
        raise ValueError(
            "A discretionary action must start with 'Suggested for human review:'"
        )


def registry_records(registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(record) for record in registry.values()]
