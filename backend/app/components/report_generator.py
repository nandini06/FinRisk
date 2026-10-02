from __future__ import annotations

import json
from typing import Any

from ..llm.gemini_client import GeminiProviderError, generate_json
from ..schemas.investigation import ReportOutput
from .evidence import validate_report_citations


def _validate_report(
    payload: dict[str, Any],
    transaction_id: str,
    evidence_registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report = ReportOutput.model_validate(payload).model_dump()
    if report["transaction_id"] != transaction_id:
        raise ValueError(
            f"transaction_id must be {transaction_id!r}, got {report['transaction_id']!r}"
        )
    validate_report_citations(report, evidence_registry)
    return report


def generate_investigation_report(
    transaction: dict[str, Any],
    customer: dict[str, Any],
    features: dict[str, Any],
    anomaly_score: float,
    evidence_registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    transaction_id = str(transaction.get("transaction_id") or "")
    prompt = f"""
You are a financial risk investigator. Create a concise evidence-grounded report.

Return JSON matching this contract:
- transaction_id: exactly {transaction_id}
- risk_level: LOW, MEDIUM, or HIGH; this is the model's overall evidence-based assessment,
  not a severity copied from one policy
- risk_level_rationale: explicit explanation of the overall assessment
- risk_level_source_ids: nonempty source IDs supporting that assessment
- confidence: finite number from 0 to 1 (an uncalibrated model assessment)
- summary: string
- risk_factors: array of objects with factor, severity, policy_severities,
  severity_rationale, explanation, source_ids
- similar_cases: array of objects with case_id, similarity_reason, decision
- recommended_action: string
- recommended_action_basis: POLICY_REQUIRED or DISCRETIONARY_HUMAN_REVIEW
- recommended_action_source_ids: nonempty source IDs supporting the action
- recommended_action_rationale: explanation of why the action is required or suggested

Every risk factor needs a nonempty source_ids list containing only IDs from the registry.
Similar case IDs must exist as case:<case_id> in the registry. A source existing does not
automatically support a claim; use its actual content. Do not reproduce an evidence list.

Severity and action rules:
- A risk factor's severity and the top-level risk_level are model assessments. Do not say a
  policy defines, assigns, or requires that model-assessed level unless the cited policy actually
  has that severity in its metadata.
- policy_severities must list every cited policy ID with exactly the severity found in that
  policy's metadata. Use an empty list when the factor cites no policy.
- If a factor severity is higher than every cited policy severity, explain the elevation in
  severity_rationale using cited fact or feature evidence. Do not attribute the elevation to policy.
- Do not choose HIGH or MEDIUM merely to match an expected/reference label; none is supplied.
- Use POLICY_REQUIRED only when the cited policy text explicitly requires the complete action.
  Prefix such text with "Policy-required:" and cite the supporting policy.
- If any material part of an action is a prudent recommendation rather than an explicit policy
  requirement, use DISCRETIONARY_HUMAN_REVIEW, prefix the text with
  "Suggested for human review:", and do not imply that holding, EDD, escalation, or documentation
  is policy-mandated without policy text that says so.

Baseline and geography rules:
- customer.avg_monthly_transaction_amount is the dataset-provided customer monthly-average
  amount baseline. It is not an average individual transaction amount. Its source generation
  window and aggregation code are unavailable, so describe ratios as comparisons with the
  provided synthetic monthly baseline, not as independently reconstructed customer history.
- amount_vs_customer_avg is transaction amount divided by that provided monthly baseline.
- Apply AML-003's 10x comparison only when a positive, explicitly monthly baseline is available.
  If the available baseline has a different or unknown meaning, keep the observed deviation as
  a factual finding but state that AML-003's threshold cannot be confirmed.
- customer.country and transaction.origin_country are different facts. When mentioning both,
  explicitly label the customer country and the transaction origin country; never combine them
  with a slash or imply that the customer has two origin countries.

transaction:
{json.dumps(transaction, indent=2, default=str)}
customer:
{json.dumps(customer, indent=2, default=str)}
computed_features:
{json.dumps(features, indent=2, default=str)}
rule_based_anomaly_score:
{json.dumps(anomaly_score)}
evidence_registry:
{json.dumps(evidence_registry, indent=2, default=str)}
""".strip()

    error = ""
    for attempt in range(2):
        current_prompt = prompt
        if attempt:
            current_prompt += (
                "\n\nThe previous output failed schema or citation validation. Return a corrected "
                f"report using the same inputs. Validation feedback: {error}"
            )
        try:
            payload = generate_json(current_prompt, response_schema=ReportOutput)
            return _validate_report(payload, transaction_id, evidence_registry)
        except GeminiProviderError as exc:
            raise RuntimeError("Report provider request failed.") from exc
        except Exception as exc:
            error = str(exc)[:700]
    raise RuntimeError(f"Report output validation failed after one repair: {error}")
