from __future__ import annotations

from typing import Any

WEIGHTS = {
    "amount_vs_customer_avg": 0.30,
    "new_destination_country_flag": 0.20,
    "new_transaction_type_flag": 0.15,
    "kyc_risk_score": 0.15,
    "country_risk_score": 0.15,
    "unusual_channel_flag": 0.05,
}

# Transactions at or above 50x customer average are treated as max anomaly contribution.
AMOUNT_RATIO_MAX = 50.0

# Country risk in this dataset is typically 1-5, so scale to 0-1.
COUNTRY_RISK_MAX = 5.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y"}
    return False


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_feature_value(feature_name: str, raw_value: Any) -> float:
    if feature_name == "amount_vs_customer_avg":
        ratio = _to_float(raw_value, default=0.0)
        return _clamp_0_1(ratio / AMOUNT_RATIO_MAX)

    if feature_name in {"new_destination_country_flag", "new_transaction_type_flag", "unusual_channel_flag"}:
        return 1.0 if _to_bool(raw_value) else 0.0

    if feature_name == "country_risk_score":
        risk = _to_float(raw_value, default=0.0)
        return _clamp_0_1(risk / COUNTRY_RISK_MAX)

    if feature_name == "kyc_risk_score":
        return _clamp_0_1(_to_float(raw_value, default=0.0))

    return 0.0


def calculate_anomaly_score(features: dict[str, Any]) -> dict[str, Any]:
    contributions: list[dict[str, Any]] = []
    weighted_sum = 0.0

    for feature_name, weight in WEIGHTS.items():
        normalized_value = _normalize_feature_value(feature_name, features.get(feature_name))
        contribution = weight * normalized_value
        weighted_sum += contribution

        contributions.append(
            {
                "feature": feature_name,
                "value": features.get(feature_name),
                "normalized_value": round(normalized_value, 6),
                "weight": weight,
                "contribution": round(contribution, 6),
            }
        )

    anomaly_score = _clamp_0_1(weighted_sum)
    contributions.sort(key=lambda item: item["contribution"], reverse=True)

    top_contributing_features = [item for item in contributions if item["contribution"] > 0][:3]

    return {
        "anomaly_score": round(anomaly_score, 6),
        "top_contributing_features": top_contributing_features,
    }
