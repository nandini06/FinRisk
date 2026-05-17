from __future__ import annotations

from typing import Any

from ..ml.anomaly_score import calculate_anomaly_score
from ..ml.features import extract_features


def run_anomaly_analysis(transaction: dict[str, Any], customer: dict[str, Any]) -> dict[str, Any]:
    features = extract_features(transaction=transaction, customer=customer)
    score_result = calculate_anomaly_score(features=features)

    return {
        "features": features,
        "anomaly_score": score_result["anomaly_score"],
        "top_contributing_features": score_result["top_contributing_features"],
    }
