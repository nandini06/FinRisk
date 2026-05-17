from __future__ import annotations

from typing import Any

HIGH_VALUE_THRESHOLD = 5_000_000.0
DEFAULT_COUNTRY_RISK = 1.0
HIGH_RISK_COUNTRIES = {"iran", "north korea", "syria", "russia"}
UNUSUAL_CHANNELS = {"api", "advisor"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return default


def _normalized_set(raw_value: Any) -> set[str]:
    if raw_value is None:
        return set()
    if isinstance(raw_value, list):
        values = raw_value
    else:
        text = str(raw_value).replace(",", "|")
        values = text.split("|")
    return {value.strip().lower() for value in values if value and value.strip()}


def extract_features(transaction: dict[str, Any], customer: dict[str, Any]) -> dict[str, Any]:
    metadata = transaction.get("metadata_json") or {}

    amount = _to_float(transaction.get("amount"))
    customer_avg = _to_float(
        customer.get("avg_monthly_transaction_amount", customer.get("avg_transaction_amount")),
        default=0.0,
    )
    if customer_avg > 0:
        amount_vs_customer_avg = round(amount / customer_avg, 4)
    else:
        amount_vs_customer_avg = _to_float(metadata.get("amount_vs_customer_avg"), default=0.0)

    destination_country = str(transaction.get("destination_country") or "").strip().lower()
    transaction_type = str(transaction.get("transaction_type") or "").strip().lower()
    channel = str(transaction.get("channel") or "").strip().lower()

    usual_countries = _normalized_set(customer.get("usual_countries"))
    usual_transaction_types = _normalized_set(customer.get("usual_transaction_types"))

    explicit_new_country = metadata.get("new_destination_country_flag")
    if explicit_new_country is not None:
        new_destination_country_flag = _to_bool(explicit_new_country)
    else:
        new_destination_country_flag = bool(destination_country) and destination_country not in usual_countries

    explicit_new_type = metadata.get("new_transaction_type_flag")
    if explicit_new_type is not None:
        new_transaction_type_flag = _to_bool(explicit_new_type)
    else:
        new_transaction_type_flag = bool(transaction_type) and transaction_type not in usual_transaction_types

    kyc_risk_score = _to_float(customer.get("kyc_risk_score"), default=0.0)

    country_risk_score = _to_float(metadata.get("country_risk_score"), default=-1.0)
    if country_risk_score < 0:
        country_risk_score = 5.0 if destination_country in HIGH_RISK_COUNTRIES else DEFAULT_COUNTRY_RISK

    high_value_flag = amount >= HIGH_VALUE_THRESHOLD
    unusual_channel_flag = channel in UNUSUAL_CHANNELS

    return {
        "amount_vs_customer_avg": amount_vs_customer_avg,
        "new_destination_country_flag": new_destination_country_flag,
        "new_transaction_type_flag": new_transaction_type_flag,
        "kyc_risk_score": kyc_risk_score,
        "country_risk_score": country_risk_score,
        "high_value_flag": high_value_flag,
        "unusual_channel_flag": unusual_channel_flag,
    }
