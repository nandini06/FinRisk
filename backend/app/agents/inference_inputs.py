from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any

# These are deliberately flat allowlists. Unknown or nested fields never flow into prompts.
TRANSACTION_FIELDS = {
    "transaction_id",
    "customer_id",
    "amount",
    "currency",
    "transaction_type",
    "channel",
    "destination_country",
    "origin_country",
    "transaction_timestamp",
    "description",
}
TRANSACTION_METADATA_FIELDS = {
    "merchant_category",
    "country_risk_score",
    "new_destination_country_flag",
    "new_transaction_type_flag",
}
CUSTOMER_FIELDS = {
    "customer_id",
    "country",
    "kyc_level",
    "kyc_risk_score",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return deepcopy(value)


def _allowlisted(source: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    return {name: _json_value(source[name]) for name in fields if name in source}


def build_inference_inputs(
    transaction: dict[str, Any], customer: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return new, prompt-safe records without mutating the source records."""
    clean_transaction = _allowlisted(transaction, TRANSACTION_FIELDS)
    metadata = transaction.get("metadata_json")
    if isinstance(metadata, dict):
        allowed_metadata = _allowlisted(metadata, TRANSACTION_METADATA_FIELDS)
        if allowed_metadata:
            clean_transaction["metadata_json"] = allowed_metadata

    clean_customer = _allowlisted(customer, CUSTOMER_FIELDS)
    # The database column has a legacy generic name, but seed_postgres maps it
    # directly from data/customers.csv.avg_monthly_transaction_amount. Expose
    # the dataset meaning to inference rather than inviting the model to treat
    # it as an average of individual transactions.
    monthly_average = customer.get(
        "avg_monthly_transaction_amount", customer.get("avg_transaction_amount")
    )
    if monthly_average is not None:
        clean_customer["avg_monthly_transaction_amount"] = _json_value(monthly_average)
    return clean_transaction, clean_customer
