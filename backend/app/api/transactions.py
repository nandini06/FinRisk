from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.repositories import (
    get_customer_by_customer_id,
    get_transaction_by_transaction_id,
)

router = APIRouter()


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _model_to_dict(model: Any) -> dict[str, Any]:
    return {
        column.name: _serialize_value(getattr(model, column.name))
        for column in model.__table__.columns
    }


@router.get("/transactions/{transaction_id}")
def get_transaction_details(transaction_id: str, db: Session = Depends(get_db)) -> dict[str, dict]:
    transaction = get_transaction_by_transaction_id(db, transaction_id)
    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{transaction_id}' not found.",
        )

    customer = get_customer_by_customer_id(db, transaction.customer_id)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{transaction.customer_id}' not found for transaction '{transaction_id}'.",
        )

    return {
        "transaction": _model_to_dict(transaction),
        "customer": _model_to_dict(customer),
    }
