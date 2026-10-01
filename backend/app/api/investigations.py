from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..agents.workflow import run_investigation
from ..db.database import get_db
from ..db.repositories import get_latest_report_by_transaction_id

router = APIRouter()


@router.post("/investigate/{transaction_id}")
def investigate_transaction(transaction_id: str) -> dict:
    try:
        return run_investigation(transaction_id=transaction_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation workflow failed ({type(exc).__name__}).",
        ) from exc


@router.get("/reports/{transaction_id}")
def get_investigation_report(transaction_id: str, db: Session = Depends(get_db)) -> dict:
    report = get_latest_report_by_transaction_id(db=db, transaction_id=transaction_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No saved report found for transaction '{transaction_id}'.",
        )

    return report.report_json
