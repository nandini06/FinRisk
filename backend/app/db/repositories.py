from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .models import Customer, HistoricalCase, InvestigationReport, Transaction


def is_legacy_reference_report_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"example_id", "expected_risk_level", "gold_explanation"}
    example_id = payload.get("example_id")
    return (
        required.issubset(payload)
        and isinstance(example_id, str)
        and example_id.startswith("INV-")
        and "transaction_id" not in payload
        and "trace" not in payload
        and "verification" not in payload
    )


def get_customer_by_customer_id(db: Session, customer_id: str) -> Customer | None:
    stmt = select(Customer).where(Customer.customer_id == customer_id)
    return db.scalar(stmt)


def get_transaction_by_transaction_id(db: Session, transaction_id: str) -> Transaction | None:
    stmt = select(Transaction).where(Transaction.transaction_id == transaction_id)
    return db.scalar(stmt)


def get_transaction_with_customer(
    db: Session, transaction_id: str
) -> tuple[Transaction, Customer] | None:
    transaction = get_transaction_by_transaction_id(db, transaction_id)
    if transaction is None:
        return None

    customer = get_customer_by_customer_id(db, transaction.customer_id)
    if customer is None:
        return None

    return transaction, customer


def get_historical_case_by_case_id(db: Session, case_id: str) -> HistoricalCase | None:
    stmt = select(HistoricalCase).where(HistoricalCase.case_id == case_id)
    return db.scalar(stmt)


def create_investigation_report(
    db: Session,
    transaction_id: str,
    report_json: dict,
    risk_level: str | None = None,
    confidence: float | None = None,
    summary: str | None = None,
) -> InvestigationReport:
    report = InvestigationReport(
        transaction_id=transaction_id,
        risk_level=risk_level,
        confidence=confidence,
        summary=summary,
        report_json=report_json,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_latest_report_by_transaction_id(
    db: Session, transaction_id: str
) -> InvestigationReport | None:
    stmt = (
        select(InvestigationReport)
        .where(InvestigationReport.transaction_id == transaction_id)
        .order_by(InvestigationReport.created_at.desc())
    )
    for report in db.scalars(stmt):
        if not is_legacy_reference_report_payload(report.report_json):
            return report
    return None


def get_legacy_reference_reports(db: Session) -> list[InvestigationReport]:
    return [
        report
        for report in db.scalars(select(InvestigationReport))
        if is_legacy_reference_report_payload(report.report_json)
    ]


def delete_legacy_reference_reports(db: Session) -> int:
    ids = [report.id for report in get_legacy_reference_reports(db)]
    if not ids:
        return 0
    result = db.execute(delete(InvestigationReport).where(InvestigationReport.id.in_(ids)))
    db.commit()
    return int(result.rowcount or 0)
