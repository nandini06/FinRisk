from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.api.investigations import router
from backend.app.db.database import Base, get_db
from backend.app.db.models import Customer, InvestigationReport, Transaction
from backend.app.db.repositories import delete_legacy_reference_reports


def _app_and_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(Customer(customer_id="C-1", kyc_risk_score=0.0))
    session.add(Transaction(transaction_id="TXN-1", customer_id="C-1", amount=1))
    session.commit()
    app = FastAPI()
    app.include_router(router)

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    return app, session


def _legacy():
    return {
        "example_id": "INV-0001",
        "expected_risk_level": "LOW",
        "gold_explanation": "Reference only",
        "recommended_action": "None",
    }


def test_report_endpoint_404_when_only_reference_row_exists():
    app, session = _app_and_session()
    session.add(
        InvestigationReport(
            transaction_id="TXN-1", risk_level="LOW", report_json=_legacy(), created_at=datetime.utcnow()
        )
    )
    session.commit()
    response = TestClient(app).get("/reports/TXN-1")
    assert response.status_code == 404
    session.close()


def test_report_endpoint_skips_newer_reference_and_returns_genuine_report():
    app, session = _app_and_session()
    now = datetime.utcnow()
    session.add(
        InvestigationReport(
            transaction_id="TXN-1",
            report_json={"transaction_id": "TXN-1", "status": "supported"},
            created_at=now,
        )
    )
    session.add(
        InvestigationReport(
            transaction_id="TXN-1", report_json=_legacy(), created_at=now + timedelta(seconds=1)
        )
    )
    session.commit()
    response = TestClient(app).get("/reports/TXN-1")
    assert response.status_code == 200
    assert response.json()["status"] == "supported"
    session.close()


def test_cleanup_only_deletes_identifiable_reference_rows():
    _app, session = _app_and_session()
    legacy = InvestigationReport(transaction_id="TXN-1", report_json=_legacy())
    generated = InvestigationReport(
        transaction_id="TXN-1",
        report_json={"transaction_id": "TXN-1", "status": "supported", "trace": {}},
    )
    lookalike = InvestigationReport(
        transaction_id="TXN-1",
        report_json={"example_id": "INV-9999", "summary": "genuine old report"},
    )
    session.add_all([legacy, generated, lookalike])
    session.commit()
    assert delete_legacy_reference_reports(session) == 1
    remaining = session.query(InvestigationReport).all()
    assert {row.report_json.get("status", row.report_json.get("summary")) for row in remaining} == {
        "supported",
        "genuine old report",
    }
    session.close()
