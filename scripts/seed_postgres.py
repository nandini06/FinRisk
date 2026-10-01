from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, func, select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.database import Base, SessionLocal, engine  # noqa: E402
from backend.app.db.models import (  # noqa: E402
    Customer,
    HistoricalCase,
    InvestigationReport,
    Transaction,
)

DATA_DIR = ROOT_DIR / "data"


def read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    return pd.read_csv(path)


def to_none(value):
    if pd.isna(value):
        return None
    return value


def seed_customers(db, df: pd.DataFrame) -> int:
    rows: list[Customer] = []
    for row in df.to_dict(orient="records"):
        usual_countries = str(to_none(row.get("usual_countries")) or "")
        country = usual_countries.split("|")[0] if usual_countries else None
        rows.append(
            Customer(
                customer_id=str(row["customer_id"]),
                country=country,
                kyc_level=to_none(row.get("customer_segment")),
                kyc_risk_score=float(to_none(row.get("kyc_risk_score")) or 0.0),
                avg_transaction_amount=to_none(row.get("avg_monthly_transaction_amount")),
            )
        )

    db.add_all(rows)
    return len(rows)


def seed_transactions(db, df: pd.DataFrame) -> int:
    rows: list[Transaction] = []
    for row in df.to_dict(orient="records"):
        metadata_json = {
            "merchant_category": to_none(row.get("merchant_category")),
            "country_risk_score": to_none(row.get("country_risk_score")),
            "amount_vs_customer_avg": to_none(row.get("amount_vs_customer_avg")),
            "new_destination_country_flag": to_none(row.get("new_destination_country_flag")),
            "new_transaction_type_flag": to_none(row.get("new_transaction_type_flag")),
        }
        rows.append(
            Transaction(
                transaction_id=str(row["transaction_id"]),
                customer_id=str(row["customer_id"]),
                amount=row["amount"],
                currency=to_none(row.get("currency")),
                transaction_type=to_none(row.get("transaction_type")),
                channel=to_none(row.get("channel")),
                destination_country=to_none(row.get("destination_country")),
                origin_country=to_none(row.get("source_country")),
                transaction_timestamp=pd.to_datetime(to_none(row.get("timestamp"))),
                metadata_json=metadata_json,
            )
        )

    db.add_all(rows)
    return len(rows)


def seed_historical_cases(db, df: pd.DataFrame) -> int:
    rows: list[HistoricalCase] = []
    for row in df.to_dict(orient="records"):
        rows.append(
            HistoricalCase(
                case_id=str(row["case_id"]),
                risk_category=to_none(row.get("risk_factors")),
                decision=to_none(row.get("decision")),
                jurisdiction=to_none(row.get("destination_country")),
                summary=str(row["summary"]),
            )
        )

    db.add_all(rows)
    return len(rows)


def print_table_counts(db) -> None:
    tables = [
        ("customers", Customer),
        ("transactions", Transaction),
        ("historical_cases", HistoricalCase),
        ("investigation_reports", InvestigationReport),
    ]
    for table_name, model in tables:
        count = db.scalar(select(func.count()).select_from(model))
        print(f"{table_name}: {count}")


def main() -> None:
    customers_df = read_csv("customers.csv")
    transactions_df = read_csv("transactions.csv")
    historical_cases_df = read_csv("historical_cases.csv")
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        db.execute(delete(InvestigationReport))
        db.execute(delete(Transaction))
        db.execute(delete(HistoricalCase))
        db.execute(delete(Customer))
        db.commit()

        customer_count = seed_customers(db, customers_df)
        transaction_count = seed_transactions(db, transactions_df)
        historical_case_count = seed_historical_cases(db, historical_cases_df)
        db.commit()

        print("Inserted rows:")
        print(f"customers: {customer_count}")
        print(f"transactions: {transaction_count}")
        print(f"historical_cases: {historical_case_count}")
        print("investigation_reports: 0 (reference answers are evaluation-only)")
        print("\nRow counts in database:")
        print_table_counts(db)


if __name__ == "__main__":
    main()
