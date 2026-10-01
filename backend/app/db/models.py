from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Float, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from .database import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    customer_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255))
    country: Mapped[str | None] = mapped_column(String(64))
    kyc_level: Mapped[str | None] = mapped_column(String(32))
    kyc_risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    # Legacy storage name. seed_postgres maps this directly from the synthetic
    # dataset's avg_monthly_transaction_amount field.
    avg_transaction_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="customer")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("customers.customer_id"),
        nullable=False,
        index=True,
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    currency: Mapped[str | None] = mapped_column(String(16))
    transaction_type: Mapped[str | None] = mapped_column(String(64))
    channel: Mapped[str | None] = mapped_column(String(64))
    destination_country: Mapped[str | None] = mapped_column(String(64))
    origin_country: Mapped[str | None] = mapped_column(String(64))
    transaction_timestamp: Mapped[datetime | None] = mapped_column(DateTime)
    description: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    customer: Mapped["Customer"] = relationship(back_populates="transactions")
    reports: Mapped[list["InvestigationReport"]] = relationship(back_populates="transaction")


class HistoricalCase(Base):
    __tablename__ = "historical_cases"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    risk_category: Mapped[str | None] = mapped_column(String(128))
    decision: Mapped[str | None] = mapped_column(String(128))
    jurisdiction: Mapped[str | None] = mapped_column(String(64))
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class InvestigationReport(Base):
    __tablename__ = "investigation_reports"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("transactions.transaction_id"),
        nullable=False,
        index=True,
    )
    risk_level: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float)
    summary: Mapped[str | None] = mapped_column(Text)
    report_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    transaction: Mapped["Transaction"] = relationship(back_populates="reports")
