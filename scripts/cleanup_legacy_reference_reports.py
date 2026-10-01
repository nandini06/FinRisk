from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.database import SessionLocal, engine  # noqa: E402
from backend.app.db.repositories import (  # noqa: E402
    delete_legacy_reference_reports,
    get_legacy_reference_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or remove only identifiable seeded reference-report rows."
    )
    parser.add_argument("--apply", action="store_true", help="Perform deletion (default is dry-run).")
    parser.add_argument("--backup", type=Path, help="Required JSON backup path with --apply.")
    parser.add_argument(
        "--confirm-database",
        help="Required with --apply; must exactly match the configured database name.",
    )
    return parser.parse_args()


def _backup_rows(path: Path, rows: list) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database": engine.url.database,
        "rows": [
            {
                "id": row.id,
                "transaction_id": row.transaction_id,
                "risk_level": row.risk_level,
                "confidence": row.confidence,
                "summary": row.summary,
                "report_json": row.report_json,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    with SessionLocal() as db:
        rows = get_legacy_reference_reports(db)
        preview = [f"id={row.id} transaction_id={row.transaction_id}" for row in rows[:20]]
        print(f"Legacy seeded reference reports found: {len(rows)}")
        for line in preview:
            print(line)
        if len(rows) > len(preview):
            print(f"... and {len(rows) - len(preview)} more")

        if not args.apply:
            print("Dry run only; no rows changed.")
            return

        host = (engine.url.host or "").lower()
        database = engine.url.database or ""
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise SystemExit("Apply is restricted to a configured local database host.")
        if not args.backup:
            raise SystemExit("--backup PATH is required with --apply.")
        if args.confirm_database != database:
            raise SystemExit(f"--confirm-database must exactly match {database!r}.")

        _backup_rows(args.backup.resolve(), rows)
        deleted = delete_legacy_reference_reports(db)
        print(f"Backup written to: {args.backup.resolve()}")
        print(f"Deleted legacy reference reports: {deleted}")


if __name__ == "__main__":
    main()
