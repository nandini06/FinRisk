# FinRisk Investigator

An explainable financial-risk investigation backend built with:

- FastAPI
- PostgreSQL
- ChromaDB
- Gemini (`google-genai`)
- Deterministic anomaly scoring + agentic retrieval workflow

This project analyzes suspicious transactions, retrieves relevant policies and historical cases, generates a structured risk report, verifies evidence quality, and stores the final report.

## 1) Project Overview

The backend investigates a transaction by combining:

- Structured data from PostgreSQL (`customers`, `transactions`, `historical_cases`, `investigation_reports`)
- Semantic evidence retrieval from ChromaDB (`risk_policies`, `historical_cases`)
- LLM-driven planning/reporting/verification (Gemini)
- Rule-based feature extraction and weighted anomaly scoring

## 2) Architecture

Runtime flow:

1. Fetch transaction + customer from PostgreSQL
2. Extract anomaly features and compute anomaly score
3. Planner agent creates retrieval queries
4. Retrieve + rerank policies and similar cases from ChromaDB
5. Report agent generates structured investigation report
6. Verifier agent checks claim support and confidence
7. Optional one-time re-retrieval and report regeneration
8. Save final report in PostgreSQL and return JSON response

## 3) Why PostgreSQL + ChromaDB

- PostgreSQL stores transactional and report records with strong consistency.
- ChromaDB stores free-text evidence for semantic retrieval.
- This split keeps structured operations fast while enabling flexible text search.

## 4) Agentic Workflow

Implemented agents:

- `planner_agent`: builds targeted retrieval queries
- `policy_agent`: gets top policy evidence
- `case_agent`: gets top similar historical cases
- `anomaly_agent`: runs features + anomaly scoring
- `report_agent`: creates structured risk report JSON
- `verifier_agent`: validates support and suggests re-retrieval
- `workflow`: orchestrates end-to-end execution

## 5) Self-Corrective Retrieval

Verifier output can trigger one retry:

- `needs_more_evidence = true`
- use `suggested_query` for extra policy/case retrieval
- regenerate report once

Retry is intentionally capped to a single pass.

## 6) Explainability Design

Final report includes:

- `risk_level`, `confidence`, summary
- explicit `risk_factors`
- cited evidence
- similar historical cases
- recommended action

Risk factors are validated to include evidence citations.

## 7) Setup Instructions

## Prerequisites

- Python 3.11+
- PostgreSQL running on `localhost:5432`
- Gemini API key

## Environment

Create `.env` in repo root:

```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/finrisk
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
CHROMA_DB_PATH=./chroma_db
```

## Install dependencies

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

## Create database

```sql
CREATE DATABASE finrisk;
```

## Seed PostgreSQL

```powershell
python scripts/seed_postgres.py
```

## Seed ChromaDB

```powershell
python scripts/seed_chroma.py
```

## Run API

```powershell
python -m uvicorn backend.app.main:app --reload
```

Open docs:

- `http://127.0.0.1:8000/docs`

## 8) API Endpoints

## Health

- `GET /health`

Response:

```json
{"status":"ok"}
```

## Transactions

- `GET /transactions/{transaction_id}`

Returns:

```json
{
  "transaction": {},
  "customer": {}
}
```

## Investigate

- `POST /investigate/{transaction_id}`

Behavior:

- runs end-to-end investigation workflow
- returns final report JSON
- `404` if transaction is not found
- `500` if workflow fails

## Reports

- `GET /reports/{transaction_id}`

Returns latest saved report for that transaction from PostgreSQL.

## 9) Example Investigation Output

```json
{
  "transaction_id": "TXN-000002",
  "risk_level": "HIGH",
  "confidence": 0.87,
  "summary": "High-risk transaction based on amount, jurisdiction, and behavior deviation.",
  "risk_factors": [
    {
      "factor": "High-value cash payment",
      "severity": "HIGH",
      "explanation": "Amount exceeds high-value threshold.",
      "evidence": "AML-001"
    }
  ],
  "evidence": [],
  "similar_cases": [],
  "recommended_action": "Escalate for manual review."
}
```

## 10) Troubleshooting

- `uvicorn is not recognized`
  - Use `python -m uvicorn backend.app.main:app --reload`
  - Ensure your virtual environment is activated.

- `password authentication failed for user "postgres"`
  - Fix `DATABASE_URL` credentials in `.env`.
  - URL-encode password if it contains special characters.

- `{"detail":"Method Not Allowed"}`
  - `/investigate/{transaction_id}` requires `POST`, not `GET`.

- Empty or failed report generation
  - Check `GEMINI_API_KEY` and `GEMINI_MODEL`.
  - Verify internet/API access and account quota.

## 11) Current Project Structure

```text
backend/
  app/
    agents/
    api/
    db/
    llm/
    ml/
    rag/
    main.py
  requirements.txt
data/
scripts/
chroma_db/
```

## 12) Future Improvements

- add auth and audit trails
- add model-based anomaly detection
- add async/background investigation jobs
- add observability and evaluation metrics
- add frontend dashboard
