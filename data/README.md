# Synthetic Financial Risk Investigation Dataset

This dataset is fully synthetic and intended for an Agentic / Self-Corrective / Explainable RAG project.

Files:
- customers.csv: customer profile and behavior baselines
- transactions.csv: labeled financial transactions with engineered risk features
- risk_policies.csv: unstructured policy text for RAG ingestion
- historical_cases.csv: synthetic investigator case summaries for similar-case retrieval
- investigation_examples.csv: gold-style explanations for evaluation and prompt testing

Suggested use:
1. Store customers and transactions in PostgreSQL.
2. Ingest risk_policies and historical_cases into Qdrant.
3. Train anomaly/classification models on transactions.csv.
4. Use investigation_examples.csv for report generation evaluation.

Labels:
- normal: low-risk baseline transaction
- review: medium risk requiring more information
- suspicious: high risk requiring manual review/SAR-style escalation
