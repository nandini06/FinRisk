from fastapi import FastAPI

from .api.health import router as health_router
from .api.investigations import router as investigations_router
from .api.transactions import router as transactions_router

app = FastAPI(title="FinRisk Investigator API")
app.include_router(health_router)
app.include_router(transactions_router)
app.include_router(investigations_router)
