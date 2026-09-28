import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.routes import alerts, models, predict, vehicles
from voltguard.database import models as db_models
from voltguard.database.config import engine, get_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema bootstrap for local/dev. Production deployments should run migrations instead.
    db_models.Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="VoltGuard API",
    description="Predictive diagnostics and self-healing intelligence for EV Fleets.",
    version="1.0.0",
    lifespan=lifespan,
)

# Comma-separated list, e.g. "https://fleet.example.com,http://localhost:8501"
_cors_origins = [
    o.strip()
    for o in os.getenv("VOLTGUARD_CORS_ORIGINS", "http://localhost:8501").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the VoltGuard API"}


@app.get("/health")
def health(db: Session = Depends(get_db)):
    """Liveness/readiness probe: verifies the database is reachable."""
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


app.include_router(predict.router)
app.include_router(models.router)
app.include_router(alerts.router)
app.include_router(vehicles.router)
