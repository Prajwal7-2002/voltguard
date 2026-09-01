from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import predict, models, alerts, vehicles
from voltguard.database.config import engine, Base
from voltguard.database import models as db_models

# Ensure tables are built dynamically on start (avoids Alembic complexity for MVP)
db_models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="VoltGuard API",
    description="Predictive diagnostics and self-healing intelligence for EV Fleets.",
    version="1.0.0"
)

# CORS Rules
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the VoltGuard API"}

# Include Routers
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(alerts.router)
app.include_router(vehicles.router)
