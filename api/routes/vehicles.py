import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.schemas import FleetHealthResponse, VehicleHealthResponse

from voltguard.database.config import get_db
from voltguard.database.models import VehicleDB

router = APIRouter(prefix="/vehicles", tags=["Vehicles"])

@router.get("", response_model=FleetHealthResponse)
def get_fleet_overview(db: Session = Depends(get_db)):
    """Retrieve the aggregated health overview for the entire fleet natively from SQL."""
    vehicles_db = db.query(VehicleDB).all()
    vehicles = []
    at_risk = 0
    
    for v_db in vehicles_db:
        # Resolve time format safely
        last_run = v_db.last_reading_time.isoformat() if v_db.last_reading_time else datetime.now(timezone.utc).isoformat()
        
        v = VehicleHealthResponse(
            vehicle_id=v_db.id,
            health_score=v_db.health_score,
            active_faults=v_db.active_faults,
            last_reading_time=last_run
        )
        vehicles.append(v)
        
        if len(v_db.active_faults) > 0 or v_db.health_score < 80:
            at_risk += 1
            
    return FleetHealthResponse(
        total_vehicles=len(vehicles_db),
        healthy_vehicles=len(vehicles_db) - at_risk,
        at_risk_vehicles=at_risk,
        vehicles=vehicles
    )

@router.get("/{vehicle_id}/health", response_model=VehicleHealthResponse)
def get_vehicle_health(vehicle_id: str, db: Session = Depends(get_db)):
    """Retrieve detailed health information for a specific vehicle gracefully from Database."""
    v_db = db.query(VehicleDB).filter(VehicleDB.id == vehicle_id).first()
    if not v_db:
        raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found in database.")
        
    last_run = v_db.last_reading_time.isoformat() if v_db.last_reading_time else datetime.now(timezone.utc).isoformat()
    return VehicleHealthResponse(
        vehicle_id=v_db.id,
        health_score=v_db.health_score,
        active_faults=v_db.active_faults,
        last_reading_time=last_run
    )

def require_demo_endpoints() -> None:
    """Hide destructive demo routes unless VOLTGUARD_ENABLE_DEMO_ENDPOINTS=1."""
    if os.getenv("VOLTGUARD_ENABLE_DEMO_ENDPOINTS") != "1":
        raise HTTPException(status_code=404, detail="Not Found")


@router.post("/_seed_mock_fleet", dependencies=[Depends(require_demo_endpoints)])
def seed_mock_fleet(db: Session = Depends(get_db)):
    """Demo only: replace all vehicles with a fixed mock fleet."""
    # Delete old seeded anomalies if testing reset
    db.query(VehicleDB).delete()
    
    defaults = [
        VehicleDB(id="v1", health_score=92.5, active_faults=[]),
        VehicleDB(id="v2", health_score=65.0, active_faults=[1]),
        VehicleDB(id="v3", health_score=98.0, active_faults=[])
    ]
    db.add_all(defaults)
    db.commit()
    return {"status": "seeded 3 vehicles into SqliteDB permanently!"}
