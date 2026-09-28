"""Replay-simulator sessions for the web dashboard.

A session owns N simulated vehicles replaying PMSM drives. Each tick runs the
real prediction pipeline and returns the prediction next to the proxy ground
truth, so the UI can show hits and false alarms. Sessions live in process
memory (demo scope) and nothing is written to the database.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.routes.predict import get_predictor
from voltguard.diagnostics.predictor import FaultPredictor
from voltguard.features.engine import FAULT_LABELS
from voltguard.root_cause.analyzer import resolve_fault
from voltguard.simulator.dataset import load_replay_data
from voltguard.simulator.vehicle import VehicleSimulator

router = APIRouter(prefix="/simulator", tags=["Simulator"])

MAX_SESSIONS = 20
TOP_DRIVERS = 8
READING_FIELDS = (
    "ambient",
    "coolant",
    "motor_speed",
    "torque",
    "i_d",
    "i_q",
    "stator_winding",
    "pm",
    "battery_temp",
    "soc",
    "profile_id",
)


class SessionCreate(BaseModel):
    vehicles: int = Field(1, ge=1, le=12)
    auto_derate: bool = Field(True, description="Apply load-reducing actions to the replay.")


class TickRequest(BaseModel):
    steps: int = Field(1, ge=1, le=200)


class _Session:
    def __init__(self, session_id: str, vehicles: int, auto_derate: bool):
        self.id = session_id
        self.auto_derate = auto_derate
        self.tick = 0
        self.vehicles = [VehicleSimulator(f"{session_id}-v{i + 1:03d}") for i in range(vehicles)]
        self.lock = threading.Lock()


_sessions: OrderedDict[str, _Session] = OrderedDict()
_sessions_lock = threading.Lock()


def _drop_history(predictor: FaultPredictor, session: _Session) -> None:
    for v in session.vehicles:
        predictor.history.pop(v.vehicle_id, None)


def _get_session(session_id: str) -> _Session:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired.")
        _sessions.move_to_end(session_id)
        return session


def _display_id(session: _Session, vehicle_id: str) -> str:
    return vehicle_id.removeprefix(f"{session.id}-")


@router.post("/sessions", status_code=201)
def create_session(body: SessionCreate, predictor: FaultPredictor = Depends(get_predictor)):
    session = _Session(uuid.uuid4().hex[:12], body.vehicles, body.auto_derate)
    with _sessions_lock:
        _sessions[session.id] = session
        while len(_sessions) > MAX_SESSIONS:
            _, evicted = _sessions.popitem(last=False)
            _drop_history(predictor, evicted)
    return {
        "session_id": session.id,
        "vehicles": [_display_id(session, v.vehicle_id) for v in session.vehicles],
        "auto_derate": session.auto_derate,
        "dataset_available": load_replay_data() is not None,
    }


@router.post("/sessions/{session_id}/tick")
def tick_session(
    session_id: str, body: TickRequest, predictor: FaultPredictor = Depends(get_predictor)
):
    session = _get_session(session_id)
    frames = []
    with session.lock:
        for _ in range(body.steps):
            session.tick += 1
            for vehicle in session.vehicles:
                frames.append(_step(session, vehicle, predictor))
    return {"session_id": session.id, "tick": session.tick, "frames": frames}


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, predictor: FaultPredictor = Depends(get_predictor)):
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is not None:
        _drop_history(predictor, session)


def _step(session: _Session, vehicle: VehicleSimulator, predictor: FaultPredictor) -> dict:
    reading = vehicle.tick()
    model_input = {
        k: v for k, v in reading.items() if k not in ("vehicle_id", "true_fault", "derated")
    }
    pred = predictor.predict(vehicle.vehicle_id, model_input)

    fault = pred["predicted_fault"]
    root_cause, action, derate_applied = None, None, False
    if fault != 0:
        root_cause, action = resolve_fault(fault, pred["explanations"], model_input)
        if session.auto_derate:
            derate_applied = vehicle.apply_effect(action)

    drivers = sorted(pred["explanations"].items(), key=lambda kv: abs(kv[1]), reverse=True)
    true_fault = reading["true_fault"]
    return {
        "tick": session.tick,
        "vehicle_id": _display_id(session, vehicle.vehicle_id),
        "reading": {k: reading[k] for k in READING_FIELDS},
        "derated": reading["derated"],
        "predicted_fault": fault,
        "predicted_label": FAULT_LABELS[fault],
        "fault_score": round(1.0 - float(pred["probabilities"].get(0, 0.0)), 4),
        "probabilities": {
            FAULT_LABELS[int(k)]: round(float(p), 4) for k, p in pred["probabilities"].items()
        },
        "top_drivers": [{"feature": f, "shap": round(v, 4)} for f, v in drivers[:TOP_DRIVERS]],
        "true_fault": true_fault,
        "true_label": FAULT_LABELS[true_fault] if true_fault is not None else None,
        "root_cause": root_cause,
        "action": action,
        "derate_applied": derate_applied,
    }
