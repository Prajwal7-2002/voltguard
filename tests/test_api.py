import pytest
from pydantic import ValidationError

from api.schemas import SensorReading
from voltguard.database.models import PredictionLog, VehicleDB


def valid_sensor_reading() -> dict:
    return {
        "vehicle_id": "v1",
        "ambient": 25,
        "coolant": 35,
        "u_d": 10,
        "u_q": 20,
        "motor_speed": 1000,
        "torque": 5,
        "i_d": 2,
        "i_q": 3,
        "pm": 40,
        "stator_yoke": 45,
        "stator_tooth": 50,
        "stator_winding": 55,
    }


def test_sensor_reading_rejects_impossible_temperature():
    payload = valid_sensor_reading()
    payload["stator_winding"] = 300

    with pytest.raises(ValidationError):
        SensorReading.model_validate(payload)


def test_sensor_reading_rejects_empty_vehicle_id():
    payload = valid_sensor_reading()
    payload["vehicle_id"] = ""

    with pytest.raises(ValidationError):
        SensorReading.model_validate(payload)


def test_read_root(client):
    """Verify standard application heartbeat."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_database_seeding(client):
    """Verify the SQLite Database correctly injects 3 test fleets."""
    response = client.post("/vehicles/_seed_mock_fleet")
    assert response.status_code == 200
    assert "seeded" in response.json()["status"]


def test_fleet_overview(client):
    """Verify FastAPI correctly queries the SQL database through ORM."""
    # Ensure it's seeded first
    client.post("/vehicles/_seed_mock_fleet")

    response = client.get("/vehicles")
    assert response.status_code == 200

    payload = response.json()
    assert payload["total_vehicles"] == 3
    # v1 + v3 healthy, v2 at risk
    assert payload["at_risk_vehicles"] == 1


def test_single_vehicle_health(client):
    """Verify single ORM query resolves correctly via dynamic path routing."""
    response = client.get("/vehicles/v1/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["vehicle_id"] == "v1"
    assert payload["health_score"] >= 90.0


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_seed_endpoint_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("VOLTGUARD_ENABLE_DEMO_ENDPOINTS", raising=False)
    assert client.post("/vehicles/_seed_mock_fleet").status_code == 404


def test_predict_logs_and_updates_vehicle(client, stub_predictor, db_session):
    payload = valid_sensor_reading() | {"vehicle_id": "audit-1"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_fault"] == 1

    logs = db_session.query(PredictionLog).filter_by(vehicle_id="audit-1").all()
    assert len(logs) == 1
    assert logs[0].predicted_fault == 1
    vehicle = db_session.get(VehicleDB, "audit-1")
    assert vehicle.active_faults == [1]
    assert vehicle.health_score == pytest.approx(20.0)


def test_batch_failure_is_atomic(client, stub_predictor, db_session):
    good = valid_sensor_reading() | {"vehicle_id": "batch-1"}
    bad = good | {"torque": 999}
    response = client.post("/predict/batch", json={"readings": [good, bad]})
    assert response.status_code == 500
    assert "index 1" in response.json()["detail"]

    assert db_session.query(PredictionLog).filter_by(vehicle_id="batch-1").count() == 0


def test_batch_rejects_empty(client, stub_predictor):
    assert client.post("/predict/batch", json={"readings": []}).status_code == 422
