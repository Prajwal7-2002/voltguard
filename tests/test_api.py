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
