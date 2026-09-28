"""Endpoints backing the Next.js dashboard."""

SCENARIO = {
    "ambient": 25,
    "coolant": 60,
    "u_d": -50,
    "u_q": 80,
    "motor_speed": 4000,
    "torque": 100,
    "i_d": -120,
    "i_q": 150,
}


def test_simulator_session_lifecycle(client, stub_predictor):
    created = client.post("/simulator/sessions", json={"vehicles": 2})
    assert created.status_code == 201
    body = created.json()
    assert body["vehicles"] == ["v001", "v002"]
    assert body["dataset_available"] is False

    ticked = client.post(f"/simulator/sessions/{body['session_id']}/tick", json={"steps": 3})
    assert ticked.status_code == 200
    frames = ticked.json()["frames"]
    assert len(frames) == 6  # 3 steps x 2 vehicles
    frame = frames[0]
    assert frame["predicted_label"] == "Stator Overheat"
    assert frame["fault_score"] == 0.8
    assert frame["true_fault"] is None  # no dataset -> no ground truth
    assert frame["top_drivers"][0] == {"feature": "coolant", "shap": 0.5}
    assert frame["derate_applied"] is True  # stator actions reduce load
    assert frames[-1]["derated"] is True

    assert client.delete(f"/simulator/sessions/{body['session_id']}").status_code == 204
    assert not stub_predictor.history  # per-vehicle buffers released
    missing = client.post(f"/simulator/sessions/{body['session_id']}/tick", json={"steps": 1})
    assert missing.status_code == 404


def test_simulator_without_auto_derate(client, stub_predictor):
    sid = client.post("/simulator/sessions", json={"auto_derate": False}).json()["session_id"]
    frames = client.post(f"/simulator/sessions/{sid}/tick", json={"steps": 2}).json()["frames"]
    assert not any(f["derated"] or f["derate_applied"] for f in frames)
    assert frames[0]["action"]  # still recommended, just not applied


def test_simulator_rejects_bad_sizes(client, stub_predictor):
    assert client.post("/simulator/sessions", json={"vehicles": 50}).status_code == 422


def test_explain_is_stateless(client, stub_predictor):
    response = client.post("/predict/explain", json=SCENARIO)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_fault"] == 1
    assert body["root_cause"] and body["action"]
    assert stub_predictor.history == {}


def test_data_summary_without_dataset(client):
    response = client.get("/data/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_available"] is False
    assert body["profiles"] == []
    assert set(body["feature_ranges"]) == set(SCENARIO)
    assert body["profiles_per_class"]["2"] == 1


def test_data_summary_with_dataset(client, replay_dataset):
    body = client.get("/data/summary").json()
    assert body["dataset_available"] is True
    assert [p["profile_id"] for p in body["profiles"]] == [1, 2]
    assert body["profiles"][0]["fault_rate"] == 0.0
    assert body["profiles"][1]["fault_rate"] > 0


def test_simulator_reports_ground_truth(client, stub_predictor, replay_dataset):
    sid = client.post("/simulator/sessions", json={}).json()["session_id"]
    frames = client.post(f"/simulator/sessions/{sid}/tick", json={"steps": 5}).json()["frames"]
    assert all(isinstance(f["true_fault"], int) for f in frames)
    assert all(f["true_label"] for f in frames)
