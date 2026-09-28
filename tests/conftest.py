import os

# Tests run without the 300MB replay dataset, locally and in CI alike.
os.environ["VOLTGUARD_REPLAY_DATA"] = "tests/__no_dataset__.csv"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from api.main import app  # noqa: E402
from api.routes.predict import get_predictor  # noqa: E402
from voltguard.database import models  # noqa: E402
from voltguard.database.config import get_db  # noqa: E402

# Use a purely in-memory SQLite database for testing to ensure isolation
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Force load of all schemas before creating tables
    models.Base.metadata.create_all(bind=engine)
    yield
    models.Base.metadata.drop_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override FastAPI dependencies to securely route to the fake test db
app.dependency_overrides[get_db] = override_get_db

# The mock-fleet seed route is opt-in; the fleet tests rely on it.
os.environ.setdefault("VOLTGUARD_ENABLE_DEMO_ENDPOINTS", "1")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


class StubPredictor:
    """Deterministic stand-in so API tests don't need a trained model or SHAP."""

    def __init__(self, fault: int = 1, p_nominal: float = 0.2):
        self.fault, self.p_nominal = fault, p_nominal
        self.history: dict = {}

    def predict(self, vehicle_id, raw_data):
        if raw_data.get("torque") == 999:
            raise RuntimeError("boom")
        self.history[vehicle_id] = raw_data
        probs = {0: self.p_nominal, self.fault: 1 - self.p_nominal} if self.fault else {0: 1.0}
        return {
            "predicted_fault": self.fault,
            "fault_label": "Stator Overheat" if self.fault else "Nominal",
            "confidence": max(probs.values()),
            "probabilities": probs,
            "explanations": {"coolant": 0.5, "torque": -0.2},
            "input": raw_data,
        }


@pytest.fixture
def stub_predictor():
    stub = StubPredictor()
    app.dependency_overrides[get_predictor] = lambda: stub
    yield stub
    app.dependency_overrides.pop(get_predictor, None)


@pytest.fixture
def replay_dataset(tmp_path, monkeypatch):
    """A tiny PMSM-shaped replay CSV (2 drives x 40 rows) with a hot tail in drive 2."""
    import numpy as np
    import pandas as pd

    from voltguard.simulator import dataset

    rng = np.random.RandomState(0)
    n = 80
    temps = np.r_[np.full(40, 40.0), np.linspace(40, 120, 40)]
    df = pd.DataFrame(
        {
            "u_q": rng.normal(50, 5, n),
            "coolant": np.r_[np.full(40, 25.0), np.linspace(25, 80, 40)],
            "stator_winding": temps,
            "u_d": rng.normal(-20, 5, n),
            "stator_tooth": temps,
            "motor_speed": rng.normal(3000, 100, n),
            "i_d": rng.normal(-50, 5, n),
            "i_q": rng.normal(60, 5, n),
            "pm": temps,
            "stator_yoke": temps,
            "ambient": np.full(n, 22.0),
            "torque": rng.normal(50, 5, n),
            "profile_id": np.r_[np.full(40, 1), np.full(40, 2)],
        }
    )
    path = tmp_path / "replay.csv"
    df.to_csv(path, index=False)
    monkeypatch.setenv("VOLTGUARD_REPLAY_DATA", str(path))

    caches = [
        dataset.load_replay_data,
        dataset.stress_start_indices,
        dataset.feature_ranges,
        dataset.profile_summary,
    ]
    for fn in caches:
        fn.cache_clear()
    yield df
    for fn in caches:
        fn.cache_clear()
