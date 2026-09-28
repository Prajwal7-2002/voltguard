# VoltGuard — EV Powertrain Predictive Diagnostics & Self-Healing System

## What Is This?

VoltGuard is a real-time machine learning system that gives **early warning of thermal stress in EV traction motors** from electrical and control telemetry. It classifies each sensor tick with XGBoost, explains predictions via SHAP, recommends a corrective action, and in simulation closes the loop by derating the motor. See the README for measured performance and limitations.

**Core problem it solves:** EVs fail catastrophically when thermal or electrical faults go undetected. VoltGuard makes EV powertrains self-healing.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost 2.0+ (CPU-optimized, ~5ms inference) |
| Explainability | SHAP 0.44+ TreeExplainer |
| Feature Engineering | scikit-learn 1.3+, Pandas 2.0+, NumPy |
| API | FastAPI 0.110+ + Uvicorn + Pydantic 2.0+ |
| Dashboard | Streamlit 1.30+ + Plotly 5.18+ |
| Database | SQLAlchemy 2.0+ ORM → SQLite (swappable to PostgreSQL) |
| Config | Pydantic Settings + PyYAML (`config/thresholds.yaml`) |
| Quality | Pytest 8.0+, Ruff 0.3+, mypy 1.8+ |
| Dataset | Paderborn University PMSM dataset (Kaggle) |

---

## Architecture

```
d:/cdre-e2w/
│
├── api/                        # FastAPI microservice
│   ├── main.py                 # App init, CORS, routing
│   ├── schemas.py              # Pydantic request/response models
│   └── routes/
│       ├── predict.py          # POST /predict, /predict/batch, /predict/forecast
│       ├── alerts.py           # WebSocket /ws/alerts (real-time streaming)
│       ├── models.py           # Model versioning endpoints
│       └── vehicles.py         # Fleet health endpoints
│
├── voltguard/                  # Core ML package
│   ├── core/
│   │   ├── logging.py          # Structured logging
│   │   └── events.py           # In-process EventBus (pub/sub)
│   ├── diagnostics/
│   │   ├── predictor.py        # FaultPredictor — model + SHAP inference
│   │   ├── trainer.py          # train_model() — XGBoost training pipeline
│   │   ├── evaluator.py        # evaluate_model() — metrics + plots
│   │   ├── registry.py         # ModelRegistry — versioned joblib artifacts
│   │   └── drift.py            # DriftDetector — z-score baseline comparison
│   ├── features/
│   │   └── engine.py           # engineer_features(), generate_fault_codes()
│   ├── root_cause/
│   │   └── analyzer.py         # resolve_fault() — root cause + spare parts
│   ├── self_heal/              # LEGACY 3-class battery code, unused (pending removal)
│   ├── simulator/
│   │   ├── fleet.py            # FleetSimulator — multi-vehicle orchestration
│   │   └── vehicle.py          # VehicleSimulator — CSV playback + sensors
│   ├── forecaster/
│   │   └── engine.py           # FaultForecaster — extrapolation early warning
│   └── database/
│       ├── config.py           # SQLAlchemy engine + SessionLocal
│       └── models.py           # ORM: VehicleDB, PredictionLog
│
├── config/
│   ├── settings.py             # Pydantic VoltGuardConfig (typed)
│   └── thresholds.yaml         # Domain thresholds (battery, motor, healing)
│
├── dashboard/                  # Streamlit UI (5 pages)
│   ├── app.py                  # Main page — model metrics, class distribution
│   └── pages/
│       ├── 01_live_monitor.py  # Real-time vehicle telemetry
│       ├── 02_fleet_overview.py
│       ├── 03_explainability.py  # SHAP feature importance
│       ├── 04_historical.py    # Time-series trend analysis
│       └── 05_model_performance.py  # ROC, confusion matrix, per-class F1
│
├── scripts/
│   ├── train.py                # CLI model training
│   ├── simulate.py             # Fleet simulation demo
│   ├── auto_retrain.py         # Automated retraining loop
│   └── trip_generator.py       # Synthetic data generation
│
├── data/
│   ├── raw/train.csv
│   ├── raw/train_trips.csv
│   └── comprehensive_fault_training_data.csv   # Kaggle PMSM dataset
│
├── models/
│   ├── latest.txt              # Pointer to active model version
│   └── v1/
│       ├── model.pkl           # XGBoost model (joblib)
│       ├── scaler.pkl          # StandardScaler
│       ├── metrics.json        # Accuracy, F1, CV scores
│       └── baseline.json       # Mean/std for drift detection
│
├── tests/                      # Pytest suite (API, simulator, MLOps)
├── notebooks/                  # EDA, experiments, SHAP analysis
├── Makefile                    # Task runner
└── pyproject.toml              # Poetry config (Python 3.10+)
```

---

## How It Works: The Full Pipeline

### 1. Data Ingestion
`VehicleSimulator` reads the Paderborn PMSM CSV row-by-row at 2Hz. It computes synthetic sensors: `battery_temp = ambient + (power_draw × 0.002)`, and simulates state-of-charge (SOC) drain.

**Raw sensor inputs:** `ambient`, `coolant`, `u_d`, `u_q`, `motor_speed`, `torque`, `i_d`, `i_q`

### 2. Feature Engineering (`voltguard/features/engine.py`)
Leak-proof features computed from upstream control signals only:

```python
thermal_delta       = coolant - ambient
dt_dt               = coolant.diff()          # Temperature change rate
acceleration_gradient = motor_speed.diff()
power_draw          = (u_d * i_d) + (u_q * i_q)
battery_temp        = ambient + min(abs(power_draw) * 0.002, 40.0)
```

Fault labels are generated from **separate** stator/thermal channels via quantile thresholds (94th–97th percentile) — the model never sees them directly.

### 3. Fault Classification (`voltguard/diagnostics/predictor.py`)
`FaultPredictor.predict()` maintains a 10-tick rolling history buffer per vehicle. It scales features, runs XGBoost inference, and generates SHAP explanations.

**5 Fault Classes:**
| Code | Label |
|---|---|
| 0 | Nominal |
| 1 | Stator Overheat |
| 2 | Battery Thermal Stress |
| 3 | Coolant System Failure |
| 4 | Inverter Over-Current |

**Sample prediction response:**
```json
{
  "predicted_fault": 1,
  "fault_label": "Stator Overheat",
  "confidence": 0.92,
  "probabilities": {"0": 0.08, "1": 0.92},
  "explanations": {"thermal_delta": 15.2, "coolant": 10.5}
}
```

### 4. Root Cause Analysis (`voltguard/root_cause/analyzer.py`)
`resolve_fault()` combines domain rules with SHAP top-features to identify the exact cause and recommend spare parts:

```python
# Example: Fault Code 1, high RPM + high coolant
root_cause = "Sustained High-RPM Stator Overheat"
action     = "Derate motor torque by 40%. Part: Stator winding coil assembly"
```

### 5. Closed-Loop Derate (`voltguard/simulator/vehicle.py`)
When a prediction is non-nominal, `resolve_fault()` returns a recommended action. `VehicleSimulator.apply_effect()` treats load-reducing actions (derate, limit, reduce, cut, throttle) as a command: for `healing.derate_ticks` ticks it scales motor speed by `1 - throttle_rpm_reduction` and torque and current by `1 - throttle_current_reduction`. Temperatures are replayed from the dataset, so the derate changes the model's inputs, not the physical thermal response.

Every API prediction is written to the `prediction_logs` table, which is the audit trail.

### 6. Fleet Orchestration (`voltguard/simulator/fleet.py`)
`FleetSimulator` runs N vehicles in parallel, calling the full pipeline each tick and returning structured results for the API and dashboard.

---

## Key Design Decisions

**XGBoost over LSTMs** — Edge devices require sub-10ms inference. XGBoost on CPU delivers ~5ms. LSTMs are too slow and power-hungry for real-time EV control.

**Leak-proof feature design** — Stator temperature labels (`stator_winding`, `stator_tooth`, `stator_yoke`, `pm`) are used only to generate fault codes, never as model inputs. This prevents label leakage.

**ACID database with dependency injection** — `SessionLocal` is injected into FastAPI routes so concurrent vehicle telemetry never causes race conditions. SQLite by default; swap to PostgreSQL via `DATABASE_URL` env var.

**Closed-loop actuation** — VoltGuard doesn't just alert. It predicts → explains → decides → acts → mutates state → re-predicts. Dashboards alone don't save motors.

---

## How to Run

```bash
# Install
pip install -e ".[dev]"

# Train model
make train
# or: python scripts/train.py --data data/raw/train_trips.csv

# Start API (http://localhost:8000)
make api
# or: uvicorn api.main:app --reload --port 8000

# Start dashboard (http://localhost:8501)
make dashboard
# or: streamlit run dashboard/app.py

# Run fleet simulation demo (3 vehicles, 20 ticks)
make demo
# or: python scripts/simulate.py --vehicles 3 --ticks 20 --speed 0.3

# Tests & quality
make test         # pytest --cov=voltguard
make lint         # ruff check
make typecheck    # mypy
```

---

## Configuration

**`config/thresholds.yaml`** — all tunable domain parameters:
```yaml
battery:
  temp_high: 60.0                  # Alert threshold (°C)
  charging_cycles_critical: 800    # Battery aging limit

motor:
  rpm_critical: 11000.0            # Overspeed limit

healing:
  cooling_rate: 2.0                # °C reduction per tick
  throttle_rpm_reduction: 0.3      # 30% RPM cut on throttle action
```

**Environment variables:**
```bash
DATABASE_URL=postgresql://user:pass@localhost/voltguard   # Swap SQLite → PostgreSQL
VOLTGUARD_CONFIG=/path/to/custom_thresholds.yaml
```

---

## Model Performance

Measured on **14 held-out drive profiles** (309k rows). The model never saw them during training. See `models/v2/metrics.json` and `notebooks/04_evaluation.ipynb`.

| Metric | Model v2 | Always "Nominal" |
|---|---|---|
| Macro F1 (evaluable classes) | 0.34 | 0.24 |
| Macro F1, 5-fold drive-level CV | 0.45 ± 0.02 | — |
| Accuracy | 0.90 | 0.96 |

- Recall is highest for Coolant System Failure (0.66) and Stator Overheat (0.37). Precision is low (0.10–0.16), so expect many false alarms.
- Inverter Over-Current is not learned (F1 0). Battery Thermal Stress occurs in only one drive and cannot be evaluated.
- Earlier ~99% figures came from a random row split. At 2 Hz this leaks near-duplicate rows between train and test.
- **Drift detection:** z-score comparison against `models/<version>/baseline.json` (mean/std per feature).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Single vehicle fault prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/predict/forecast` | Extrapolation-based early warning |
| GET | `/vehicles/{id}/health` | Fleet health score |
| GET | `/health` | Liveness/readiness probe |
| WS | `/ws/alerts` | Alert stream (currently echoes client messages) |