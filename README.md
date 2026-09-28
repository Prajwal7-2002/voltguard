# ⚡ VoltGuard: EV Motor Thermal-Stress Early Warning

VoltGuard gives early warning of thermal stress in an EV traction motor using only
electrical and control telemetry. It uses the **Paderborn University PMSM dataset**
(1.33M rows sampled at 2 Hz across 69 drive profiles), classifies each tick with
XGBoost, explains the result with SHAP, recommends a corrective action, and closes the
loop in a simulator by derating the motor.

It ships as a FastAPI service with a prediction audit log, a Streamlit dashboard,
a versioned model registry, feature-drift checks, Docker, and CI.

---

## Results: read these before anything else

All metrics come from **whole drive profiles the model never saw during training**
(`models/v2/metrics.json`). They sit next to a baseline that always predicts "Nominal",
because about 91% of rows are Nominal.

| Held-out drives (14 profiles, 309k rows) | Model v2 | Always "Nominal" |
|---|---|---|
| Macro F1 (evaluable classes) | **0.34** | 0.24 |
| Macro F1, 5-fold drive-level CV | **0.45 ± 0.02** | — |
| Accuracy | 0.90 | 0.96 |

| Class | Precision | Recall | F1 | Drives containing it |
|---|---|---|---|---|
| Nominal | 0.99 | 0.92 | 0.95 | 69 |
| Stator Overheat | 0.16 | 0.37 | 0.22 | 39 |
| Coolant System Failure | 0.10 | 0.66 | 0.17 | 25 |
| Inverter Over-Current | 0.00 | 0.00 | 0.00 | 15 |
| Battery Thermal Stress | not evaluable | | | 1 |

**How to read this.** The model is recall-oriented because of class-balanced training.
It catches about two-thirds of coolant-stress ticks and a third of stator-overheat ticks
on unseen drives, at the cost of many false alarms, so its accuracy is *below* the
trivial baseline. It is an early-warning signal to combine with other checks, not a
standalone fault detector.

### Why earlier versions reported ~99%
`models/v1` was evaluated on a **random row split**. At 2 Hz, neighbouring rows are
near-duplicates, so every test drive also appeared in training, and v1 scored
98.8% accuracy / 0.93 macro F1. When v1's training setup is re-run with whole drives held
out, it scores 88–93% accuracy and 0.25–0.33 macro F1 across five splits. The always-Nominal
baseline scores 88–93% and 0.19 on the same splits. v1 is kept for comparison; its
`metrics.json` records the leaky split.

### Limitations
- **The labels are proxies.** The dataset has no fault annotations. A "fault" means a
  hidden thermal channel (stator winding, tooth, yoke or magnet temperature, none of
  which are model inputs) is in its top 3–6%. See `generate_fault_codes` in
  [voltguard/features/engine.py](voltguard/features/engine.py).
- **"Battery Thermal Stress" occurs in only one drive**, so it can't be tested on an
  unseen drive. `battery_temp` is also a synthetic feature derived from electrical power.
- **Scores vary across drives.** With 14 test drives, treat the CV standard deviation as
  the error bar.
- **The simulator replays recorded temperatures**, so a derate changes the model's
  inputs but not the physical thermal response.

---

## Architecture

```text
telemetry ──► FastAPI /predict ──► FaultPredictor (features → XGBoost → SHAP)
                  │                        │
                  │                        └─► root_cause.resolve_fault → recommended action
                  ▼
          PredictionLog + VehicleDB (SQLAlchemy; SQLite locally, PostgreSQL via DATABASE_URL)

simulator: VehicleSimulator replays PMSM drives ─► predict ─► action ─► derate speed/torque/current
```

| Path | Responsibility |
|---|---|
| `api/` | FastAPI app: prediction, batch, forecast, fleet health, model metadata, `/health` |
| `voltguard/features/` | Feature engineering and proxy label generation |
| `voltguard/diagnostics/` | Training (`trainer.py`), evaluation, registry, predictor, drift |
| `voltguard/root_cause/` | Maps fault class and SHAP evidence to root cause and action |
| `voltguard/simulator/` | PMSM replay with a closed-loop derate effect |
| `voltguard/database/` | ORM models and session management |
| `dashboard/` | Streamlit UI (live monitor, fleet, explainability, model performance) |
| `models/vN/` | Versioned artifacts; `models/latest.txt` selects the served version |

**Design choices**
- **XGBoost on engineered tabular features**, not a sequence model: CPU inference in
  milliseconds and native SHAP support. Short-term dynamics come from diff features
  (`dt_dt`, `acceleration_gradient`) computed over a rolling per-vehicle buffer.
- **Labels are built from channels the model cannot see.** This avoids the obvious
  leakage of predicting a temperature from itself.
- **Every prediction is audited**: each `/predict` call writes a `PredictionLog` row and
  updates that vehicle's health score and active faults.

---

## Quickstart

### 1. Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

### 2. Run the API (uses the committed model `v2`)
```bash
uvicorn api.main:app --reload          # http://localhost:8000/docs
```
Or with Docker:
```bash
docker build -t voltguard . && docker run -p 8000:8000 voltguard
```

### 3. Dashboard and simulator (need the dataset)
Download the [PMSM temperature dataset](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature),
save `measures_v2.csv` as `data/comprehensive_fault_training_data.csv`, then:
```bash
streamlit run dashboard/app.py
python scripts/simulate.py --vehicles 3 --ticks 20
```

### 4. Retrain
```bash
python scripts/train.py                 # writes models/vN and updates models/latest.txt
python scripts/generate_evaluation.py   # regenerates notebooks/04_evaluation.ipynb
```

---

## API

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Classify one reading; logged to `prediction_logs` |
| POST | `/predict/batch` | 1–1000 readings, all-or-nothing |
| POST | `/predict/forecast` | Trend-extrapolated early warning |
| GET | `/vehicles` | Fleet health overview |
| GET | `/vehicles/{id}/health` | Health of one vehicle |
| GET | `/models/latest`, `/models/{version}` | Model metrics |
| GET | `/health` | Liveness/readiness (database reachable) |
| WS | `/ws/alerts` | Alert stream (currently echoes client messages) |

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./voltguard.db` | Any SQLAlchemy URL |
| `VOLTGUARD_CORS_ORIGINS` | `http://localhost:8501` | Comma-separated allowed origins |
| `VOLTGUARD_ENABLE_DEMO_ENDPOINTS` | `0` | Set to `1` to enable `POST /vehicles/_seed_mock_fleet`, which wipes the fleet table |
| `VOLTGUARD_CONFIG` | `config/thresholds.yaml` | Thresholds and model hyperparameters |

## Development

```bash
make test     # pytest with coverage
make lint     # ruff
make format   # ruff format
```
CI (`.github/workflows/ci.yml`) runs lint and tests on Python 3.10 and 3.12, builds
the Docker image, and smoke-tests `/health`.

## Roadmap
- Database migrations (Alembic) instead of `create_all` at startup
- Authentication on write endpoints
- Connect `/ws/alerts` to the internal event bus
- Persist the per-vehicle feature buffer (currently in process memory, one worker only)
- Save models in XGBoost's native JSON format instead of pickle
