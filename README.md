# VoltGuard

VoltGuard tries to answer one question: can you tell that an EV motor is heading
into thermal trouble just from its electrical and control signals, before the
temperatures you can't measure directly get there?

It's built on the [Paderborn University PMSM dataset](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature):
1.33 million readings, sampled at 2 Hz across 69 recorded drives of a permanent-magnet
motor on a test bench. An XGBoost model looks at voltages, currents, speed, torque,
coolant and ambient temperature, and flags when the motor looks stressed. SHAP explains
each call, a small rules layer suggests what to do about it, and a simulator replays real
drives so you can watch the whole loop run, including backing off the motor when
an alert fires.

Around that sits a FastAPI service, a Next.js dashboard, a versioned model registry,
drift checks, Docker Compose and CI.

## Results

Tested on 14 drives that were held out of training (309k rows). The baseline is a model
that predicts "Nominal" for every row.

| Metric | Model v2 | Baseline |
|---|---|---|
| Macro F1 | 0.34 | 0.24 |
| Macro F1, 5-fold CV over drives | 0.45 ± 0.02 | – |
| Accuracy | 0.90 | 0.96 |

| Class | Precision | Recall | F1 | Drives with this class |
|---|---|---|---|---|
| Nominal | 0.99 | 0.92 | 0.95 | 69 |
| Stator overheat | 0.16 | 0.37 | 0.22 | 39 |
| Coolant stress | 0.10 | 0.66 | 0.17 | 25 |
| Inverter over-current | 0.00 | 0.00 | 0.00 | 15 |
| Battery thermal stress | – | – | – | 1 |

Training uses balanced class weights, so the model favours recall over precision. On
unseen drives it catches 66% of coolant-stress rows and 37% of stator-overheat rows, but
most of its alerts are false alarms, and its accuracy ends up below the baseline. It
never predicts inverter over-current correctly. Battery thermal stress only appears in one
drive, so it can't be tested and is left out of the macro scores.

v1 of this model reported 98.8% accuracy and 0.93 macro F1. That came from a random row
split. The data is sampled at 2 Hz, so neighbouring rows are almost identical, and every
test drive had rows in the training set too. With whole drives held out, the same setup
scores 88–93% accuracy and 0.25–0.33 macro F1 across five splits, against 88–93% and 0.19
for the baseline. The training code now splits by drive for both the test set and
cross-validation. `models/v1` is kept for comparison.

### Limitations

- There are no real failures in the dataset. The labels are thresholds on the hidden
  temperatures (stator winding, tooth, yoke and magnet): a row counts as a fault when one of
  them is in its top 3–6%. The model doesn't get these temperatures as inputs. "Coolant
  System Failure" means the row looks like coolant stress, not that a part failed. See
  `generate_fault_codes` in [voltguard/features/engine.py](voltguard/features/engine.py).
- `battery_temp` isn't measured. It's calculated from electrical power.
- 14 test drives is a small sample. Use the CV standard deviation as the error bar.
- The simulator replays recorded temperatures. A derate changes the model's inputs but not
  the temperatures that follow.

## The dashboard

The Next.js app in `web/` has five pages:

- **Overview**: what the system does, the headline numbers against the baseline, how rare
  each condition is, and how stress is spread across drives.
- **Live monitor**: replays one recorded drive through the real model. Every alert is
  checked against what the hidden temperatures actually did, so you see hits, false
  alarms and misses as they happen, along with the SHAP reasons and any back-off action.
- **Fleet**: eight vehicles replaying different drives, with a heatmap of alerts and a
  running count of how many turned out to be real.
- **Explainability**: sliders for the eight inputs, set to realistic ranges from the data,
  so you can poke at the model and see which inputs push it where.
- **Model**: the full evaluation, per-condition results and a confusion matrix.

There's also an older Streamlit app in `dashboard/`. I kept it as an internal tool for
checking drift baselines and feature schemas. It isn't the main UI anymore.

## How it fits together

```text
browser ──► Next.js (web/) ──/api──► FastAPI (api/)
                                        │
                   ┌────────────────────┼─────────────────────┐
                   ▼                    ▼                     ▼
           FaultPredictor         simulator sessions     PredictionLog, vehicles
     features → XGBoost → SHAP    replay real drives     (SQLite locally, Postgres
                   │              + ground truth          via DATABASE_URL)
                   ▼
     root_cause → suggested action → derate in the simulator
```

| Folder | What's in it |
|---|---|
| `api/` | The FastAPI app. `routes/simulator.py` runs the replay sessions the dashboard uses |
| `web/` | The Next.js dashboard |
| `voltguard/features/` | Feature engineering and the proxy labels |
| `voltguard/diagnostics/` | Training, evaluation, the model registry, prediction and drift checks |
| `voltguard/root_cause/` | Turns a prediction and its SHAP values into a likely cause and an action |
| `voltguard/simulator/` | Replays the dataset and applies back-off actions |
| `voltguard/database/` | Database models and sessions |
| `models/vN/` | Trained models. `models/latest.txt` picks the one the API serves |
| `dashboard/` | The internal Streamlit tool |

A few decisions worth explaining:

- **XGBoost on tabular features rather than a sequence model.** It runs in milliseconds
  on a CPU and SHAP supports it natively. Short-term trends come from rate-of-change
  features computed over a small rolling buffer per vehicle.
- **The labels come from signals the model can't see.** Otherwise it would just be
  predicting a temperature from itself.
- **Every prediction is logged.** Each call to `/predict` writes a row to the audit table and
  updates that vehicle's health score.

## Running it

### With Docker

```bash
docker compose up --build
```

The dashboard is at http://localhost:3000 and the API docs are at http://localhost:8000/docs.

### Locally

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn api.main:app --reload      # API on :8000, serves the committed model v2
```

In a second terminal:

```bash
cd web
pnpm install
pnpm dev                           # dashboard on :3000
```

### Getting the dataset

The model ships with the repo, so the API and most of the dashboard work without the
data. The simulator needs it for ground truth, though. Download `measures_v2.csv` from
[Kaggle](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) and save it
as `data/comprehensive_fault_training_data.csv`. Without it, the simulator replays a
fixed placeholder reading and can't tell you whether an alert was right.

### Retraining

```bash
python scripts/train.py                 # saves models/vN and points latest.txt at it
python scripts/generate_evaluation.py   # rebuilds notebooks/04_evaluation.ipynb from the new metrics
```

### Other tools

```bash
streamlit run dashboard/app.py                         # drift and feature-schema checks
python scripts/simulate.py --vehicles 3 --ticks 20     # terminal version of the simulator
```

### If something goes wrong on Windows

If the API starts but fails with a DLL or "Application Control policy" error the first time
it loads the model, that's Windows Smart App Control blocking native libraries such as
pyarrow and numba. Turning it off (Windows Security → App & browser control) or running with
Docker gets around it.

## API

| Method | Path | What it does |
|---|---|---|
| POST | `/predict` | Classifies one reading and logs it |
| POST | `/predict/batch` | Classifies 1–1000 readings. All of them succeed or none are saved |
| POST | `/predict/forecast` | Extrapolates recent trends and predicts a few ticks ahead |
| POST | `/predict/explain` | Scores one what-if scenario without logging or remembering it |
| POST | `/simulator/sessions` | Starts a replay with 1–12 vehicles |
| POST | `/simulator/sessions/{id}/tick` | Advances 1–200 ticks and returns predictions, ground truth and actions |
| DELETE | `/simulator/sessions/{id}` | Ends a replay |
| GET | `/data/summary` | Class balance, stress per drive, and input ranges |
| GET | `/vehicles` | Fleet health overview |
| GET | `/vehicles/{id}/health` | Health of one vehicle |
| GET | `/models/latest`, `/models/{version}` | Metrics for a model version |
| GET | `/health` | Checks the API and database are up |
| WS | `/ws/alerts` | Alert stream. For now it only echoes messages back |

## Configuration

| Variable | Default | What it's for |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./voltguard.db` | Any SQLAlchemy database URL |
| `VOLTGUARD_CORS_ORIGINS` | `http://localhost:3000,http://localhost:8501` | Browser origins allowed to call the API |
| `VOLTGUARD_REPLAY_DATA` | `data/comprehensive_fault_training_data.csv` | The dataset the simulator replays |
| `VOLTGUARD_API_URL` | `http://localhost:8000` | Set on the web app: where its `/api` proxy sends requests |
| `VOLTGUARD_ENABLE_DEMO_ENDPOINTS` | `0` | Set to `1` to allow `POST /vehicles/_seed_mock_fleet`, which wipes and reseeds the fleet table |
| `VOLTGUARD_CONFIG` | `config/thresholds.yaml` | Thresholds and model settings |

Relative paths are resolved from the project folder, so it doesn't matter where you start
the API from.

## Development

```bash
make test        # Python tests with coverage
make lint        # ruff
make format      # ruff format
cd web && pnpm lint && pnpm typecheck && pnpm build
```

The tests don't need the 300 MB dataset. They generate a tiny fake one instead. CI runs the
Python tests on 3.10 and 3.12, lints, type-checks and builds the web app, then builds both
Docker images and checks that the API comes up.

## What's next

- Database migrations with Alembic instead of creating tables at startup
- Authentication on the endpoints that write data
- Hooking `/ws/alerts` up to real events
- Moving the per-vehicle buffers and simulator sessions out of process memory, so the API
  can run with more than one worker
- Saving models in XGBoost's JSON format instead of pickle
- Better labels. A dataset with real recorded failures would make most of the caveats above go away
