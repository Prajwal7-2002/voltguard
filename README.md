# ⚡ VoltGuard: Predictive Drivetrain Diagnostics & Self-Healing Engine

**VoltGuard** is a production-grade machine learning architecture designed to ingest highly granular, 2Hz Electric Vehicle (EV) motor telemetry, mathematically predict thermal breakdown in real-time, and execute automated physical actuation to prevent catastrophic hardware failure.

> **The Narrative Shift:** While many dashboards rely on synthetic random number generators, VoltGuard's logic pipeline is strictly calibrated against the **Paderborn University Electric Motor Temperature (PMSM) Dataset**. It natively parses real-world synchronous machine arrays (`u_d`, `u_q`, `motor_speed`, `coolant`, `stator_winding`) to solve actual thermal management problems. 

---

## 🏗️ 1. Engineering Decisions & Architecture

This SaaS uses strict decoupling to separate the "Data Science" from the "Web Infrastructure."

### Decision 1: XGBoost tabluar mapping vs LSTMs
Given that time-series data is inherently sequential, LSTMs (Long Short-Term Memory networks) are the standard academic choice. However, in an EV edge-computing environment where predictions must happen in milliseconds to prevent a fire, LSTMs are too slow and computationally expensive. 
*   **The Pivot:** We implemented **XGBoost (Extreme Gradient Boosting)** combined with highly engineered static features (e.g. `thermal_delta = coolant - ambient`). XGBoost runs orders of magnitude faster on CPU-bound microservices and completely ignores standard variance noise.

### Decision 2: Isolated SQLAlchemy Subsystem (`voltguard.db`)
Using in-memory global dictionaries for fleet mapping creates fatal race conditions when thousands of vehicles start sending JSON arrays to the server concurrently.
*   **The Pivot:** We inject an asynchronous `SessionLocal` SQLite (easily swappable to PostgreSQL) hook into the FastAPI routes. It guarantees ACID compliance so that if Vehicle `v-404` crashes the system, the rest of the fleet is not disrupted.

### Decision 3: The SHAP "Closed-Loop" Actuator
Throwing an alert on a dashboard doesn't save the motor. 
*   **The Pivot:** We pass the XGBoost output through a SHAP (SHapley Additive exPlanations) `TreeExplainer`. The analyzer parses the exact node that caused the failure prediction (e.g., "stator winding exceeded threshold"). Once isolated, VoltGuard sends a literal return request to the `self_heal.effects` loop, clamping the `motor_speed` RPM until thermal equilibrium is restored.

---

## 🛠️ 2. Core Operational Modules

```text
VoltGuard-Platform/
├── api/                       # RESTful FastAPI microservice for Fleet Telemetry matching
├── dashboard/                 # Streamlit frontend with strictly defined UI tabs
├── data/                      # 📌 Target directory for the PMSM measures_v2.csv dataset
├── notebooks/                 # MLOps Evaluation Pipelines (Data Analysis, Feature Importance)
├── scripts/                   # System automation nodes (Retraining, Downloads)
├── tests/                     # 10 Pytest fixture routes validating mathematical API boundaries
└── voltguard/                 # 🧠 The native intelligence Python package
    ├── database/              # Schema mappings & ORM
    ├── diagnostics/           # Model loading, scaling, and Data Drift scoring algorithms
    ├── features/              # Feature creation mapping PMSM physics
    └── self_heal/             # The intervention logic
```

---

## 🔬 3. What the Model Gets Wrong (Error Bounds)

VoltGuard is highly accurate, but intentionally conservative. As documented heavily in `notebooks/04_evaluation.ipynb`, the model actively hallucinates False Positives during one specific scenario:
*   **High Ambient + Sudden Torque Spike:** If the EV accelerates massively out of a stoplight on a hot day, `u_q` and `motor_speed` spike instantly while `ambient` is already high. The tree logic incorrectly correlates this with an imminent `stator_winding` failure, even if the `coolant` hasn't failed yet. 

---

## 🚀 4. How to Launch the System

*Because this repository uses real data, you must process the telemetry before booting.*

### Step 1. Ingest Data
Download the real Kaggle Paderborn University dataset, place `measures_v2.csv` into the `/data/` folder, and rename it to `comprehensive_fault_training_data.csv`. Run the data ingestion prompt:
```powershell
python scripts/download_real_telemetry.py
```

### Step 2. Start the API Server
```powershell
uvicorn api.main:app --reload
```

### Step 3. Launch the Command Center UI
Open a split terminal and run:
```powershell
streamlit run dashboard/app.py
```
