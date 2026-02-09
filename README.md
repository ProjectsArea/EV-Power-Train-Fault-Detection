# EV Fault Detection – AI-Based EV Powertrain Fault Detection (Flask)

This project is a Flask web application for **real-time EV powertrain/battery fault monitoring**.
It streams sensor samples from a dataset, extracts window-based features, runs an ML model (XGBoost) to estimate **risk probability**, and visualizes results on:

- Landing page (`/`)
- Detection page (`/detection`)
- Simulation page (`/simulation`)
- Analytics dashboard (`/analytics`)

The simulator runs in a **shared background thread** so that all pages show the **same live inputs** even when you navigate between pages.

---

## Key Features

- Shared backend simulator loop (navigation does not stop the simulation)
- Dataset-driven streaming from `data/EV_Battery_Fault_Diagnosis.csv`
- Window feature extraction + scaling (`StandardScaler`) + XGBoost probability output
- Live UI using Bootstrap + Chart.js
- Risk status based on threshold (default `0.6`)
- Analytics endpoint for dashboards

---

## Project Structure

```
Fault EV Power Train/
├─ app.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ EV_Battery_Fault_Diagnosis.csv
├─ models/
│  ├─ safety_xgb_model.json
│  ├─ raw_feature_scaler.pkl
│  ├─ window_feature_scaler.pkl
│  └─ config.pkl
├─ static/
│  └─ css/
│     └─ theme.css
└─ templates/
   ├─ index.html
   ├─ detection.html
   ├─ simulation.html
   └─ analytics.html
```

---

## Requirements

- Python **3.9+** (recommended 3.10/3.11)
- Windows / Linux / macOS

All Python dependencies are listed in `requirements.txt`.

---

## Setup (Step-by-step)

### 1) Copy the project to the new PC
- Copy the entire folder `Fault EV Power Train`.
- Make sure the following folders/files exist on the new PC:
  - `data/EV_Battery_Fault_Diagnosis.csv`
  - `models/` (all model + scaler files)

### 2) Create a virtual environment

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run the application
```bash
python app.py
```

Then open:
- `http://127.0.0.1:5000/`

---

## How the Live Data Works

- The backend starts a background simulation thread.
- Pages poll:
  - `GET /api/latest` for the latest sample
  - `GET /api/analytics` for analytics dashboard payload

Because the simulator is backend-driven, switching pages does not restart the simulation.

---

## API Endpoints (useful)

- `GET /api/latest`
  - returns the latest shared simulator payload
- `POST /api/sim/control`
  - start/stop simulator and change speed
- `GET /api/analytics`
  - analytics payload (risk series, performance metrics, trend analysis, etc.)
- `GET /reset`
  - reset simulator history and counters

---

## Troubleshooting

### 1) Model/scaler files not found
If you see errors like:
- `FileNotFoundError: models/...`

Fix:
- Ensure the `models/` folder exists and contains:
  - `safety_xgb_model.json`
  - `raw_feature_scaler.pkl`
  - `window_feature_scaler.pkl`
  - `config.pkl`

### 2) CSV file not found
If you see:
- `FileNotFoundError: data/EV_Battery_Fault_Diagnosis.csv`

Fix:
- Ensure the `data/` folder exists and has the dataset file.

### 3) Port already in use
If Flask says the port is busy:
- Close the other running instance
- Or change the port inside `app.py` in `app.run(...)`

### 4) Charts not updating
- Hard refresh the page (`Ctrl + F5`)
- Ensure the Flask server is running and `/api/latest` returns JSON.

---

## Notes

- UI: Bootstrap 5 + Chart.js
- Backend: Flask
- ML: XGBoost + scikit-learn scalers

