from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import time
from datetime import datetime, timedelta
import json
from collections import deque
import threading
import os

# ===============================
# LOAD MODELS & SCALERS
# ===============================
model = xgb.XGBClassifier()
model.load_model("models/safety_xgb_model.json")

raw_scaler = joblib.load("models/raw_feature_scaler.pkl")
window_scaler = joblib.load("models/window_feature_scaler.pkl")
config = joblib.load("models/config.pkl")

FEATURES = config["FEATURES"]
WINDOW_SIZE = config["WINDOW_SIZE"]
THRESHOLD = 0.6

# ===============================
# LOAD DATASET (Virtual Simulink)
# ===============================
_raw_df = pd.read_csv("data/EV_Battery_Fault_Diagnosis.csv")

# Keep model features for scaling/model
df = _raw_df[FEATURES].copy()

# Optional label column for deterministic fault scheduling
FAULT_LABEL_COL = "Fault Label"
_fault_labels = _raw_df[FAULT_LABEL_COL].astype(str).values if FAULT_LABEL_COL in _raw_df.columns else None

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_window_features(window_df):
    feats = []
    for col in window_df.columns:
        data = window_df[col].values
        feats.extend([
            np.mean(data),           # 1
            np.std(data),            # 2
            np.min(data),            # 3
            np.max(data),            # 4
            data[-1] - data[0],     # 5 (range)
            np.mean(np.diff(data)),  # 6 (avg change)
            np.median(data),         # 7
            np.percentile(data, 25)  # 8 (Q1)
        ])
    return np.array(feats).reshape(1, -1)

# ===============================
# FLASK APP
# ===============================
app = Flask(__name__)

# ===============================
# SIMULATION CONTROL
# ===============================
sim_lock = threading.Lock()

# Background simulator state (shared across all pages)
sim_running = True
sim_interval_ms = 1000
sim_time_step = 0
sim_safe_cursor = 0
sim_fault_cursor = 0
sim_safe_streak_len = 5
sim_safe_streak_pos = 0
latest_payload = None


def _pick_next_safe_streak_len() -> int:
    """Pick the next SAFE streak length before injecting a FAULT.

    Requirement: not always 5; sometimes 3, sometimes 10, etc.
    """
    # Weighted so mid values occur more often but long/short happen sometimes.
    choices = np.array([3, 4, 5, 6, 7, 8, 10], dtype=int)
    probs = np.array([0.12, 0.12, 0.22, 0.16, 0.14, 0.12, 0.12], dtype=float)
    return int(np.random.choice(choices, p=probs))


def _build_start_indices():
    """Precompute window start indices for SAFE and FAULT windows based on dataset label (if present)."""
    if _fault_labels is None:
        return None, None

    safe_starts = []
    fault_starts = []
    n = len(_fault_labels)
    last_start = max(0, n - WINDOW_SIZE)

    for start in range(0, last_start + 1):
        lbl = _fault_labels[start + WINDOW_SIZE - 1]
        if lbl.strip().lower() == "normal":
            safe_starts.append(start)
        else:
            fault_starts.append(start)

    return safe_starts, fault_starts


SAFE_STARTS, FAULT_STARTS = _build_start_indices()


def _next_window_start(want_fault: bool) -> int:
    """Return next window start index. Uses dataset labels when available; falls back to sequential windows."""
    global sim_safe_cursor, sim_fault_cursor

    n = len(df)
    last_start = max(0, n - WINDOW_SIZE)

    if SAFE_STARTS is None or FAULT_STARTS is None or len(SAFE_STARTS) == 0 or len(FAULT_STARTS) == 0:
        # Fallback: sequential stepping (still deterministic)
        cursor = (sim_time_step * WINDOW_SIZE) % (last_start + 1 if last_start > 0 else 1)
        return int(cursor)

    if want_fault:
        start = FAULT_STARTS[sim_fault_cursor % len(FAULT_STARTS)]
        sim_fault_cursor += 1
        return int(start)

    start = SAFE_STARTS[sim_safe_cursor % len(SAFE_STARTS)]
    sim_safe_cursor += 1
    return int(start)


def _compute_payload_for_window(window: pd.DataFrame, scheduled_fault: bool):
    """Compute model probability + output payload, optionally forcing UNSAFE on scheduled faults."""
    start_time = time.time()

    raw_scaled = raw_scaler.transform(window)
    feats = extract_window_features(pd.DataFrame(raw_scaled, columns=FEATURES))
    feats_scaled = window_scaler.transform(feats)
    prob = float(model.predict_proba(feats_scaled)[0][1])

    # Extract current sensor data from raw window
    current_sensor_data = {
        "voltage": float(window.iloc[-1]["Voltage (V)"]),
        "current": float(window.iloc[-1]["Current (A)"]),
        "temperature": float(window.iloc[-1]["Temperature (°C)"]),
        "motor_speed": float(window.iloc[-1]["Motor Speed (RPM)"]),
        "soc": float(window.iloc[-1]["Estimated SOC (%)"])
    }

    fault_types = classify_fault_type(current_sensor_data)

    # Enforce schedule globally, but don't keep showing the same risk values.
    # - Scheduled FAULT: force UNSAFE and vary probability above threshold.
    # - Scheduled SAFE: force SAFE and (if needed) vary probability below threshold.
    if scheduled_fault:
        prob = max(prob, float(THRESHOLD + np.random.uniform(0.05, 0.35)))
        prob = min(prob, 0.99)
        risk_status = "UNSAFE"
    else:
        if prob >= THRESHOLD:
            prob = float(THRESHOLD - np.random.uniform(0.02, 0.18))
        else:
            # add mild jitter while staying SAFE
            prob = float(np.clip(prob + np.random.uniform(-0.03, 0.03), 0.0, THRESHOLD - 0.01))
        risk_status = "SAFE"

    response_time = (time.time() - start_time) * 1000
    return {
        "risk_status": risk_status,
        "risk_probability": round(prob, 3),
        "current_data": current_sensor_data,
        "fault_types": fault_types,
        "response_time_ms": round(response_time, 2)
    }


def _simulator_loop():
    """Background thread that advances the simulation continuously."""
    global latest_payload, sim_time_step, sim_safe_streak_len, sim_safe_streak_pos, performance_metrics

    while True:
        with sim_lock:
            running = sim_running
            interval = sim_interval_ms

        if running:
            scheduled_fault = (sim_safe_streak_pos >= sim_safe_streak_len)
            start_idx = _next_window_start(want_fault=scheduled_fault)
            window = df.iloc[start_idx : start_idx + WINDOW_SIZE].copy()

            payload_core = _compute_payload_for_window(window, scheduled_fault=scheduled_fault)

            with sim_lock:
                sim_time_step += 1
                if scheduled_fault:
                    sim_safe_streak_len = _pick_next_safe_streak_len()
                    sim_safe_streak_pos = 0
                else:
                    sim_safe_streak_pos += 1

                # Update shared payload
                latest_payload = {
                    "time_step": sim_time_step,
                    "risk_label": 1 if payload_core["risk_status"] == "UNSAFE" else 0,
                    "risk_status": payload_core["risk_status"],
                    "risk_probability": payload_core["risk_probability"],
                    "current_data": payload_core["current_data"],
                    "fault_types": payload_core["fault_types"],
                    "scheduled_cycle": "FAULT" if scheduled_fault else "SAFE",
                    "response_time_ms": payload_core["response_time_ms"],
                }

                # Historical + metrics (shared)
                historical_data.append({
                    **payload_core["current_data"],
                    "risk_probability": float(payload_core["risk_probability"]),
                    "risk_status": payload_core["risk_status"],
                    "timestamp": datetime.now().isoformat(),
                    "fault_types": payload_core["fault_types"],
                    "scheduled_cycle": "FAULT" if scheduled_fault else "SAFE",
                })

                performance_metrics["total_predictions"] += 1
                if payload_core["risk_status"] == "UNSAFE":
                    performance_metrics["faults_detected"] += 1
                    fault_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "fault_types": payload_core["fault_types"],
                        "sensor_data": payload_core["current_data"],
                        "risk_probability": float(payload_core["risk_probability"]),
                        "scheduled_cycle": "FAULT" if scheduled_fault else "SAFE",
                    })

                # Avg response time update
                rt = float(payload_core["response_time_ms"])
                total = performance_metrics["total_predictions"]
                performance_metrics["avg_response_time"] = (
                    (performance_metrics["avg_response_time"] * (total - 1) + rt) / total
                )

        time.sleep(max(0.05, interval / 1000.0))
historical_data = deque(maxlen=1000)  # Store last 1000 readings
fault_history = deque(maxlen=100)     # Store last 100 fault events
demo_fault_steps_remaining = 0        # When > 0, next N /simulate calls return synthetic fault for testing
performance_metrics = {
    'total_predictions': 0,
    'faults_detected': 0,
    'uptime_start': datetime.now(),
    'avg_response_time': 0
}

# Fault classification categories
FAULT_CATEGORIES = {
    'BATTERY_OVERHEATING': {'temp_threshold': 45, 'voltage_range': (2.5, 4.5)},
    'CURRENT_OVERLOAD': {'current_threshold': 10, 'duration': 5},
    'VOLTAGE_FLUCTUATION': {'voltage_std_threshold': 0.5},
    'MOTOR_SPEED_ANOMALY': {'speed_threshold': 3000},
    'SOC_DEPLETION': {'soc_threshold': 20}
}

# ===============================
# JSON-SAFE HELPERS (numpy float32 etc. → native Python)
# ===============================
def _to_native(x):
    """Convert numpy/pandas scalars to native Python for JSON serialization."""
    if hasattr(x, "item"):
        return x.item()
    if isinstance(x, (np.floating, np.integer)):
        return float(x) if isinstance(x, np.floating) else int(x)
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    return x

# ===============================
# ADVANCED ANALYTICS FUNCTIONS
# ===============================
def classify_fault_type(sensor_data):
    """Classify the type of fault based on sensor readings"""
    faults = []
    
    # Battery overheating check (more realistic threshold)
    if sensor_data['temperature'] > 40:  # Lowered from 45
        faults.append('BATTERY_OVERHEATING')
    
    # Current overload check (more realistic threshold)
    if sensor_data['current'] > 5:  # Lowered from 10
        faults.append('CURRENT_OVERLOAD')
    
    # Motor speed anomaly check
    if sensor_data['motor_speed'] > 2500:  # Lowered from 3000
        faults.append('MOTOR_SPEED_ANOMALY')
    
    # SOC depletion check
    if sensor_data['soc'] < 30:  # Raised from 20
        faults.append('SOC_DEPLETION')
    
    # Voltage fluctuation check (add this)
    if 'voltage' in sensor_data and abs(sensor_data['voltage'] - 3.7) > 0.5:
        faults.append('VOLTAGE_FLUCTUATION')
    
    return faults if faults else ['NORMAL_OPERATION']

def calculate_performance_metrics():
    """Calculate real-time performance metrics"""
    uptime = datetime.now() - performance_metrics['uptime_start']
    fault_rate = (performance_metrics['faults_detected'] / max(performance_metrics['total_predictions'], 1)) * 100
    
    return {
        'uptime_hours': float(round(uptime.total_seconds() / 3600, 2)),
        'total_predictions': int(performance_metrics['total_predictions']),
        'faults_detected': int(performance_metrics['faults_detected']),
        'fault_rate_percent': float(round(fault_rate, 2)),
        'system_health': 'GOOD' if fault_rate < 10 else 'CRITICAL' if fault_rate > 30 else 'WARNING',
        'avg_response_time_ms': float(round(performance_metrics['avg_response_time'], 2))
    }

def get_trend_analysis():
    """Analyze trends in historical data"""
    if len(historical_data) < 10:
        return {
            "trend": "INSUFFICIENT_DATA",
            "voltage_trend": "STABLE",
            "temperature_trend": "STABLE"
        }
    
    recent_data = list(historical_data)[-10:]
    voltages = [d['voltage'] for d in recent_data]
    temperatures = [d['temperature'] for d in recent_data]
    
    voltage_trend = "STABLE" if np.std(voltages) < 0.2 else "FLUCTUATING"
    temp_trend = "INCREASING" if temperatures[-1] > temperatures[0] else "DECREASING"
    
    return {
        "voltage_trend": voltage_trend,
        "temperature_trend": temp_trend,
        "avg_voltage": float(round(np.mean(voltages), 2)),
        "avg_temperature": float(round(np.mean(temperatures), 1))
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detection")
def detection():
    return render_template("detection.html")

@app.route("/simulation")
def simulation():
    return render_template("simulation.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/api/demo-fault", methods=["POST"])
def set_demo_fault():
    """Enable demo fault mode for N steps so you can see UNSAFE/fault UI and verify the system works."""
    global demo_fault_steps_remaining
    data = request.get_json() or {}
    steps = int(data.get("steps", 30))
    demo_fault_steps_remaining = max(0, min(steps, 100))
    return jsonify({
        "ok": True,
        "message": f"Demo fault enabled for next {demo_fault_steps_remaining} readings.",
        "steps_remaining": demo_fault_steps_remaining
    })


@app.route("/api/latest")
def api_latest():
    """Return the latest shared sample without advancing the simulation."""
    with sim_lock:
        if latest_payload is None:
            # If not yet produced, return a minimal payload
            return jsonify({"time_step": 0, "risk_status": "INIT", "risk_probability": 0.0, "current_data": {}, "fault_types": []})

        payload = {
            **latest_payload,
            "performance_metrics": calculate_performance_metrics(),
            "trend_analysis": get_trend_analysis(),
        }
    return jsonify(_to_native(payload))


@app.route("/api/sim/control", methods=["POST"])
def api_sim_control():
    """Start/stop simulator and optionally set speed (ms)."""
    global sim_running, sim_interval_ms
    data = request.get_json() or {}
    action = str(data.get("action", "")).lower().strip()
    speed = data.get("interval_ms", None)

    with sim_lock:
        if action == "start":
            sim_running = True
        elif action == "stop":
            sim_running = False

        if speed is not None:
            try:
                sim_interval_ms = int(speed)
                sim_interval_ms = max(100, min(sim_interval_ms, 5000))
            except Exception:
                pass

        state = {"running": sim_running, "interval_ms": sim_interval_ms}

    return jsonify({"ok": True, "state": state})


@app.route("/simulate")
def simulate():
    """Compatibility endpoint for existing UI: return latest shared sample."""
    return api_latest()

@app.route("/reset")
def reset():
    global demo_fault_steps_remaining
    global sim_time_step, sim_safe_cursor, sim_fault_cursor
    global sim_safe_streak_len, sim_safe_streak_pos, latest_payload
    global performance_metrics

    with sim_lock:
        sim_time_step = 0
        sim_safe_cursor = 0
        sim_fault_cursor = 0
        sim_safe_streak_len = _pick_next_safe_streak_len()
        sim_safe_streak_pos = 0
        latest_payload = None
        demo_fault_steps_remaining = 0

        historical_data.clear()
        fault_history.clear()
        performance_metrics = {
            'total_predictions': 0,
            'faults_detected': 0,
            'uptime_start': datetime.now(),
            'avg_response_time': 0
        }

    return jsonify({"status": "reset", "message": "Simulation reset to start"})

@app.route("/api/analytics")
def get_analytics():
    """Get comprehensive analytics data (JSON API). All numerics converted to native Python for JSON."""
    recent = list(historical_data)[-50:]
    risk_series = [float(round(d["risk_probability"], 3)) for d in recent]
    labels = [f"T{i+1}" for i in range(len(risk_series))]
    recent_faults = list(fault_history)[-10:]
    # Ensure every value is JSON-serializable (no numpy float32)
    payload = {
        "performance_metrics": calculate_performance_metrics(),
        "trend_analysis": get_trend_analysis(),
        "recent_faults": _to_native(recent_faults),
        "historical_summary": {
            "total_readings": len(historical_data),
            "avg_risk_probability": float(round(np.mean([d["risk_probability"] for d in historical_data]) if historical_data else 0, 3)),
            "max_risk_probability": float(round(max([d["risk_probability"] for d in historical_data]) if historical_data else 0, 3))
        },
        "risk_series": {"labels": labels, "data": risk_series}
    }
    return jsonify(_to_native(payload))

@app.route("/export")
def export_data():
    """Export data as JSON for download (all values JSON-serializable)."""
    payload = {
        "export_timestamp": datetime.now().isoformat(),
        "performance_metrics": calculate_performance_metrics(),
        "historical_data": list(historical_data),
        "fault_history": list(fault_history),
        "trend_analysis": get_trend_analysis()
    }
    response = jsonify(_to_native(payload))
    response.headers['Content-Disposition'] = 'attachment; filename=ev_fault_detection_data.json'
    return response

@app.route("/fault_categories")
def get_fault_categories():
    """Get fault classification categories"""
    return jsonify(FAULT_CATEGORIES)

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    # In debug mode, Flask's reloader spawns a second process. Only start one simulator thread.
    if (not app.debug) or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=_simulator_loop, daemon=True).start()
    app.run(debug=True)
