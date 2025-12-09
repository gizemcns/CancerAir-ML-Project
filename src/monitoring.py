

import sqlite3
import json
from datetime import datetime
import os

BASE = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE, "monitoring.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        payload TEXT,
        prediction INTEGER,
        probabilities TEXT,
        model_version TEXT,
        latency_ms INTEGER,
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

def log_prediction(payload: dict, prediction: int, probabilities: list, model_version="v1", latency_ms=0):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO prediction_logs (payload, prediction, probabilities, model_version, latency_ms, created_at) VALUES (?,?,?,?,?,?)",
            (json.dumps(payload), int(prediction), json.dumps(probabilities), model_version, int(latency_ms), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# initialize DB on import/time of service start
init_db()
