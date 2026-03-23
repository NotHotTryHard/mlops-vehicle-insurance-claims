import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from src.data.utils import load_config, parse_date


def db_stream(
    batch_size=None,
    date_ge=None,
    date_ge_shift_days=0,
):
    cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)
    root = cfg_path.parent

    features = cfg["columns"]["features"]
    target = cfg["columns"]["target"]
    db_path = root / cfg["data_storage"]["data_path"]
    fetch_size = batch_size or int(cfg["batch"]["size"])
    dt_fmt = cfg["columns"].get("datetime_format", "%Y-%m-%d")

    effective_date_ge = None
    if date_ge:
        base_date = parse_date(date_ge, dt_fmt, strict=True)
        effective_date_ge = (date.fromisoformat(base_date) + timedelta(days=date_ge_shift_days)).isoformat()

    query = "SELECT raw_json FROM raw_events"
    params = []
    if effective_date_ge:
        query += " WHERE event_date IS NOT NULL AND event_date >= ?"
        params.append(effective_date_ge)
    query += " ORDER BY event_date, id"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(query, params)
    try:
        while True:
            rows = cur.fetchmany(fetch_size)
            if not rows:
                break

            for (raw_json,) in rows:
                record = json.loads(raw_json)
            yield record
    finally:
        conn.close()
