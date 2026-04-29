import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from src.data.utils import load_config, parse_date


def db_stream(
    batch_size=None,
    date_ge=None,
    date_le=None,
    date_ge_shift_days=0,
    *,
    config_path="config.yaml",
):
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    root = cfg_path.parent

    db_path = root / cfg["data_storage"]["data_path"]
    fetch_size = batch_size or int(cfg["batch"]["size"])
    dt_fmt = cfg["columns"].get("datetime_format", "%Y-%m-%d")

    effective_date_ge = None
    if date_ge:
        base_date = parse_date(date_ge, dt_fmt, strict=True)
        effective_date_ge = (date.fromisoformat(base_date) + timedelta(days=date_ge_shift_days)).isoformat()

    effective_date_le = None
    if date_le:
        base_le = parse_date(date_le, dt_fmt, strict=True)
        effective_date_le = date.fromisoformat(base_le).isoformat()

    query = "SELECT raw_json FROM raw_events"
    clauses = []
    params = []
    if effective_date_ge:
        clauses.append("event_date IS NOT NULL AND event_date >= ?")
        params.append(effective_date_ge)
    if effective_date_le:
        clauses.append("event_date IS NOT NULL AND event_date <= ?")
        params.append(effective_date_le)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY event_date, id"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(query, params)
    batch = []
    try:
        while True:
            rows = cur.fetchmany(fetch_size)
            if not rows:
                break

            for (raw_json,) in rows:
                record = json.loads(raw_json)
                batch.append(record)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        conn.close()
