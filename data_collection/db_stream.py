import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator, Optional

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def db_stream(
    config_path: str = "config.yaml",
    batch_size: Optional[int] = None,
    date_ge: Optional[str] = None,
    date_ge_shift_days: int = 0,
) -> Iterator[tuple[list[dict], list]]:
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    root = cfg_path.parent

    features = cfg["columns"]["features"]
    target = cfg["columns"]["target"]
    db_path = root / cfg["storage"]["db_path"]
    fetch_size = batch_size or int(cfg["batch"]["size"])

    effective_date_ge = None
    if date_ge:
        effective_date_ge = (
            date.fromisoformat(date_ge) + timedelta(days=date_ge_shift_days)
        ).isoformat()

    query = "SELECT raw_json FROM raw_events"
    params: list[str] = []
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

            x_batch = []
            y_batch = []
            for (raw_json,) in rows:
                record = json.loads(raw_json)
                x_batch.append({col: record.get(col) for col in features})
                y_batch.append(record.get(target))

            yield x_batch, y_batch
    finally:
        conn.close()
