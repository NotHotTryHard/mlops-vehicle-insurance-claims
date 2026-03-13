import csv
import json
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stream_batches(csv_path: Path, batch_size: int):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        batch = []
        for idx, row in enumerate(reader, start=1):
            batch.append((idx, row))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL,
            row_number INTEGER NOT NULL,
            event_date TEXT,
            raw_json TEXT NOT NULL,
            loaded_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def parse_event_date(value: str, fmt: str) -> Optional[str]:
    if not value:
        return None
    try:
        return datetime.strptime(value, fmt).date().isoformat()
    except ValueError:
        return None


def ingest(config_path: str = "config.yaml", max_batches: Optional[int] = None):
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    root = cfg_path.parent

    batch_size = int(cfg["batch"]["size"])
    dt_col = cfg["columns"]["datetime"]
    dt_fmt = cfg["columns"].get("datetime_format", "%Y-%m-%d")
    db_path = root / cfg["storage"]["db_path"]
    data_sources = [root / src["path"] for src in cfg["data_sources"]]

    conn = init_db(db_path)
    cur = conn.cursor()

    for source in data_sources:
        pass

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    ingest(args.config, args.max_batches)
