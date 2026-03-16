import csv
import json
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path
import tqdm

try:
    from data_collection.utils import load_config, parse_date
except ImportError:
    from utils import load_config, parse_date


def stream_batches(csv_path, batch_size):
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


def db_init(db_path):
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


def db_add_tables(config_path="config.yaml", paths=None, max_batches=None):
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    root = cfg_path.parent

    batch_size = int(cfg["batch"]["size"])
    dt_col = cfg["columns"]["datetime"]
    dt_fmt = cfg["columns"].get("datetime_format", "%Y-%m-%d")
    db_path = root / cfg["storage"]["db_path"]
    if paths:
        data_sources = paths
    else:
        data_sources = [root / src["path"] for src in cfg["data_sources"]]

    conn = db_init(db_path)
    cur = conn.cursor()

    for source in data_sources:
        print(f"Streaming from {source}...")
        for batch_idx, batch in tqdm.tqdm(enumerate(stream_batches(source, batch_size), start=1)):
            loaded_at = datetime.now().isoformat(timespec="seconds")
            rows_to_insert = [
                (
                    str(source),
                    row_num,
                    parse_date(row.get(dt_col), dt_fmt, strict=False),
                    json.dumps(row, ensure_ascii=False),
                    loaded_at,
                )
                for row_num, row in batch
            ]
            cur.executemany(
                """
                INSERT INTO raw_events (source_path, row_number, event_date, raw_json, loaded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            conn.commit()
            if max_batches and batch_idx >= max_batches:
                print('Early stop')
                break

    conn.close()
    print(f"Done. SQLite DB: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    db_add_tables(config_path=args.config, max_batches=args.max_batches)
