import csv
import json
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path
import tqdm

from src.data.utils import load_config, parse_date
from src.data.quality.eda import eda_column_names, run_automatic_eda
from src.data.quality.stats import DataStatsGlobalAnalyzer


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
    db_path = root / cfg["data_storage"]["data_path"]
    if paths:
        data_sources = paths
    else:
        data_sources = [root / src["path"] for src in cfg["data_sources"]]

    conn = db_init(db_path)
    cur = conn.cursor()

    eda_cfg = cfg.get("eda") or {}
    eda_max = int(eda_cfg.get("max_rows", 30_000))
    eda_cols = eda_column_names(cfg)
    eda_rows: list[dict] = []

    analyzer = DataStatsGlobalAnalyzer(cfg, missing_values=(None, ""), round_precision=3, dt_col=dt_col, dt_fmt=dt_fmt)
    analyzer.merge_existing_reports(cfg, root)
    for source in data_sources:
        print(f"Streaming from {source}...")
        for batch_idx, batch in tqdm.tqdm(enumerate(stream_batches(source, batch_size), start=1)):
            loaded_at = datetime.now().isoformat(timespec="seconds")
            analyzer.update(batch)
            if len(eda_rows) < eda_max:
                for _row_num, row in batch:
                    if len(eda_rows) >= eda_max:
                        break
                    eda_rows.append({c: row.get(c) for c in eda_cols})
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
    print(analyzer.meta_analyzer)
    analyzer.save_report()
    run_automatic_eda(str(cfg_path.resolve()), eda_rows)


def ensure_db():
    config_path = "config.yaml"
    cfg = load_config(config_path)
    root = Path(config_path).parent
    db_path = root / cfg["data_storage"]["data_path"]

    if db_path.exists():
        conn = sqlite3.connect(db_path)
        try:
            (count,) = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()
        finally:
            conn.close()

        if count > 0:
            return

    sources = [root / src["path"] for src in cfg.get("data_sources") or []]
    existing = [p for p in sources if p.is_file()]

    if existing:
        print(
            f"DB not found or empty - loading {len(existing)} CSV file(s) from data_sources..."
        )
        db_add_tables(config_path, paths=existing)
        return

    print(
        "DB not found or empty. "
        "Creating empty SQLite schema. Load rows with:\n"
        "  python run.py --mode add_data --path-csv /path/to/file.csv"
    )
    conn = db_init(db_path)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    db_add_tables(config_path=args.config, max_batches=args.max_batches)
