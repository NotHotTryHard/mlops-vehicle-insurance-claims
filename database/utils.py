import csv
from datetime import date, datetime
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_date(value, fmt, strict=True):
    if value is None:
        if strict:
            raise ValueError("Date value is required")
        return None

    raw = str(value).strip()
    if not raw:
        if strict:
            raise ValueError("Date value is empty")
        return None

    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            if strict:
                raise
            return None


def load_raw_csv(csv_path):
    cfg = load_config("config.yaml")
    feature_cols = cfg["columns"]["features"]
    target_col = cfg["columns"]["target"]
    csv_path = Path(csv_path)
    X, y = [], []
    required_cols = list(feature_cols) + [target_col]

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(row.get(col, "") == "" for col in required_cols):
                continue
            X.append({col: row[col] for col in feature_cols})
            y.append(row[target_col])

    return X, y


def load_raw(path_csv=None):
    from database.db_stream import db_stream

    if path_csv:
        X_raw, y_raw = load_raw_csv(str(path_csv))
        return X_raw, np.array([float(v) for v in y_raw], dtype=np.float32)

    all_x, all_y = [], []
    for x_batch, y_batch in db_stream():
        for x, y in zip(x_batch, y_batch):
            if y is not None:
                all_x.append(x)
                all_y.append(y)
    return all_x, np.array([float(v) for v in all_y], dtype=np.float32)
