import csv
from datetime import date, datetime
from pathlib import Path

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


def load_training_data_quick(csv_path, feature_cols, target_col):
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


