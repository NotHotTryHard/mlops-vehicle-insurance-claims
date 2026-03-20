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


def load_training_data_ready(csv_path, feature_cols, target_col):
    X_raw, y_raw = load_training_data_quick(csv_path, feature_cols, target_col)

    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        is_numeric = True
        for row in X_raw:
            try:
                float(row[col])
            except ValueError:
                is_numeric = False
                break
        if is_numeric:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    rows_count = len(X_raw)
    numeric_part = np.zeros((rows_count, len(numeric_cols)), dtype=np.float64)
    for j, col in enumerate(numeric_cols):
        numeric_part[:, j] = [float(row[col]) for row in X_raw]

    one_hot_cols = []
    for col in categorical_cols:
        values = sorted({row[col] for row in X_raw})
        for value in values:
            one_hot_cols.append((col, value))

    cat_part = np.zeros((rows_count, len(one_hot_cols)), dtype=np.float64)
    for i, row in enumerate(X_raw):
        for j, (col, value) in enumerate(one_hot_cols):
            if row[col] == value:
                cat_part[i, j] = 1.0

    X = np.concatenate([numeric_part, cat_part], axis=1)
    y = np.array([float(v) for v in y_raw], dtype=np.float64)

    return X, y
