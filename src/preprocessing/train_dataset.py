from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.utils import load_raw

from .feature_engineering import apply_feature_engineering_rows
from .train_matrix import TrainMatrixPreprocessor, make_train_matrix_preprocessor, matrix_xy_for_model
from .train_target import targets_from_rows


def _rows_after_fe_and_association_rules(cfg: dict, rows: list, config_path: str) -> list:
    """FE + бинарные ``ar_rule_*`` (как при очистке из БД), для пути ``--path-csv``."""
    from src.data.quality.association import augment_train_rows_with_rule_features

    rows = apply_feature_engineering_rows(cfg, rows)
    return augment_train_rows_with_rule_features(cfg, rows, config_path)


def stack_xy_batches(parts_x, parts_y, model_kind: str):
    if not parts_x:
        raise ValueError("No batches to concatenate.")
    y = np.concatenate(parts_y)
    if model_kind == "catboost":
        df = pd.concat(parts_x, ignore_index=True)
        for c in [col for col in df.columns if str(col).startswith("cat__")]:
            if not isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = pd.Series(df[c]).round().astype(np.int32).astype("category")
        return df, y
    return np.vstack([np.asarray(x, dtype=np.float32) for x in parts_x]), y


def materialize_xy_from_cleaned_db(
    cfg: dict,
    model_name: str,
    config_path: str,
    *,
    date_ge=None,
    date_le=None,
    variant_name: Optional[str] = None,
):
    from src.data.quality.clean import stream_cleaned_batches

    target_col = cfg["columns"]["target"]
    preprocessor = make_train_matrix_preprocessor(
        cfg, model_name, config_path=config_path, variant_name=variant_name
    )
    first = True
    parts_x, parts_y = [], []
    for batch in stream_cleaned_batches(config_path, date_ge=date_ge, date_le=date_le):
        if not batch:
            continue
        batch = apply_feature_engineering_rows(cfg, batch)
        if first:
            preprocessor.fit(batch)
            first = False
        yb = targets_from_rows(batch, target_col, cfg)
        Xb, yb2 = matrix_xy_for_model(preprocessor, batch, yb, cfg)
        parts_x.append(Xb)
        parts_y.append(yb2)
    if first:
        raise ValueError(
            "No cleaned rows in DB for the given date filters. "
            "Load data (add_data) and run the quality pipeline first."
        )
    X, y = stack_xy_batches(parts_x, parts_y, model_name)
    return preprocessor, X, y


def load_train_rows_y(
    cfg: dict,
    *,
    config_path: str,
    path_csv: Optional[Path] = None,
    date_until: Optional[str] = None,
) -> tuple[list, np.ndarray]:
    has_csv = path_csv is not None
    has_date = date_until is not None
    if has_csv == has_date:
        raise ValueError("Choose either path_csv or date_until.")
    if has_csv:
        X_rows, y = load_raw(path_csv)
        return _rows_after_fe_and_association_rules(cfg, X_rows, config_path), y
    from src.data.quality.clean import stream_cleaned_batches

    rows_all: list = []
    ys_parts: list[np.ndarray] = []
    target_col = cfg["columns"]["target"]
    for batch in stream_cleaned_batches(config_path, date_le=date_until):
        if not batch:
            continue
        batch = apply_feature_engineering_rows(cfg, batch)
        ys_parts.append(targets_from_rows(batch, target_col, cfg))
        rows_all.extend(batch)
    if not rows_all:
        raise ValueError(
            "No cleaned rows in DB for the given date filters. "
            "Load data (add_data) and run the quality pipeline first."
        )
    return rows_all, np.concatenate(ys_parts)


def build_train_dataset(
    cfg: dict,
    model_name: str,
    *,
    config_path: str = "config.yaml",
    path_csv: Optional[Path] = None,
    date_until: Optional[str] = None,
    variant_name: Optional[str] = None,
    rows: Optional[list] = None,
    y: Optional[np.ndarray] = None,
):
    from_rows = rows is not None and y is not None
    from_path = path_csv is not None or date_until is not None
    if from_rows == from_path:
        raise ValueError("Provide exactly one of: (path_csv XOR date_until) OR (rows AND y).")
    if from_rows:
        preprocessor = make_train_matrix_preprocessor(
            cfg, model_name, config_path=config_path, variant_name=variant_name
        )
        preprocessor.fit(rows)
        return preprocessor, *matrix_xy_for_model(preprocessor, rows, y, cfg)

    if path_csv is not None:
        X_raw, y_raw = load_raw(path_csv)
        X_raw = _rows_after_fe_and_association_rules(cfg, X_raw, config_path)
        preprocessor = make_train_matrix_preprocessor(
            cfg, model_name, config_path=config_path, variant_name=variant_name
        )
        preprocessor.fit(X_raw)
        return preprocessor, *matrix_xy_for_model(preprocessor, X_raw, y_raw, cfg)

    return materialize_xy_from_cleaned_db(
        cfg,
        model_name,
        config_path,
        date_le=date_until,
        variant_name=variant_name,
    )


def build_val_dataset(
    cfg: dict,
    *,
    preprocessor: TrainMatrixPreprocessor,
    config_path: str = "config.yaml",
    path_csv: Optional[Path] = None,
    date_until: Optional[str] = None,
):
    has_csv = path_csv is not None
    has_date = date_until is not None
    if has_csv == has_date:
        raise ValueError("Choose either path_csv or date_until.")

    if has_csv:
        X_raw, y_raw = load_raw(path_csv)
        X_raw = _rows_after_fe_and_association_rules(cfg, X_raw, config_path)
        return matrix_xy_for_model(preprocessor, X_raw, y_raw, cfg)

    from src.data.quality.clean import stream_cleaned_batches

    target_col = cfg["columns"]["target"]
    parts_x, parts_y = [], []
    for batch in stream_cleaned_batches(config_path, date_le=date_until):
        if not batch:
            continue
        batch = apply_feature_engineering_rows(cfg, batch)
        yb = targets_from_rows(batch, target_col, cfg)
        xb, yb2 = matrix_xy_for_model(preprocessor, batch, yb, cfg)
        parts_x.append(xb)
        parts_y.append(yb2)
    if not parts_x:
        raise ValueError(
            "No cleaned rows in DB for validation (check date filter and pipeline)."
        )
    return stack_xy_batches(parts_x, parts_y, preprocessor.model_kind)
