from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.utils import load_config

from .base import BasePreprocessor
from .feature_engineering import apply_feature_engineering_rows


def default_preprocessing():
    return {
        "target_missing_fill": 0.0,
        "default_variant": "catboost_ord",
        "variants": {
            "catboost_ord": {
                "numeric": {"impute": "median", "scale": False},
                "categorical": {"impute": "most_frequent", "encode": "ordinal"},
            },
            "mlp_ohe": {
                "numeric": {"impute": "median", "scale": True},
                "categorical": {"impute": "most_frequent", "encode": "onehot"},
            },
            "mlp_ord": {
                "numeric": {"impute": "median", "scale": True},
                "categorical": {"impute": "most_frequent", "encode": "ordinal"},
            },
        },
    }


def merge_preprocessing_from_config(cfg):
    out = default_preprocessing()
    user = cfg.get("preprocessing") or {}
    for key, val in user.items():
        if key == "variants":
            continue
        out[key] = val
    out["variants"] = {**out["variants"], **(user.get("variants") or {})}
    return out


def defaults_for_model(model_kind):
    if model_kind == "mlp":
        return {
            "numeric": {"impute": "median", "scale": True},
            "categorical": {"impute": "most_frequent", "encode": "onehot"},
        }
    return {
        "numeric": {"impute": "median", "scale": False},
        "categorical": {"impute": "most_frequent", "encode": "ordinal"},
    }


def variant_with_defaults(model_kind, variant):
    base = defaults_for_model(model_kind)
    if not variant:
        return base
    merged = dict(base)
    for key in ("numeric", "categorical"):
        if key in variant:
            merged[key] = {**base.get(key, {}), **variant[key]}
    return merged


def target_fill(cfg):
    prep = cfg.get("preprocessing") or {}
    return float(prep.get("target_missing_fill", 0.0))


def target_value_or_fill(value, fill):
    """One target cell: use number, or ``fill`` if missing / bad."""
    if value is None:
        return fill
    if isinstance(value, str) and value.strip() == "":
        return fill
    try:
        x = float(value)
        if not np.isfinite(x):
            return fill
        return x
    except (TypeError, ValueError):
        return fill


def target_vector_from_rows(rows, target_col, cfg):
    fill = target_fill(cfg)
    return np.array(
        [target_value_or_fill(row.get(target_col), fill) for row in rows],
        dtype=np.float32,
    )


def fill_target_y(y, cfg):
    fill = target_fill(cfg)
    a = np.asarray(y, dtype=np.float64)
    out = np.where(np.isfinite(a), a, fill)
    return out.astype(np.float32)


def load_feature_matrix_columns(config_path="config.yaml"):
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    qp = cfg_path.parent / cfg["data_storage"]["quality_path"]
    if not qp.exists():
        raise ValueError(
            f"Missing quality file {qp}. Run the quality pipeline (or run_cleaning_summary) first."
        )
    with qp.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    block = data.get("feature_matrix_columns")
    if not block:
        raise ValueError(
            f"feature_matrix_columns missing in {qp}. Run run_cleaning_summary / quality pipeline first."
        )
    return list(block["numeric"]), list(block["categorical"])


class TrainMatrixPreprocessor(BasePreprocessor):
    def __init__(self, cfg, *, model_kind, variant=None, num_cols=None, cat_cols=None):
        self.cfg = cfg
        self.model_kind = model_kind
        self.variant = variant_with_defaults(model_kind, variant)
        if num_cols is None or cat_cols is None:
            raise ValueError("num_cols and cat_cols are required (use make_train_matrix_preprocessor).")
        self._num_cols = list(num_cols)
        self._cat_cols = list(cat_cols)
        self._column_transformer = None
        self.catboost_cat_indices = None

    def _rows_to_frame(self, X):
        df = pd.DataFrame(X)
        for c in self._num_cols + self._cat_cols:
            if c not in df.columns:
                df[c] = np.nan
        for c in self._num_cols:
            # invalid text -> NaN, then imputer fixes NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _build(self):
        nv = self.variant["numeric"]
        cv = self.variant["categorical"]
        num_impute = nv.get("impute", "median")
        scale = bool(nv.get("scale", self.model_kind == "mlp"))

        num_steps = [("imputer", SimpleImputer(strategy=num_impute))]
        if scale:
            num_steps.append(("scaler", StandardScaler()))
        num_pipe = Pipeline(num_steps)

        cat_impute = cv.get("impute", "most_frequent")
        encode = cv.get("encode", "onehot" if self.model_kind == "mlp" else "ordinal")

        if encode == "onehot":
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy=cat_impute)),
                    (
                        "enc",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            max_categories=50,
                        ),
                    ),
                ]
            )
            self.catboost_cat_indices = None
        elif encode == "ordinal":
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy=cat_impute)),
                    (
                        "enc",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            dtype=np.int64,
                        ),
                    ),
                ]
            )
            n_num = len(self._num_cols)
            self.catboost_cat_indices = list(
                range(n_num, n_num + len(self._cat_cols))
            )
        else:
            raise ValueError(f"Unknown categorical encode: {encode}")

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, self._num_cols),
                ("cat", cat_pipe, self._cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0,
        )

    def fit(self, X):
        df = self._rows_to_frame(X)
        use_cols = self._num_cols + self._cat_cols
        self._column_transformer = self._build()
        self._column_transformer.fit(df[use_cols])
        return self

    def transform(self, X):
        if self._column_transformer is None:
            raise RuntimeError("Call fit before transform.")
        df = self._rows_to_frame(X)
        use_cols = self._num_cols + self._cat_cols
        out = self._column_transformer.transform(df[use_cols])
        return np.asarray(out, dtype=np.float32)

    def transform_frame(self, X):
        """DataFrame of features; category dtype on ordinal cols when model is catboost."""
        arr = self.transform(X)
        names = list(self._column_transformer.get_feature_names_out())
        df = pd.DataFrame(arr, columns=names)
        if self.model_kind == "catboost" and self.catboost_cat_indices:
            for i in self.catboost_cat_indices:
                c = names[i]
                df[c] = pd.Series(df[c]).round().astype(np.int32).astype("category")
        return df


def cat_features_from_frame(df):
    return [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]


def concat_xy_batches(parts_x, parts_y, model_kind):
    """Merge batch outputs from ``features_xy_for_model`` (CatBoost DataFrame vs MLP ndarray)."""
    if not parts_x:
        raise ValueError("No batches to concatenate.")
    y = np.concatenate(parts_y)
    if model_kind == "catboost":
        return pd.concat(parts_x, ignore_index=True), y
    stacked = np.vstack([np.asarray(x, dtype=np.float32) for x in parts_x])
    return stacked, y


def features_xy_for_model(preprocessor, X_rows, y, cfg):
    """
    Same preprocessing for train and validation: target fill, then ``transform_frame``.
    Row dicts must already include feature-engineering columns if enabled in config (apply
    ``apply_feature_engineering_rows`` before this when training/val match the streaming pipeline).
    """
    y = fill_target_y(y, cfg)
    X = preprocessor.transform_frame(X_rows)
    if preprocessor.model_kind == "catboost":
        return X, y
    return X.values.astype(np.float32), y


def accumulate_xy_from_cleaned_db(
    config_path="config.yaml",
    model_name="catboost",
    *,
    date_ge=None,
    date_le=None,
):
    """
    Train path: cleaned DB batches → FE → fit ``TrainMatrixPreprocessor`` on first batch →
    concatenate ``(X, y)`` like a single in-memory train set.
    """
    from src.data.quality.clean import stream_cleaned_batches

    cfg = load_config(Path(config_path).resolve())
    target_col = cfg["columns"]["target"]
    preprocessor = make_train_matrix_preprocessor(cfg, model_name, config_path=config_path)
    first = True
    parts_x, parts_y = [], []
    for batch in stream_cleaned_batches(
        config_path, date_ge=date_ge, date_le=date_le
    ):
        if not batch:
            continue
        batch = apply_feature_engineering_rows(cfg, batch)
        if first:
            preprocessor.fit(batch)
            first = False
        y = target_vector_from_rows(batch, target_col, cfg)
        Xb, yb = features_xy_for_model(preprocessor, batch, y, cfg)
        parts_x.append(Xb)
        parts_y.append(yb)
    if first:
        raise ValueError(
            "No cleaned rows in DB for the given date filters. "
            "Load data (add_data) and run the quality pipeline first."
        )
    X, y = concat_xy_batches(parts_x, parts_y, model_name)
    return preprocessor, X, y


def make_train_matrix_preprocessor(cfg, model_name, config_path="config.yaml"):
    prep = merge_preprocessing_from_config(cfg)
    vname = prep["default_variant"]
    spec = prep["variants"][vname]
    num_cols, cat_cols = load_feature_matrix_columns(config_path)
    return TrainMatrixPreprocessor(
        cfg,
        model_kind=model_name,
        variant=spec,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )


def stream_full_train_pipeline(config_path="config.yaml", model_name="catboost"):
    from src.data.quality.pipeline import stream_analysis_and_cleaning_pipeline

    cfg = load_config(Path(config_path).resolve())
    target_col = cfg["columns"]["target"]
    preprocessor = make_train_matrix_preprocessor(cfg, model_name, config_path=config_path)
    first = True
    for batch in stream_analysis_and_cleaning_pipeline(config_path):
        if not batch:
            continue
        batch = apply_feature_engineering_rows(cfg, batch)
        if first:
            preprocessor.fit(batch)
            first = False
        y = target_vector_from_rows(batch, target_col, cfg)
        yield features_xy_for_model(preprocessor, batch, y, cfg)


def stream_full_val_pipeline(
    config_path="config.yaml",
    *,
    preprocessor,
    date_ge=None,
    date_le=None,
):
    """
    Validation streaming: cleaned DB batches only (``stream_cleaned_batches``), no quality report
    or cleaning-summary step. Pass a **fitted** preprocessor (e.g. from a saved train bundle).

    Yields the same ``(X, y)`` shape as ``features_xy_for_model`` / ``val_call`` (CatBoost
    DataFrame vs MLP ndarray).
    """
    from src.data.quality.clean import stream_cleaned_batches

    cfg = load_config(Path(config_path).resolve())
    target_col = cfg["columns"]["target"]
    for batch in stream_cleaned_batches(
        config_path, date_ge=date_ge, date_le=date_le
    ):
        if not batch:
            continue
        batch = apply_feature_engineering_rows(cfg, batch)
        y = target_vector_from_rows(batch, target_col, cfg)
        yield features_xy_for_model(preprocessor, batch, y, cfg)


def accumulate_xy_val_from_cleaned_db(
    config_path="config.yaml",
    *,
    preprocessor,
    date_ge=None,
    date_le=None,
):
    """Concatenate validation batches (preprocessor loaded from a saved train bundle)."""
    parts_x, parts_y = [], []
    for X, y in stream_full_val_pipeline(
        config_path,
        preprocessor=preprocessor,
        date_ge=date_ge,
        date_le=date_le,
    ):
        parts_x.append(X)
        parts_y.append(y)
    if not parts_x:
        raise ValueError(
            "No cleaned rows in DB for validation (check date filter and pipeline)."
        )
    return concat_xy_batches(parts_x, parts_y, preprocessor.model_kind)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    mname = sys.argv[2] if len(sys.argv) > 2 else "catboost"
    for _step in stream_full_train_pipeline(path, mname):
        import pdb

        pdb.set_trace()
