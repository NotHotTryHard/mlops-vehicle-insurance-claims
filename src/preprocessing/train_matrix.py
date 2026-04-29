from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .base import BasePreprocessor
from .preprocess_config import (
    feature_matrix_column_names,
    preprocess_block,
    resolve_variant_key,
)
from .train_target import sanitize_target_array


def defaults_for_model(model_kind: str) -> dict:
    if model_kind == "mlp":
        return {
            "numeric": {"impute": "median", "scale": True},
            "categorical": {"impute": "most_frequent", "encode": "onehot"},
        }
    return {
        "numeric": {"impute": "median", "scale": False},
        "categorical": {"impute": "most_frequent", "encode": "ordinal"},
    }


def variant_with_defaults(model_kind: str, variant: Optional[dict]) -> dict:
    base = defaults_for_model(model_kind)
    if not variant:
        return base
    merged = dict(base)
    for key in ("numeric", "categorical"):
        if key in variant:
            merged[key] = {**base.get(key, {}), **variant[key]}
    return merged


class TrainMatrixPreprocessor(BasePreprocessor):
    def __init__(self, cfg, *, model_kind, variant=None, num_cols=None, cat_cols=None):
        self.cfg = cfg
        self.model_kind = model_kind
        self.variant = variant_with_defaults(model_kind, variant)
        if num_cols is None or cat_cols is None:
            raise ValueError(
                "num_cols and cat_cols are required (use make_train_matrix_preprocessor)."
            )
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
            self.catboost_cat_indices = list(range(n_num, n_num + len(self._cat_cols)))
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
        arr = self.transform(X)
        names = list(self._column_transformer.get_feature_names_out())
        df = pd.DataFrame(arr, columns=names)
        if self.model_kind == "catboost":
            for c in [c for c in names if c.startswith("cat__")]:
                df[c] = pd.Series(df[c]).round().astype(np.int32).astype("category")
        return df


def cat_features_from_frame(df: pd.DataFrame) -> list[str]:
    cat_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    if cat_cols:
        return cat_cols
    return [c for c in df.columns if str(c).startswith("cat__")]


def matrix_xy_for_model(preprocessor: TrainMatrixPreprocessor, X_rows, y, cfg: dict):
    y = sanitize_target_array(y, cfg)
    X = preprocessor.transform_frame(X_rows)
    if preprocessor.model_kind == "catboost":
        return X, y
    return X.values.astype(np.float32), y


def make_train_matrix_preprocessor(
    cfg: dict,
    model_name: str,
    config_path: str = "config.yaml",
    *,
    variant_name: Optional[str] = None,
) -> TrainMatrixPreprocessor:
    prep = preprocess_block(cfg)
    vkey = resolve_variant_key(cfg, model_name, variant_name=variant_name)
    spec = prep["variants"][vkey]
    num_cols, cat_cols = feature_matrix_column_names(config_path)
    out = TrainMatrixPreprocessor(
        cfg,
        model_kind=model_name,
        variant=spec,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    out.variant_key = vkey
    return out
