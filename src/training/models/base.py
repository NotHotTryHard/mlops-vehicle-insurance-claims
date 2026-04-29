from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from tqdm import tqdm


def _subset_xy(X, y, idx):
    idx = np.asarray(idx)
    y_arr = np.asarray(y)
    y_sub = y_arr[idx]
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx], y_sub
    return np.asarray(X)[idx], y_sub


def _aggregate_fold_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    out: dict = {}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    out["n_folds"] = len(rows)
    return out


class BaseRegressor(ABC):

    def __init__(self):
        self.metrics: dict | None = None
        self.trained_at: str | None = None
        self._init_kwargs: dict = {}

    def _fresh_clone(self):
        return type(self)(**dict(self._init_kwargs))

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def _fit_eval_split(self, X_train, y_train, X_val, y_val, **fit_kwargs):
        y_train_log = self._transform_target(y_train)
        self.fit(X_train, y_train_log, **fit_kwargs)
        self._resid_var = float(np.var(y_train_log - self.predict(X_train)))
        return self.evaluate(X_val, y_val)

    def _train_holdout(self, X, y, v: dict, **fit_kwargs) -> dict:
        test_size = float(v.get("test_size", 0.2))
        random_state = int(v.get("random_state", 42))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self._fit_eval_split(X_train, y_train, X_test, y_test, **fit_kwargs)

    def _train_kfold_cv(self, X, y, v: dict, **fit_kwargs) -> dict:
        n_splits = max(2, int(v.get("n_splits", 5)))
        random_state = int(v.get("random_state", 42))
        shuffle = bool(v.get("shuffle", True))
        n = int(len(np.asarray(y)))
        if n < n_splits:
            print(
                f"[train] CV fallback: n_samples={n} < n_splits={n_splits}; using holdout.",
                flush=True,
            )
            return self._train_holdout(X, y, v, **fit_kwargs)
        splitter = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        print(
            f"[train] {n_splits}-fold CV (kfold, shuffle={shuffle})",
            flush=True,
        )
        fold_rows: list[dict] = []
        with tqdm(
            splitter.split(np.arange(n)),
            total=n_splits,
            desc="kfold",
            unit="fold",
        ) as pbar:
            for fold_i, (tr_idx, va_idx) in enumerate(pbar):
                m = self._fresh_clone()
                X_tr, y_tr = _subset_xy(X, y, tr_idx)
                X_va, y_va = _subset_xy(X, y, va_idx)
                row = m._fit_eval_split(X_tr, y_tr, X_va, y_va, **fit_kwargs)
                fold_rows.append(row)
                pbar.set_postfix(
                    fold=fold_i + 1,
                    RMSE=f"{row['RMSE']:.4f}",
                    R2=f"{row['R2']:.4f}",
                )
        metrics = _aggregate_fold_metrics(fold_rows)
        print("[train] kfold: refit on full dataset...", flush=True)
        y_log = self._transform_target(y)
        self.fit(X, y_log, **fit_kwargs)
        self._resid_var = float(np.var(y_log - self.predict(X)))
        return metrics

    def _train_time_series_cv(self, X, y, v: dict, **fit_kwargs) -> dict:
        n_splits = max(2, int(v.get("n_splits", 5)))
        n = int(len(np.asarray(y)))
        if n < n_splits + 1:
            print(
                f"[train] TimeSeriesSplit fallback: n_samples={n} too small for "
                f"n_splits={n_splits}; using holdout.",
                flush=True,
            )
            return self._train_holdout(X, y, v, **fit_kwargs)
        splitter = TimeSeriesSplit(n_splits=n_splits)
        print(f"[train] TimeSeriesSplit (n_splits={n_splits})", flush=True)
        fold_rows: list[dict] = []
        with tqdm(
            splitter.split(np.arange(n)),
            total=n_splits,
            desc="time_series",
            unit="fold",
        ) as pbar:
            for fold_i, (tr_idx, va_idx) in enumerate(pbar):
                m = self._fresh_clone()
                X_tr, y_tr = _subset_xy(X, y, tr_idx)
                X_va, y_va = _subset_xy(X, y, va_idx)
                row = m._fit_eval_split(X_tr, y_tr, X_va, y_va, **fit_kwargs)
                fold_rows.append(row)
                pbar.set_postfix(
                    fold=fold_i + 1,
                    n_train=len(tr_idx),
                    n_val=len(va_idx),
                    RMSE=f"{row['RMSE']:.4f}",
                )
        metrics = _aggregate_fold_metrics(fold_rows)
        print("[train] time_series: refit on full dataset...", flush=True)
        y_log = self._transform_target(y)
        self.fit(X, y_log, **fit_kwargs)
        self._resid_var = float(np.var(y_log - self.predict(X)))
        return metrics

    def train(self, X, y, *, validation=None, **fit_kwargs):
        v = dict(validation or {})
        v.setdefault("type", "holdout")
        v.setdefault("test_size", 0.2)
        v.setdefault("random_state", 42)
        v.setdefault("n_splits", 5)
        v.setdefault("shuffle", True)
        kind = str(v.get("type", "holdout")).lower().replace("-", "_")
        if kind in ("timeseries", "time_series"):
            kind = "time_series"
        if kind not in ("holdout", "kfold", "time_series"):
            kind = "holdout"
        if kind == "holdout":
            self.metrics = self._train_holdout(X, y, v, **fit_kwargs)
        elif kind == "kfold":
            self.metrics = self._train_kfold_cv(X, y, v, **fit_kwargs)
        else:
            self.metrics = self._train_time_series_cv(X, y, v, **fit_kwargs)
        self.trained_at = datetime.now().isoformat(timespec="seconds")
        return self.metrics

    def update(self, X, y, **fit_kwargs):
        y_log = self._transform_target(y)
        self.fit(X, y_log, continue_training=True, **fit_kwargs)
        self._resid_var = float(np.var(y_log - self.predict(X)))
        self.metrics = self.evaluate(X, y)
        self.trained_at = datetime.now().isoformat(timespec="seconds")
        return self.metrics

    def evaluate(self, X, y):
        preds = self._inverse_transform_target(self.predict(X))
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        rmsle = float(np.sqrt(np.mean((np.log1p(preds.clip(0)) - np.log1p(y)) ** 2)))
        r2 = float(r2_score(y, preds))
        return {"RMSE": rmse, "RMSLE": rmsle, "R2": r2}

    def _transform_target(self, y):
        return np.log1p(y)

    def _inverse_transform_target(self, y_pred):
        return np.expm1(y_pred + getattr(self, "_resid_var", 0.0) / 2)
