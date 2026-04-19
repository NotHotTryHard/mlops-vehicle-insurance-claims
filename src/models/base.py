from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class BaseRegressor(ABC):

    def __init__(self):
        self.metrics: dict | None = None
        self.trained_at: str | None = None

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def train(self, X, y, test_size=0.2, random_state=42, **fit_kwargs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        y_train_log = self._transform_target(y_train)
        self.fit(X_train, y_train_log, **fit_kwargs)
        self._resid_var = float(np.var(y_train_log - self.predict(X_train)))
        self.metrics = self.evaluate(X_test, y_test)
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
