import pandas as pd
from catboost import CatBoostRegressor, Pool

from .base import BaseRegressor


class CatBoostRegressionModel(BaseRegressor):

    def __init__(self, **kwargs):
        super().__init__()
        params = {
            "loss_function": "RMSE",
            "verbose": False,
            "random_seed": 42,
            "iterations": 500,
        }
        params.update(kwargs)
        self.model = CatBoostRegressor(**params)

    def fit(self, X, y, cat_features=None, **kwargs):
        if cat_features:
            self.model.fit(X, y, cat_features=cat_features)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            model_cat_idx = list(self.model.get_cat_feature_indices())
            if model_cat_idx:
                cat_features = [X.columns[i] for i in model_cat_idx if i < len(X.columns)]
                return self.model.predict(Pool(X, cat_features=cat_features))
            # Backward compatibility: old models trained without cat_features.
            if any(isinstance(X[c].dtype, pd.CategoricalDtype) for c in X.columns):
                X = X.copy()
                for c in X.columns:
                    if isinstance(X[c].dtype, pd.CategoricalDtype):
                        X[c] = X[c].cat.codes.astype("float32")
        return self.model.predict(X)
