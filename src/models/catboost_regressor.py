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
        self._init_kwargs = dict(params)
        self.model = CatBoostRegressor(**params)

    def fit(self, X, y, cat_features=None, **kwargs):
        continue_training = bool(kwargs.pop("continue_training", False))
        fit_kwargs = dict(kwargs)
        if continue_training:
            try:
                if self.model.is_fitted():
                    fit_kwargs["init_model"] = self.model
            except Exception:
                pass
        if cat_features:
            self.model.fit(X, y, cat_features=cat_features, **fit_kwargs)
        else:
            self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            model_cat_idx = list(self.model.get_cat_feature_indices())
            if model_cat_idx:
                cat_features = [X.columns[i] for i in model_cat_idx if i < len(X.columns)]
                return self.model.predict(Pool(X, cat_features=cat_features))
        return self.model.predict(X)
