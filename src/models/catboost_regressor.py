from catboost import CatBoostRegressor

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
        return self.model.predict(X)
