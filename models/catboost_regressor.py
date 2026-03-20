from catboost import CatBoostRegressor
from .base import BaseRegressor

class CatBoostRegressionModel(BaseRegressor):

    def __init__(self, **kwargs):
        params = {
            "loss_function": "Tweedie:variance_power=1.5",
            "verbose": False,
            "random_seed": 42,
        }
        params.update(kwargs)
        self.model = CatBoostRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
