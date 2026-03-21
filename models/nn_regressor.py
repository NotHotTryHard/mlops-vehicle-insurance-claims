from sklearn.neural_network import MLPRegressor

from .base import BaseRegressor


class MLPRegressionModel(BaseRegressor):

    def __init__(self, **kwargs):
        default_params = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "random_state": 42,
        }
        default_params.update(kwargs)
        self.model = MLPRegressor(**default_params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
