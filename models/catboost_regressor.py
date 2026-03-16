from catboost import CatBoostRegressor

class CatBoostRegressionModel:

    def __init__(self, **kwargs):
        params = {
            "loss_function": "RMSE",
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
