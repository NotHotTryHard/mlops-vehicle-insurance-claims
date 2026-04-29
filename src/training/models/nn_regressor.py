import numpy as np
from sklearn.neural_network import MLPRegressor

from .base import BaseRegressor


class MLPRegressionModel(BaseRegressor):
    """Scikit-learn MLP regressor with the same wrapper API as other models."""

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        lr=5e-4,
        max_epochs=400,
        batch_size=256,
        patience=15,
        val_fraction=0.1,
        random_state=42,
        loss="huber",
        huber_delta=1.0,
        alpha=1e-4,
    ):
        super().__init__()
        self.hidden_layer_sizes = (
            tuple(hidden_layer_sizes)
            if isinstance(hidden_layer_sizes, (list, tuple))
            else hidden_layer_sizes
        )
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.loss = str(loss).lower()
        self.huber_delta = float(huber_delta)
        self.alpha = float(alpha)
        self.model = self._build_model(warm_start=False)
        self._init_kwargs = {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "val_fraction": self.val_fraction,
            "random_state": self.random_state,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "alpha": self.alpha,
        }

    def _build_model(self, *, warm_start):
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.lr,
            max_iter=self.max_epochs,
            early_stopping=True,
            validation_fraction=self.val_fraction,
            n_iter_no_change=self.patience,
            random_state=self.random_state,
            warm_start=warm_start,
        )

    def fit(self, X, y, **kwargs):
        kwargs.pop("cat_features", None)
        continue_training = bool(kwargs.pop("continue_training", False))

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if not continue_training or not hasattr(self.model, "coefs_"):
            self.model = self._build_model(warm_start=False)
        else:
            self.model.warm_start = True
            self.model.max_iter = self.max_epochs

        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)
