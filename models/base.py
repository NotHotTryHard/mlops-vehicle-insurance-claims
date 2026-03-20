from abc import ABC, abstractmethod

import numpy as np

class BaseRegressor(ABC):
    """Minimal common interface our regressors."""

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
