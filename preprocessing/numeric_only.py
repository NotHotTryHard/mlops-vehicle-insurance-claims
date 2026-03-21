import numpy as np

from .base import BasePreprocessor


def _is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


class NumericOnlyPreprocessor(BasePreprocessor):

    def fit(self, X):
        self._cols = [k for k, v in X[0].items() if _is_numeric(v)]
        return self

    def transform(self, X):
        return np.array(
            [[float(row[c]) for c in self._cols] for row in X],
            dtype=np.float32,
        )
