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
        candidates = set(X[0].keys())
        for row in X:
            candidates -= {k for k in candidates if not _is_numeric(row[k])}
            if not candidates:
                break
        self._cols = [k for k in X[0] if k in candidates]
        return self

    def transform(self, X):
        return np.array(
            [[float(row[c]) for c in self._cols] for row in X],
            dtype=np.float32,
        )
