import numpy as np

from .base import BasePreprocessor


def feature_engineering_config(cfg):
    prep = cfg.get("preprocessing") or {}
    return prep.get("feature_engineering") or {}


def engineered_numeric_column_names(cfg):
    block = feature_engineering_config(cfg)
    if not block.get("enabled"):
        return []
    out = []
    for col in block.get("log1p") or []:
        out.append(f"log1p_{col}")
    for spec in block.get("ratios") or []:
        out.append(spec["name"])
    for spec in block.get("differences") or []:
        out.append(spec["name"])
    return out


def apply_feature_engineering_rows(cfg, rows):
    return FeatureEngineeringTransformer(cfg).transform(rows)


class FeatureEngineeringTransformer(BasePreprocessor):
    def __init__(self, cfg):
        self.cfg = feature_engineering_config(cfg)

    def fit(self, X):
        return self

    def transform(self, X):
        if not self.cfg.get("enabled"):
            return X

        out = []
        for row in X:
            new_row = dict(row)

            for col in self.cfg.get("log1p", []) or []:
                key = f"log1p_{col}"
                try:
                    v = float(row.get(col))
                    new_row[key] = float(np.log1p(max(v, 0.0)))
                except (TypeError, ValueError):
                    pass

            for spec in self.cfg.get("ratios", []) or []:
                name = spec["name"]
                num_c = spec["numerator"]
                den_c = spec["denominator"]
                eps = float(spec.get("eps", 1e-6))
                try:
                    num = float(row.get(num_c))
                    den = float(row.get(den_c))
                    new_row[name] = num / (den + eps) if den + eps != 0 else 0.0
                except (TypeError, ValueError):
                    pass

            for spec in self.cfg.get("differences", []) or []:
                name = spec["name"]
                a = spec["a"]
                b = spec["b"]
                try:
                    new_row[name] = float(row.get(a)) - float(row.get(b))
                except (TypeError, ValueError):
                    pass

            out.append(new_row)
        return out


class AugmentAndNumericPreprocessor(BasePreprocessor):
    def __init__(self, cfg):
        self._fe = FeatureEngineeringTransformer(cfg)
        from .numeric_only import NumericOnlyPreprocessor

        self._num = NumericOnlyPreprocessor()

    def fit(self, X):
        X1 = self._fe.fit_transform(X)
        self._num.fit(X1)
        return self

    def transform(self, X):
        X1 = self._fe.transform(X)
        return self._num.transform(X1)
