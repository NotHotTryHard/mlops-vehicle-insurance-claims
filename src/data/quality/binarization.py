import numpy as np


class NumericBinner:
    def __init__(self, n_bins=10, precision=2):
        self.n_bins = n_bins
        self.bins = {}
        self.precision = precision

    def fit_equal_width(self, stats):
        for column, stat in stats.items():
            self.bins[column] = np.round(
                np.linspace(stat["min"], stat["max"], self.n_bins + 1),
                self.precision,
            )

    def fit_quantile_edges(self, column: str, values: np.ndarray):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            self.bins[column] = np.array([0.0, 1.0])
            return
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(values, qs)
        edges = np.unique(np.round(edges.astype(np.float64), self.precision))
        if edges.size < 2:
            mn = float(np.min(values))
            mx = float(np.max(values))
            if mn == mx:
                self.bins[column] = np.array([mn, mn + 1e-9])
            else:
                self.bins[column] = np.array([mn, mx])
        else:
            self.bins[column] = edges

    def transform_value(self, column, value):
        bins = self.bins[column]
        for i in range(len(bins) - 1):
            lo, hi = float(bins[i]), float(bins[i + 1])
            if lo == hi:
                continue
            last_bin = i == len(bins) - 2
            if last_bin:
                if lo <= value <= hi:
                    return (
                        f"({round(lo, self.precision)} <= {column} <= {round(hi, self.precision)})"
                    )
            elif lo <= value < hi:
                return (
                    f"({round(lo, self.precision)} <= {column} < {round(hi, self.precision)})"
                )

        if value < float(bins[0]):
            return f"({column} < {round(float(bins[0]), self.precision)})"
        return f"({column} > {round(float(bins[-1]), self.precision)})"


class Binner:
    def __init__(self, n_bins=10, missing_values=None):
        self.numeric_binner = NumericBinner(n_bins)
        self.missing_values = missing_values if missing_values is not None else (None, "")

    def fit_equal_width(self, num_stats):
        self.numeric_binner.fit_equal_width(num_stats)

    def fit_quantile_per_column(
        self,
        samples: dict[str, list],
        n_bins: int,
        columns: list[str],
        num_stats: dict | None = None,
    ):
        self.numeric_binner.n_bins = n_bins
        self.numeric_binner.bins = {}
        for column in columns:
            vals = samples.get(column, [])
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size < 2 and num_stats and column in num_stats:
                mn = float(num_stats[column]["min"])
                mx = float(num_stats[column]["max"])
                if mn == mx:
                    arr = np.array([mn, mn + 1e-9], dtype=np.float64)
                else:
                    arr = np.linspace(mn, mx, min(n_bins + 1, 5), dtype=np.float64)
            self.numeric_binner.fit_quantile_edges(column, arr)

    def binarize_row(self, row, num_columns, cat_columns):
        result = []

        for column in num_columns:
            value = row.get(column)
            if value in self.missing_values:
                continue

            try:
                value = float(value)
                result.append(self.numeric_binner.transform_value(column, value))
            except (TypeError, ValueError):
                continue

        for column in cat_columns:
            value = row.get(column)

            if value in self.missing_values:
                continue

            value = str(value).strip()
            result.append(f"{column} = {value}")
        return result
