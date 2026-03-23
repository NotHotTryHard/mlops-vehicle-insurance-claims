import numpy as np

class NumericBinner:
    def __init__(self, n_bins=10, precision=2):
        self.n_bins = n_bins
        self.bins = {}
        self.precision = precision

    def fit(self, stats):
        for column, stat in stats.items():
            self.bins[column] = np.round(np.linspace(stat["min"], stat["max"], self.n_bins + 1))

    def transform_value(self, column, value):
        bins = self.bins[column]

        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                return f"{round(float(bins[i]), self.precision)} <= {column} < {round(float(bins[i + 1]), self.precision)})"
        if value < bins[0]:
            return f"{column} < {round(float(bins[0]), self.precision)}"
        else:
            return f"{column} > {round(float(bins[-1]), self.precision)}"


class Binner:
    def __init__(self, n_bins=10, missing_values = (None, "")):
        self.numeric_binner = NumericBinner(n_bins)
        self.missing_values = missing_values
    
    def fit(self, num_stats):
        self.numeric_binner.fit(num_stats)

    def binarize_row(self, row, num_columns, cat_columns):
        result = []

        for column in num_columns:
            value = row.get(column)
            if value in self.missing_values:
                continue

            try:
                value = float(value)
                result.append(self.numeric_binner.transform_value(column, value))
            except:
                continue
    
        for column in cat_columns:
            value = row.get(value)

            if value in self.missing_values:
                continue
            
            value = value.strip()
            result.append(f"{column} = {value}")
        return result
