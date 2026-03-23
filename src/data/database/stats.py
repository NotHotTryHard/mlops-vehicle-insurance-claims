from collections import Counter, defaultdict
import yaml


class DataStatsAnalyzer:
    def __init__(self, cfg, missing_values=(None, ""), round_precision=3):
        self.missing_values = missing_values
        self.precision = round_precision
        self.result_path = cfg["data_storage"]["statistics_path"]

        features = cfg["columns"]["features"]
        self.cat_features = features["categorical"]
        self.num_features = features["numeric"] + [cfg["columns"]["target"]]
        
        self.num_stats = {
            col: {
                "count": 0,
                "sum": 0.0,
                "min": None,
                "max": None,
                "mean": 0.0,
                "missing": 0,
                "missing_frequency": 0.0,
                "nonvalid": 0,
                "nonvalid_frequency": 0.0,
            }
            for col in self.num_features
        }

        self.cat_stats = {
            col: {
                "count": 0,
                "frequency": Counter(),
                "missing": 0,
                "missing_frequency": 0.0,
            }
            for col in self.cat_features
        }
        self.result_stats = {}

    def numeric_update(self, column, value):
        stats = self.num_stats[column]

        if value in self.missing_values:
            stats["missing"] += 1
            return

        try:
            value = float(value)
        except:
            stats["nonvalid"] += 1
            return

        stats["count"] += 1
        stats["sum"] += value
        stats["min"] = value if stats["min"] is None else min(stats["min"], value)
        stats["max"] = value if stats["max"] is None else max(stats["max"], value)
        
    def categorical_update(self, column, value):
        stats = self.cat_stats[column]

        value = value.strip()
        if value in self.missing_values:
            stats["missing"] += 1
            return
        
        stats["count"] += 1
        stats["frequency"][value] += 1


    def update(self, batch):
        for _, row in batch:
            for column in self.num_features:
                self.numeric_update(column, row.get(column))
               
            for column in self.cat_features:
                self.categorical_update(column, row.get(column))

    def finalize_stats(self):
        for column in self.num_features:
            stats = self.num_stats[column]
            if stats["count"]:
                stats["mean"] = round(stats["max"] / stats["count"], self.precision)
                stats["missing_frequency"] = round(stats["missing"] / stats["count"], self.precision)
                stats["nonvalid_frequency"] = round(stats["nonvalid"] / stats["count"], self.precision)
            else:
                stats["missing_frequency"] = 1.0
        for column in self.cat_features:
            stats = self.cat_stats[column]
            # sort values in descending order by frequency
            sorted_freq = dict(sorted(
                stats["frequency"].items(),
                key=lambda x: x[1],
                reverse=True,
            ))
            stats["frequency"] = sorted_freq
        self.result_stats = {
            "numeric_features": self.num_stats,
            "categorical_features": self.cat_stats,
        }


    def save_stats(self):
        with open(self.result_path, "w") as f:
            yaml.dump(self.result_stats, f, allow_unicode=True, sort_keys=False) 
        print(f"Feature statistics was saved in {self.result_path}")
