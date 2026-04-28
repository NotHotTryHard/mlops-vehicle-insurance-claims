import pickle
from collections import Counter
from datetime import datetime
from math import sqrt
from pathlib import Path

import yaml

class DataStatsGlobalAnalyzer:
    def __init__(
        self,
        cfg,
        missing_values=(None, ""),
        round_precision=3,
        dt_col="INSR_BEGIN", 
        dt_fmt="%d-%b-%y"
    ):
        self.meta_analyzer = DataMetaAnalyzer(
            dt_col=dt_col,
            dt_fmt=dt_fmt,
            result_path=cfg["data_storage"]["meta_path"],
            id_col=cfg["columns"].get("id"),
            round_precision=round_precision,
        )
        self.stats_analyzer = DataStatsAnalyzer(cfg, missing_values=missing_values, round_precision=round_precision)

    def merge_existing_reports(self, cfg, root: Path) -> None:
        stat_path = root / cfg["data_storage"]["statistics_path"]
        meta_path = root / cfg["data_storage"]["meta_path"]
        self.stats_analyzer.merge_from_yaml(stat_path)
        self.meta_analyzer.merge_from_yaml(meta_path)

    def update(self, batch):
        self.meta_analyzer.update(batch)
        self.stats_analyzer.update(batch)

    def save_report(self):
        self.meta_analyzer.save_report()
        self.stats_analyzer.save_report()
        print(f"Meta information was saved into {self.meta_analyzer.result_path}")
        print(f"Statistics was saved into {self.stats_analyzer.result_path}")

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
                "std": 0.0,
                "welford_mean": 0.0,
                "welford_m2": 0.0,
                "zero_count": 0,
                "zero_frequency": 0.0,
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

    def merge_from_yaml(self, path: Path) -> None:
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        num = data.get("numeric_features") or {}
        for col in self.num_features:
            prev_stats = num.get(col)
            if not prev_stats:
                continue
            stats = self.num_stats[col]
            stats["count"] = int(prev_stats.get("count", 0) or 0)
            stats["sum"] = float(prev_stats.get("sum", 0) or 0.0)
            min_value, max_value = prev_stats.get("min"), prev_stats.get("max")
            stats["min"] = float(min_value) if min_value is not None else None
            stats["max"] = float(max_value) if max_value is not None else None
            stats["missing"] = int(prev_stats.get("missing", 0) or 0)
            stats["nonvalid"] = int(prev_stats.get("nonvalid", 0) or 0)
            stats["zero_count"] = int(prev_stats.get("zero_count", 0) or 0)
            welford_mean = prev_stats.get("welford_mean")
            welford_m2 = prev_stats.get("welford_m2")
            if welford_mean is not None and welford_m2 is not None:
                stats["welford_mean"] = float(welford_mean)
                stats["welford_m2"] = float(welford_m2)
            else:
                # legacy yaml: std only; can't merge variance correctly
                stats["welford_mean"] = float(stats["sum"] / stats["count"]) if stats["count"] else 0.0
                stats["welford_m2"] = 0.0
            # these values are calculated only in _finalize
            stats["mean"] = 0.0
            stats["std"] = 0.0
            stats["zero_frequency"] = 0.0
            stats["missing_frequency"] = 0.0
            stats["nonvalid_frequency"] = 0.0
        cat = data.get("categorical_features") or {}
        for col in self.cat_features:
            prev_stats = cat.get(col)
            if not prev_stats:
                continue
            stats = self.cat_stats[col]
            stats["missing"] = int(prev_stats.get("missing", 0) or 0)
            stats["missing_frequency"] = float(prev_stats.get("missing_frequency", 0.0) or 0.0)
            stats["count"] = int(prev_stats.get("count", 0) or 0)
            freq = prev_stats.get("frequency", {})
            stats["frequency"] = Counter({str(k): int(v) for k, v in freq.items()})
        self.result_stats = {}

    def _numeric_update(self, column, value):
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
        if value == 0.0:
            stats["zero_count"] += 1
        n = stats["count"]
        delta = value - stats["welford_mean"]
        stats["welford_mean"] += delta / n
        delta2 = value - stats["welford_mean"]
        stats["welford_m2"] += delta * delta2

    def _categorical_update(self, column, value):
        stats = self.cat_stats[column]

        if value is None:
            stats["missing"] += 1
            return
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value in self.missing_values:
            stats["missing"] += 1
            return
        
        stats["count"] += 1
        stats["frequency"][value] += 1


    def update(self, batch):
        for _, row in batch:
            for column in self.num_features:
                self._numeric_update(column, row.get(column))
               
            for column in self.cat_features:
                self._categorical_update(column, row.get(column))

    def _finalize(self):
        # if already finalized
        if len(self.result_stats):
            return

        for column in self.num_features:
            stats = self.num_stats[column]
            total = stats["count"] + stats["missing"] + stats["nonvalid"]
            n_valid = stats["count"]
            if n_valid:
                stats["mean"] = round(stats["sum"] / n_valid, self.precision)
                if n_valid > 1:
                    var = stats["welford_m2"] / (n_valid - 1)
                    stats["std"] = round(sqrt(var), self.precision) if var > 0 else 0.0
                else:
                    stats["std"] = 0.0
                stats["zero_frequency"] = round(stats["zero_count"] / n_valid, self.precision)
            else:
                stats["mean"] = 0.0
                stats["std"] = 0.0
                stats["zero_frequency"] = 0.0
            if total:
                stats["missing_frequency"] = round(stats["missing"] / total, self.precision)
                stats["nonvalid_frequency"] = round(stats["nonvalid"] / total, self.precision)
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
            total = stats["count"] + stats["missing"]
            if total:
                stats["missing_frequency"] = round(stats["missing"] / total, self.precision)
            else:
                stats["missing_frequency"] = 1.0
        self.result_stats = {
            "numeric_features": self.num_stats,
            "categorical_features": self.cat_stats,
        }


    def _save(self):
        out = Path(self.result_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.dump(self.result_stats, f, allow_unicode=True, sort_keys=False)
        print(f"Feature statistics was saved in {self.result_path}")

    def save_report(self):
        self._finalize()
        self._save()


class DataMetaAnalyzer:
    def __init__(
        self,
        dt_col="INSR_BEGIN",
        dt_fmt="%d-%b-%y",
        missing_values=None,
        result_path=None,
        id_col=None,
        round_precision=3,
    ):
        self.dt_col = dt_col
        self.dt_fmt = dt_fmt
        self.result_path = result_path
        self.missing_values = missing_values if missing_values is not None else [None, ""]
        self.id_col = id_col
        self.precision = round_precision

        self._initialize_stats()

    def _unique_ids_pickle_path(self) -> Path | None:
        if not self.result_path:
            return None
        p = Path(self.result_path)
        return p.with_name(p.stem + "_unique_ids.pkl")

    def _initialize_stats(self):
        self.total_rows = 0
        self.total_missing = 0
        self.rows_with_any_missing = 0
        self._unique_ids = set()
        self._unique_months = set()
        self.min_date = None
        self.max_date = None
        self.loaded_at = datetime.now().isoformat()

    def merge_from_yaml(self, path: Path) -> None:
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.total_rows = int(data.get("total_rows", 0) or 0)
        self.total_missing = int(data.get("total_missing", 0) or 0)
        self.rows_with_any_missing = int(data.get("rows_with_any_missing", 0) or 0)
        for s in data.get("unique_months") or []:
            if not s or not isinstance(s, str):
                continue
            parts = s.split("-", 1)
            if len(parts) == 2:
                try:
                    self._unique_months.add((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
        min_date, max_date = data.get("min_date"), data.get("max_date")
        if min_date:
            self.min_date = datetime.fromisoformat(str(min_date))
        if max_date:
            self.max_date = datetime.fromisoformat(str(max_date))
        if data.get("loaded_at"):
            self.loaded_at = str(data["loaded_at"])
        pkl = self._unique_ids_pickle_path()
        if pkl and pkl.is_file():
            with pkl.open("rb") as fp:
                self._unique_ids = pickle.load(fp)

    def update(self, batch):
        n = len(batch)
        self.total_rows += n
        self.total_missing += sum(
            1 for _, row in batch for v in row.values() if v in self.missing_values
        )
        for _, row in batch:
            if any(v in self.missing_values for v in row.values()):
                self.rows_with_any_missing += 1
            if self.id_col:
                rid = row.get(self.id_col)
                if rid not in self.missing_values:
                    self._unique_ids.add(str(rid).strip())

        dates = []
        for _, row in batch:
            raw = row.get(self.dt_col)
            if raw in self.missing_values:
                continue
            try:
                if isinstance(raw, str):
                    raw = raw.strip()
                d = datetime.strptime(str(raw), self.dt_fmt)
                dates.append(d)
                self._unique_months.add((d.year, d.month))
            except (TypeError, ValueError):
                continue

        if dates:
            batch_min = min(dates)
            batch_max = max(dates)
            self.min_date = batch_min if self.min_date is None else min(self.min_date, batch_min)
            self.max_date = batch_max if self.max_date is None else max(self.max_date, batch_max)

    def to_dict(self):
        row_miss_freq = 0.0
        if self.total_rows:
            row_miss_freq = round(self.rows_with_any_missing / self.total_rows, self.precision)
        months_sorted = sorted(self._unique_months)
        unique_months = [f"{y}-{m:02d}" for y, m in months_sorted]
        out = {
            "total_rows": self.total_rows,
            "total_missing": self.total_missing,
            "rows_with_any_missing": self.rows_with_any_missing,
            "row_any_missing_frequency": row_miss_freq,
            "n_unique_months": len(self._unique_months),
            "unique_months": unique_months,
            "min_date": self.min_date.isoformat() if self.min_date else None,
            "max_date": self.max_date.isoformat() if self.max_date else None,
            "loaded_at": self.loaded_at,
        }
        if self.id_col:
            out["n_unique_ids"] = len(self._unique_ids)
            out["id_column"] = self.id_col
        return out
    
    def __str__(self):
        report = "\nData Meta Parameters\n"
        for key, value in self.to_dict().items():
            report += f"{key}: {value}\n"
        return report

    def save_report(self):
        out = Path(self.result_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
        pkl = self._unique_ids_pickle_path()
        if pkl and self.id_col:
            with pkl.open("wb") as fp:
                pickle.dump(self._unique_ids, fp, protocol=pickle.HIGHEST_PROTOCOL)
