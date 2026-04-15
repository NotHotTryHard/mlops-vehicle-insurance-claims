from pathlib import Path

import yaml
from tqdm import tqdm

from src.data.database import db_stream
from src.data.utils import get_all_features, load_config

from .association import (
    augment_row_from_specs,
    binner_and_columns_from_stats,
    load_rule_feature_specs,
    max_rule_features_from_cfg,
)


def _quality_thresholds(cfg: dict):
    q = cfg.get("quality") or {}
    th = q.get("thresholds") or {}
    return (
        float(th.get("missing_frequency", 0.3)),
        float(th.get("nonvalid_frequency", 0.2)),
    )


def _columns_over_thresholds(stats, mf_th, nv_th):
    # Column names (from saved statistics) that fail global quality thresholds.
    bad = set()
    for col, s in stats.get("numeric_features", {}).items():
        if s.get("missing_frequency", 0) > mf_th:
            bad.add(col)
        if s.get("nonvalid_frequency", 0) > nv_th:
            bad.add(col)
    for col, s in stats.get("categorical_features", {}).items():
        if s.get("missing_frequency", 0) > mf_th:
            bad.add(col)
    return bad


class DataCleaner:
    """Builds a cleaning policy from ``config.yaml`` and ``statistics_path``."""

    def __init__(self, cfg, stats, *, config_path=None, missing_values=(None, "")):
        self.cfg = cfg
        self.stats = stats
        self.missing_values = missing_values
        self.config_path = config_path

        self.target = cfg["columns"]["target"]
        self.id_col = cfg["columns"].get("id")

        self.rule_feature_specs = []
        self._binner = None
        self._binner_num_cols = None
        self._binner_cat_cols = None
        q = cfg.get("quality") or {}
        assoc = q.get("association") or {}
        if assoc.get("add_rule_features", True) and config_path:
            qp = Path(config_path).resolve().parent / cfg["data_storage"]["quality_path"]
            max_rf = max_rule_features_from_cfg(cfg)
            self.rule_feature_specs = load_rule_feature_specs(qp, max_n=max_rf)
            if self.rule_feature_specs:
                n_bins = int(assoc.get("n_bins", 10))
                self._binner, self._binner_num_cols, self._binner_cat_cols = (
                    binner_and_columns_from_stats(
                        cfg, stats, n_bins, missing_values=missing_values
                    )
                )

        mf_th, nv_th = _quality_thresholds(cfg)
        bad_global = _columns_over_thresholds(stats, mf_th, nv_th)

        feature_names = get_all_features(cfg)
        self._feature_set = set(feature_names)

        # Drop only features that appear in config and fail thresholds in statistics.
        self.dropped_features = bad_global & self._feature_set
        self.kept_features = [
            c for c in feature_names if c not in self.dropped_features
        ]

        nums = set(cfg["columns"]["features"]["numeric"])
        self.kept_numeric = [c for c in self.kept_features if c in nums]
        self.kept_categorical = [c for c in self.kept_features if c not in nums]

    def _is_missing(self, value):
        if value is None:
            return True
        if value in self.missing_values:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    def keep_row(self, row):
        """Whether the row is usable for training / downstream steps."""
        t = row.get(self.target)
        if self._is_missing(t):
            return False
        try:
            float(t)
        except (TypeError, ValueError):
            return False

        for col in self.kept_numeric:
            v = row.get(col)
            if self._is_missing(v):
                return False
            try:
                float(v)
            except (TypeError, ValueError):
                return False

        for col in self.kept_categorical:
            v = row.get(col)
            if self._is_missing(v):
                return False

        return True

    def project_row(self, row):
        """Keep only columns needed after cleaning (features, target, optional id/datetime)."""
        keys = list(self.kept_features) + [self.target]
        if self.id_col:
            keys.append(self.id_col)
        dt = self.cfg["columns"].get("datetime")
        if dt:
            keys.append(dt)
        out = {}
        for k in keys:
            if k in row:
                out[k] = row[k]
        return out

    def clean_row(self, row):
        if not self.keep_row(row):
            return None
        out = self.project_row(row)
        # Antecedent checks use the same binning as mining (full raw row).
        if self._binner is not None and self.rule_feature_specs:
            extra = augment_row_from_specs(
                row,
                self.rule_feature_specs,
                self._binner,
                self._binner_num_cols,
                self._binner_cat_cols,
            )
            out.update(extra)
        return out

    def clean_batch(self, batch):
        out = []
        for row in batch:
            cr = self.clean_row(row)
            if cr is not None:
                out.append(cr)
        return out

    @classmethod
    def from_config(cls, config_path: str = "config.yaml", *, stats=None):
        cfg_path = Path(config_path).resolve()
        cfg = load_config(cfg_path)
        if stats is None:
            sp = cfg_path.parent / cfg["data_storage"]["statistics_path"]
            with sp.open("r", encoding="utf-8") as f:
                stats = yaml.safe_load(f)
        return cls(cfg, stats, config_path=str(cfg_path))

    def summary(self):
        s = {
            "dropped_features": sorted(self.dropped_features),
            "kept_features": self.kept_features,
            "target": self.target,
        }
        if self.rule_feature_specs:
            s["association_rule_features"] = [x["feature_name"] for x in self.rule_feature_specs]
        return s


def stream_cleaned_batches(config_path: str = "config.yaml", *, show_progress: bool = True):
    """Yield batches from the DB after row filtering and column projection."""
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    cleaner = DataCleaner.from_config(str(cfg_path))
    batch_size = int(cfg["batch"]["size"])
    stream = db_stream(batch_size=batch_size, config_path=str(cfg_path))
    if show_progress:
        stream = tqdm(stream, desc="Clean batches")
    for batch in stream:
        cleaned = cleaner.clean_batch(batch)
        if cleaned:
            yield cleaned


def run_cleaning_summary(config_path: str = "config.yaml"):
    c = DataCleaner.from_config(config_path)
    s = c.summary()
    print("Basic cleaning policy:")
    print(f"  Dropped features (over quality thresholds): {s['dropped_features']}")
    print(f"  Kept features: {s['kept_features']}")
    if s.get("association_rule_features"):
        print(f"  Association rule binary features: {s['association_rule_features']}")
    return s


if __name__ == "__main__":
    # Usage: python -m src.data.quality.clean [config.yaml]
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_cleaning_summary(cfg)
