from pathlib import Path

import yaml
from tqdm import tqdm

from src.data.database.db_stream import db_stream
from src.data.utils import get_all_features, load_config
from src.preprocessing.feature_engineering import engineered_numeric_column_names

from .association import (
    augment_row_from_specs,
    binner_and_columns_from_stats,
    load_association_binning,
    load_association_projection,
    load_rule_feature_specs,
    max_rule_features_from_cfg,
)
from .quality_report import quality_thresholds_from_cfg


def _columns_over_thresholds(cfg: dict, stats: dict) -> set:
    thresholds = quality_thresholds_from_cfg(cfg)
    bad = set()
    missing_limit = thresholds["missing_frequency"]
    nonvalid_limit = thresholds["nonvalid_frequency"]
    for col, s in stats.get("numeric_features", {}).items():
        if s.get("missing_frequency", 0) > missing_limit:
            bad.add(col)
        if s.get("nonvalid_frequency", 0) > nonvalid_limit:
            bad.add(col)
    for col, s in stats.get("categorical_features", {}).items():
        if s.get("missing_frequency", 0) > missing_limit:
            bad.add(col)
    drop_z = thresholds.get("drop_if_zero_frequency_above")
    if drop_z is not None:
        skip = set(thresholds.get("zero_frequency_skip_columns") or [])
        for col, s in stats.get("numeric_features", {}).items():
            if col in skip:
                continue
            if (s.get("zero_frequency") or 0) > drop_z:
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
        self._association_projection = None
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
                association_binning = load_association_binning(qp)
                self._association_projection = load_association_projection(qp)
                self._binner, self._binner_num_cols, self._binner_cat_cols = (
                    binner_and_columns_from_stats(
                        cfg,
                        stats,
                        n_bins,
                        missing_values=missing_values,
                        association_binning=association_binning,
                    )
                )

        bad_global = _columns_over_thresholds(cfg, stats)

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
                association_projection=self._association_projection,
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

    def feature_matrix_column_lists(self):
        """Column names for TrainMatrixPreprocessor: kept numerics + rule bins + FE numerics, then cats."""
        num = list(self.kept_numeric)
        for spec in self.rule_feature_specs:
            num.append(spec["feature_name"])
        num.extend(engineered_numeric_column_names(self.cfg))
        cat = list(self.kept_categorical)
        return num, cat


def write_feature_matrix_columns_to_quality_yaml(config_path: str, cleaner: DataCleaner) -> None:
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    qp = cfg_path.parent / cfg["data_storage"]["quality_path"]
    num, cat = cleaner.feature_matrix_column_lists()
    data = {}
    if qp.exists():
        with qp.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    data["feature_matrix_columns"] = {"numeric": num, "categorical": cat}
    qp.parent.mkdir(parents=True, exist_ok=True)
    with qp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def stream_cleaned_batches(
    config_path: str = "config.yaml",
    *,
    date_ge=None,
    date_le=None,
):
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    cleaner = DataCleaner.from_config(str(cfg_path))
    batch_size = int(cfg["batch"]["size"])
    stream = db_stream(
        batch_size=batch_size,
        date_ge=date_ge,
        date_le=date_le,
        config_path=str(cfg_path),
    )
    for batch in stream:
        cleaned = cleaner.clean_batch(batch)
        if cleaned:
            yield cleaned


def run_cleaning_summary(config_path: str = "config.yaml"):
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    c = DataCleaner.from_config(config_path)
    s = c.summary()
    print("Basic cleaning policy:")
    print(f"  Dropped features (over quality thresholds): {s['dropped_features']}")
    print(f"  Kept features: {s['kept_features']}")
    if s.get("association_rule_features"):
        print(f"  Association rule binary features: {s['association_rule_features']}")
    write_feature_matrix_columns_to_quality_yaml(config_path, c)
    print(f"  Wrote feature_matrix_columns to {cfg['data_storage']['quality_path']}")
    return s


if __name__ == "__main__":
    run_cleaning_summary("config.yaml")
