
from pathlib import Path

import yaml
from src.data.utils import load_config

from .association import AssociationRulesAnalyzer, run_association_passes

REPORT_META_KEY = "report_meta"


def _optional_float(x):
    if x is None:
        return None
    return float(x)

def quality_thresholds_from_cfg(cfg: dict) -> dict:
    quality = cfg.get("quality") or {}
    thresholds = quality.get("stats_thresholds") or quality.get("thresholds") or {}
    target = (cfg.get("columns") or {}).get("target")
    skip_zero = list(thresholds.get("zero_frequency_skip_columns") or [])
    if target and target not in skip_zero:
        skip_zero = [target] + skip_zero
    return {
        "missing_frequency": float(thresholds.get("missing_frequency", 0.3)),
        "nonvalid_frequency": float(thresholds.get("nonvalid_frequency", 0.2)),
        "row_any_missing_frequency": _optional_float(thresholds.get("row_any_missing_frequency")),
        "min_id_uniqueness_ratio": _optional_float(thresholds.get("min_id_uniqueness_ratio")),
        "zero_frequency_warn": _optional_float(thresholds.get("zero_frequency_warn")),
        "zero_frequency_skip_columns": skip_zero,
        "cv_warn": _optional_float(thresholds.get("cv_warn")),
        "min_abs_mean_for_cv": float(thresholds.get("min_abs_mean_for_cv", 1.0e-3)),
        "max_category_dominance": _optional_float(thresholds.get("max_category_dominance")),
        "drop_if_zero_frequency_above": _optional_float(thresholds.get("drop_if_zero_frequency_above")),
    }


def load_statistics_bundle(config_path: str = "config.yaml"):
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    stats_path = cfg_path.parent / cfg["data_storage"]["statistics_path"]
    with stats_path.open("r", encoding="utf-8") as f:
        stats = yaml.safe_load(f)
    return cfg, stats


def load_meta_dict(cfg: dict, config_path: Path) -> dict:
    meta_path = Path(config_path).resolve().parent / cfg["data_storage"]["meta_path"]
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_association_rules(
    config_path: str = "config.yaml",
    *,
    show_progress: bool = True,
):
    cfg_path = Path(config_path).resolve()
    cfg, _ = load_statistics_bundle(config_path)

    analyzer = AssociationRulesAnalyzer.from_config(cfg, config_path=cfg_path)
    run_association_passes(cfg, analyzer, show_progress)
    return analyzer.mine_and_report()


class QualityChecker:
    def __init__(
        self,
        stats=None,
        rules_report=None,
        result_path="",
        thresholds=None,
        meta=None,
        target_column=None,
    ):
        self.rules = rules_report or {}
        self.num_stats = stats["numeric_features"]
        self.cat_stats = stats["categorical_features"]
        self.thresholds = thresholds or quality_thresholds_from_cfg({})
        self.result_path = result_path
        self.meta = meta or {}
        self.target_column = target_column
        self.report = {}
        self.warnings = []
        self.findings = {"meta": [], "numeric": {}, "categorical": {}}

    @classmethod
    def from_config(
        cls,
        cfg,
        rules_report,
        *,
        stats=None,
        config_path=None,
        meta=None,
    ):
        if stats is None:
            if config_path is None:
                raise ValueError("Pass stats=... or config_path=... to load statistics")
            stats_path = Path(config_path).resolve().parent / cfg["data_storage"]["statistics_path"]
            with stats_path.open("r", encoding="utf-8") as f:
                stats = yaml.safe_load(f)
        if meta is None and config_path is not None:
            meta = load_meta_dict(cfg, Path(config_path))

        out = cfg["data_storage"]["quality_path"]
        return cls(
            stats=stats,
            rules_report=rules_report,
            result_path=out,
            thresholds=quality_thresholds_from_cfg(cfg),
            meta=meta,
            target_column=(cfg.get("columns") or {}).get("target"),
        )

    def _warn(self, message: str, *, bucket: str, column: str | None = None, detail=None):
        self.warnings.append(message)
        if bucket == "meta":
            self.findings["meta"].append({"message": message, "detail": detail})
        elif column:
            self.findings[bucket].setdefault(column, []).append(message)

    def _analyze_basic_numeric_and_cat(self):
        """missing / nonvalid only (explicit keys)."""
        thresholds = self.thresholds
        for key, value in self.num_stats.items():
            if self.target_column and key == self.target_column:
                continue
            mf = thresholds["missing_frequency"]
            if value.get("missing_frequency", 0) > mf:
                msg = (
                    f"Column {key} has missing_frequency {value['missing_frequency']} "
                    f"> threshold {mf}."
                )
                self._warn(msg, bucket="numeric", column=key)
            nv = thresholds["nonvalid_frequency"]
            if "nonvalid_frequency" in value and value.get("nonvalid_frequency", 0) > nv:
                msg = (
                    f"Column {key} has nonvalid_frequency {value['nonvalid_frequency']} "
                    f"> threshold {nv}."
                )
                self._warn(msg, bucket="numeric", column=key)

        mf_cat = thresholds["missing_frequency"]
        for key, value in self.cat_stats.items():
            if value.get("missing_frequency", 0) > mf_cat:
                msg = (
                    f"Column {key} has missing_frequency {value['missing_frequency']} "
                    f"> threshold {mf_cat}."
                )
                self._warn(msg, bucket="categorical", column=key)

    def _analyze_meta(self):
        thresholds = self.thresholds
        if not self.meta:
            return
        raf = thresholds.get("row_any_missing_frequency")
        if raf is not None:
            v = self.meta.get("row_any_missing_frequency")
            if v is not None and float(v) > raf:
                self._warn(
                    f"Dataset row_any_missing_frequency {v} > threshold {raf} "
                    "(share of rows with at least one missing cell).",
                    bucket="meta",
                    detail={"value": v, "threshold": raf},
                )
        min_u = thresholds.get("min_id_uniqueness_ratio")
        if min_u is not None and "n_unique_ids" in self.meta and self.meta.get("total_rows"):
            total = int(self.meta["total_rows"])
            if total > 0:
                ratio = int(self.meta["n_unique_ids"]) / total
                if ratio < min_u:
                    self._warn(
                        f"ID uniqueness ratio {ratio:.4f} < threshold {min_u} "
                        f"({self.meta.get('id_column', 'id')}: "
                        f"{self.meta.get('n_unique_ids')} unique / {total} rows).",
                        bucket="meta",
                        detail={"ratio": ratio, "threshold": min_u},
                    )

    def _analyze_numeric_advanced(self):
        thresholds = self.thresholds
        skip_zero = set(thresholds.get("zero_frequency_skip_columns") or [])
        z_warn = thresholds.get("zero_frequency_warn")
        cv_warn = thresholds.get("cv_warn")
        min_mean = thresholds.get("min_abs_mean_for_cv", 1e-3)

        for key, value in self.num_stats.items():
            if self.target_column and key == self.target_column:
                continue
            if z_warn is not None and key not in skip_zero:
                zf = value.get("zero_frequency")
                if zf is not None and float(zf) > z_warn:
                    self._warn(
                        f"Column {key} has zero_frequency {zf} > warn threshold {z_warn} "
                        f"(near-constant zeros).",
                        bucket="numeric",
                        column=key,
                    )

            if cv_warn is not None:
                count = int(value.get("count") or 0)
                sumv = float(value.get("sum") or 0.0)
                std = float(value.get("std") or 0.0)
                if count > 0:
                    m = sumv / count
                    if abs(m) >= min_mean:
                        cv = std / abs(m)
                        if cv > cv_warn:
                            self._warn(
                                f"Column {key} has coefficient of variation std/|mean|={cv:.2f} "
                                f"> {cv_warn} (mean={m:.6g}, std={std:.6g}).",
                                bucket="numeric",
                                column=key,
                            )

    def _analyze_categorical_advanced(self):
        thresholds = self.thresholds
        max_dom = thresholds.get("max_category_dominance")
        if max_dom is None:
            return
        for key, value in self.cat_stats.items():
            freq = value.get("frequency") or {}
            count = int(value.get("count") or 0)
            if not freq or count <= 0:
                continue
            top = max(int(x) for x in freq.values())
            share = top / count
            if share > max_dom:
                self._warn(
                    f"Column {key} is almost constant: top level share {share:.4f} "
                    f"> {max_dom}.",
                    bucket="categorical",
                    column=key,
                )

    def analyze_quality(self):
        print("\nAnalyzing data quality...\n")
        self.warnings = []
        self.findings = {"meta": [], "numeric": {}, "categorical": {}}

        self._analyze_meta()
        self._analyze_basic_numeric_and_cat()
        self._analyze_numeric_advanced()
        self._analyze_categorical_advanced()

        n_warn = len(self.warnings)
        if n_warn:
            print(
                f"WARNING: {n_warn} quality check(s) reported "
                f"(see {REPORT_META_KEY} in report YAML)."
            )
        else:
            print("OK: all configured quality checks passed.")

        merged = dict(self.rules)
        merged[REPORT_META_KEY] = {
            "warnings": self.warnings,
            "findings": self.findings,
            "thresholds_applied": {
                k: v for k, v in self.thresholds.items() if not k.endswith("_columns")
            },
        }
        self.report = merged

    def save_report(self):
        path = Path(self.result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prior = {}
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                prior = yaml.safe_load(f) or {}
        out = dict(self.report)
        if "feature_matrix_columns" in prior and "feature_matrix_columns" not in out:
            out["feature_matrix_columns"] = prior["feature_matrix_columns"]
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(out, f, allow_unicode=True, sort_keys=False)
        print(f"Full quality report was saved in {self.result_path}")


def build_quality_report(config_path: str = "config.yaml") -> None:
    """Mine association rules from DB, run QualityChecker on YAML stats, write ``quality_path``."""
    cfg_path = Path(config_path).resolve()
    cfg, stats = load_statistics_bundle(config_path)

    rules_report = run_association_rules(config_path, show_progress=True)
    checker = QualityChecker.from_config(
        cfg,
        rules_report,
        stats=stats,
        config_path=cfg_path,
    )
    checker.analyze_quality()
    checker.save_report()


if __name__ == "__main__":
    build_quality_report("config.yaml")
