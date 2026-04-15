
from pathlib import Path

import yaml
from tqdm import tqdm

from src.data.database import db_stream
from src.data.utils import load_config

from .association import AssociationRulesAnalyzer


def load_statistics_bundle(config_path: str = "config.yaml"):
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    stats_path = cfg_path.parent / cfg["data_storage"]["statistics_path"]
    with stats_path.open("r", encoding="utf-8") as f:
        stats = yaml.safe_load(f)
    return cfg, stats


def _quality_thresholds(cfg):
    q = cfg.get("quality") or {}
    th = q.get("thresholds") or {}
    return {
        "missing_frequency": float(th.get("missing_frequency", 0.3)),
        "nonvalid_frequency": float(th.get("nonvalid_frequency", 0.2)),
    }


def run_association_rules(
    config_path: str = "config.yaml",
    *,
    show_progress: bool = True,
):
    cfg_path = Path(config_path).resolve()
    cfg, _ = load_statistics_bundle(config_path)

    analyzer = AssociationRulesAnalyzer.from_config(cfg, config_path=cfg_path)
    batch_size = int(cfg["batch"]["size"])
    stream = db_stream(batch_size=batch_size)

    if show_progress:
        stream = tqdm(stream, desc="Finding association rules...")

    for batch in stream:
        analyzer.update(batch)

    return analyzer.mine_and_report()


class QualityChecker:
    def __init__(
        self,
        stats=None,
        rules_report=None,
        result_path="",
        thresholds=None,
    ):
        self.rules = rules_report or {}
        self.num_stats = stats["numeric_features"]
        self.cat_stats = stats["categorical_features"]
        self.thresholds = thresholds or {
            "missing_frequency": 0.3,
            "nonvalid_frequency": 0.2,
        }
        self.result_path = result_path
        self.report = {}
        self.warnings = []

    @classmethod
    def from_config(
        cls,
        cfg,
        rules_report,
        *,
        stats=None,
        config_path=None,
    ):
        if stats is None:
            if config_path is None:
                raise ValueError("Pass stats=... or config_path=... to load statistics")
            stats_path = Path(config_path).resolve().parent / cfg["data_storage"]["statistics_path"]
            with stats_path.open("r", encoding="utf-8") as f:
                stats = yaml.safe_load(f)

        out = cfg["data_storage"]["quality_path"]
        return cls(
            stats=stats,
            rules_report=rules_report,
            result_path=out,
            thresholds=_quality_thresholds(cfg),
        )

    def analyze_thresholds(self, stats):
        for key, value in stats.items():
            for name, threshold in self.thresholds.items():
                if name in value and value[name] > threshold:
                    self.warnings.append(
                        f"Column {key} has {name} greater than appropriate threshold {threshold}."
                    )

    def analyze_quality(self):
        print("\nAnalyzing data quality...\n")
        self.analyze_thresholds(self.num_stats)
        self.analyze_thresholds(self.cat_stats)
        if len(self.warnings):
            print("WARNING: some requirements were not satisfied.")
        else:
            print("OK: Data satisfies all conditions.")

        merged = dict(self.rules)
        merged["warnings"] = self.warnings
        self.report = merged

    def save_report(self):
        path = Path(self.result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(self.report, f, allow_unicode=True, sort_keys=False)
        print(f"Full quality report was saved in {self.result_path}")


def run_full_quality_pipeline(config_path: str = "config.yaml") -> None:
    """Association rules, threshold checks, write ``quality_path``."""
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
    run_full_quality_pipeline("config.yaml")
