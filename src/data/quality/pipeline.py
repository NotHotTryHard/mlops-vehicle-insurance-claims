from pathlib import Path

from src.data.utils import load_config

from .clean import run_cleaning_summary, stream_cleaned_batches
from .quality_report import run_full_quality_pipeline


def stream_analysis_and_cleaning_pipeline(
    config_path: str = "config.yaml",
):
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    pl = cfg.get("pipeline") or {}
    print("=== Quality report (association rules + thresholds) ===\n")
    run_full_quality_pipeline(str(cfg_path))
    print("\n=== Cleaning policy ===\n")
    run_cleaning_summary(str(cfg_path))
    print()

    yield from stream_cleaned_batches(config_path)
