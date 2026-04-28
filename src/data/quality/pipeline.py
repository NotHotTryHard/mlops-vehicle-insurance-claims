from pathlib import Path

from .clean import run_cleaning_summary, stream_cleaned_batches as _stream_cleaned_batches
from .quality_report import build_quality_report


def refresh_quality_artifacts(config_path: str = "config.yaml") -> None:
    cfg_path = Path(config_path).resolve()
    print("=== Quality report (association rules + thresholds) ===\n")
    build_quality_report(str(cfg_path))
    print("\n=== Cleaning policy ===\n")
    run_cleaning_summary(str(cfg_path))


def iter_cleaned_batches(
    config_path: str = "config.yaml",
    *,
    refresh_quality: bool = True,
):
    if refresh_quality:
        refresh_quality_artifacts(config_path)
        print()
    yield from _stream_cleaned_batches(config_path)


if __name__ == "__main__":
    from src.data.database import ensure_db

    ensure_db()
    refresh_quality_artifacts("config.yaml")
