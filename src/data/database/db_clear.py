import argparse
import sys
from pathlib import Path

from src.data.utils import load_config


def db_clear(config_path="config.yaml"):
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    db_path = cfg_path.parent / cfg["data_storage"]["data_path"]

    if db_path.exists():
        db_path.unlink()
        print(f"Database deleted: {db_path}")

    stats_path = cfg_path.parent / cfg["data_storage"]["statistics_path"]
    if stats_path.exists():
        stats_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    db_clear(config_path=args.config)
