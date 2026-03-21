import argparse
import sys
from pathlib import Path

try:
    from database.utils import load_config
except ImportError:
    from utils import load_config  # без этого не запускается python database/db_clear.py
                                   # так как в sys.path добавляется database, а не корень


def db_clear(config_path="config.yaml"):
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    db_path = cfg_path.parent / cfg["data_storage"]["data_path"]

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    db_path.unlink()
    print(f"Database deleted: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    db_clear(config_path=args.config)
