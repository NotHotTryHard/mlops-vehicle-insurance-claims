import numpy as np


def target_missing_fill(cfg: dict) -> float:
    return float((cfg.get("preprocessing") or {}).get("target_missing_fill", 0.0))


def scalar_target(value, fill: float) -> float:
    if value is None:
        return fill
    if isinstance(value, str) and value.strip() == "":
        return fill
    try:
        x = float(value)
        return fill if not np.isfinite(x) else x
    except (TypeError, ValueError):
        return fill


def targets_from_rows(rows: list, target_col: str, cfg: dict) -> np.ndarray:
    fill = target_missing_fill(cfg)
    return np.array(
        [scalar_target(row.get(target_col), fill) for row in rows],
        dtype=np.float32,
    )


def sanitize_target_array(y, cfg: dict) -> np.ndarray:
    fill = target_missing_fill(cfg)
    a = np.asarray(y, dtype=np.float64)
    return np.where(np.isfinite(a), a, fill).astype(np.float32)
