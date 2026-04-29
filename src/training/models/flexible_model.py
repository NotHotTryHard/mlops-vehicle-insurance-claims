from typing import Any

import numpy as np

from src.data.utils import get_all_features

_DEFAULTS: dict[str, Any] = {
    "prefer_catboost_if_n_rows_below": 8000,
    "prefer_catboost_if_row_missing_fraction_above": 0.4,
    "prefer_catboost_if_numeric_outlier_row_fraction_above": 0.3,
    "iqr_multiplier": 4.0,
    "default_if_clean": "mlp",
}


def merged_flexible_settings(cfg: dict) -> dict[str, Any]:
    """Defaults + optional `training.flexible_model` overrides (thresholds only).

    CLI --new auto turns selection on; YAML is not required.
    """
    out = dict(_DEFAULTS)
    block = (cfg.get("training") or {}).get("flexible_model") or {}
    for key, val in block.items():
        if val is not None and key != "enabled":
            out[key] = val
    return out


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def diagnose_raw_rows(rows: list, cfg: dict, flex: dict[str, Any]) -> dict[str, Any]:
    features = get_all_features(cfg)
    num_cols = list(cfg["columns"]["features"]["numeric"])
    k = float(flex.get("iqr_multiplier", 4.0))
    n = len(rows)
    if n == 0:
        return {
            "n_rows": 0,
            "missing_row_fraction": 1.0,
            "outlier_row_fraction": 0.0,
        }

    missing_rows = 0
    for row in rows:
        if any(_is_missing_value(row.get(f)) for f in features):
            missing_rows += 1

    col_values: dict[str, list[float]] = {c: [] for c in num_cols}
    for row in rows:
        for c in num_cols:
            raw = row.get(c)
            if _is_missing_value(raw):
                continue
            try:
                col_values[c].append(float(raw))
            except (TypeError, ValueError):
                continue

    bounds: dict[str, tuple[float, float] | None] = {}
    for c in num_cols:
        arr = np.asarray(col_values[c], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 4:
            bounds[c] = None
            continue
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = float(q3 - q1)
        if iqr <= 0.0:
            bounds[c] = None
        else:
            bounds[c] = (float(q1 - k * iqr), float(q3 + k * iqr))

    outlier_rows = 0
    for row in rows:
        flagged = False
        for c in num_cols:
            b = bounds.get(c)
            if b is None:
                continue
            raw = row.get(c)
            if _is_missing_value(raw):
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v) or v < b[0] or v > b[1]:
                flagged = True
                break
        if flagged:
            outlier_rows += 1

    return {
        "n_rows": n,
        "missing_row_fraction": float(missing_rows / n),
        "outlier_row_fraction": float(outlier_rows / n),
    }


def diagnose_and_choose(
    rows: list,
    cfg: dict,
    flex: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """
    Returns (model_key, reason_code, diagnosis).

    reason_code: small_sample | sparse_missing | numeric_outliers | clean_default
    """
    d = diagnose_raw_rows(rows, cfg, flex)
    n = d["n_rows"]
    if n < int(flex.get("prefer_catboost_if_n_rows_below", 8000)):
        return "catboost", "small_sample", d
    if d["missing_row_fraction"] > float(
        flex.get("prefer_catboost_if_row_missing_fraction_above", 0.4)
    ):
        return "catboost", "sparse_missing", d
    if d["outlier_row_fraction"] > float(
        flex.get("prefer_catboost_if_numeric_outlier_row_fraction_above", 0.3)
    ):
        return "catboost", "numeric_outliers", d
    default = str(flex.get("default_if_clean", "mlp")).lower()
    if default not in ("catboost", "mlp"):
        default = "mlp"
    return default, "clean_default", d
