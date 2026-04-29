from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class ModelDriftPolicyError(RuntimeError):
    """Raised when training.model_drift.fail_on blocks val after performance regression."""


_STATUS_RANK = {"ok": 0, "skipped": 0, "no_baseline": 0, "warn": 1, "critical": 2}


def model_drift_settings(cfg: dict) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": False,
        "metric": "RMSE",
        "warn_ratio": 1.1,
        "critical_ratio": 1.25,
        "fail_on": None,
        "report_path": "session/reports/model_drift_report.yaml",
        "history_path": "session/reports/model_metrics_history.yaml",
    }
    block = (cfg.get("training") or {}).get("model_drift") or {}
    for key, val in block.items():
        if val is not None:
            defaults[key] = val
    return defaults


def _stress_ratio(metric: str, current: float, baseline: float) -> float | None:
    """Single number >=1 means worse performance vs baseline (for thresholding)."""
    m = metric.upper()
    if m in ("RMSE", "RMSLE"):
        if baseline <= 1e-12:
            return None
        return float(current) / float(baseline)
    if m == "R2":
        c, b = float(current), float(baseline)
        if c >= b:
            return 1.0
        if b <= 0:
            return None
        if c <= 0:
            return float("inf")
        return float(b) / c
    return None


def assess_model_drift(
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not settings.get("enabled"):
        return {"status": "skipped", "reason": "training.model_drift.enabled is false"}

    metric = str(settings.get("metric", "RMSE")).upper()
    if not baseline_metrics or metric not in baseline_metrics:
        return {
            "status": "no_baseline",
            "metric": metric,
            "message": "No baseline metrics in model bundle (train first).",
        }
    if metric not in current_metrics:
        return {
            "status": "no_baseline",
            "metric": metric,
            "message": f"Current evaluation has no {metric}.",
        }

    cur = float(current_metrics[metric])
    base = float(baseline_metrics[metric])
    wr = float(settings.get("warn_ratio", 1.1))
    cr = float(settings.get("critical_ratio", 1.25))
    ratio = _stress_ratio(metric, cur, base)

    if ratio is None:
        status = "ok"
        detail = "Cannot compute ratio (degenerate baseline); treat as ok."
    elif ratio >= cr:
        status = "critical"
        detail = f"{metric} stress ratio {ratio:.4f} >= critical_ratio {cr}"
    elif ratio >= wr:
        status = "warn"
        detail = f"{metric} stress ratio {ratio:.4f} >= warn_ratio {wr}"
    else:
        status = "ok"
        detail = f"{metric} within tolerance (ratio={ratio:.4f})."

    return {
        "status": status,
        "metric": metric,
        "baseline": base,
        "current": cur,
        "stress_ratio": ratio,
        "warn_ratio": wr,
        "critical_ratio": cr,
        "detail": detail,
    }


def _yaml_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _history_append(root: Path, rel: str, entry: dict[str, Any]) -> None:
    path = root / rel
    rows: list = []
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            rows = yaml.safe_load(f) or []
    if not isinstance(rows, list):
        rows = []
    rows.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(rows, f, allow_unicode=True, sort_keys=False)


def append_metrics_history_entry(
    cfg: dict,
    project_root: Path,
    entry: dict[str, Any],
) -> None:
    settings = model_drift_settings(cfg)
    if not settings.get("enabled"):
        return
    rel = str(settings.get("history_path", "session/reports/model_metrics_history.yaml"))
    row = dict(entry)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    _history_append(project_root, rel, row)


def enforce_model_drift_policy(report: dict[str, Any], settings: dict[str, Any]) -> None:
    if not settings.get("enabled"):
        return
    status = str(report.get("status", "ok"))
    if status in ("skipped", "no_baseline"):
        return
    fail_on = settings.get("fail_on")
    if fail_on is None:
        return
    need = _STATUS_RANK[fail_on]
    got = _STATUS_RANK.get(status, 0)
    if got >= need:
        raise ModelDriftPolicyError(
            f"Model drift status={status!r} violates training.model_drift.fail_on={fail_on!r}: "
            f"{report.get('detail', '')}"
        )


def record_val_model_drift(
    cfg: dict,
    project_root: Path,
    *,
    model_bundle: str,
    baseline_metrics: dict[str, Any] | None,
    current_metrics: dict[str, Any],
    data_note: str | None = None,
) -> dict[str, Any]:
    settings = model_drift_settings(cfg)
    drift = assess_model_drift(current_metrics, baseline_metrics, settings)
    ts = datetime.now().isoformat(timespec="seconds")
    report: dict[str, Any] = {
        "timestamp": ts,
        "mode": "val",
        "model_bundle": model_bundle,
        "current_metrics": dict(current_metrics),
        "training_baseline_metrics": dict(baseline_metrics) if baseline_metrics else None,
        "data": data_note,
        "model_drift": drift,
    }
    if settings.get("enabled"):
        rel = str(settings.get("report_path", "session/reports/model_drift_report.yaml"))
        _yaml_dump(project_root / rel, report)
        print(
            f"Model drift report: {project_root / rel} (status={drift.get('status')})",
            flush=True,
        )
        print(f"[model drift] {drift.get('detail', '')}", flush=True)
        hist = {
            "timestamp": ts,
            "phase": "val",
            "model_bundle": model_bundle,
            "metrics": dict(current_metrics),
            "model_drift": drift,
            "data": data_note,
        }
        _history_append(project_root, str(settings["history_path"]), hist)
    enforce_model_drift_policy(drift, settings)
    return report
