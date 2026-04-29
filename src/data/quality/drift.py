import math
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.data.utils import load_config, quality_round_precision

_EPS = 1e-9

_DRIFT_STATUS_RANK = {"ok": 0, "warn": 1, "critical": 2}


class DataDriftPolicyError(RuntimeError):
    """Raised when quality.drift.fail_on (or incomplete-check) blocks the pipeline."""


def _default_drift_settings() -> dict[str, Any]:
    return {
        "reference_path": "session/reports/drift_reference.yaml",
        "report_path": "session/reports/drift_report.yaml",
        "run_check_after_add_data": False,
        "fail_on": None,
        "fail_on_incomplete": False,
        "mean_shift_sigma_warn": 2.0,
        "mean_shift_sigma_critical": 4.0,
        "std_ratio_warn_low": 0.5,
        "std_ratio_warn_high": 2.0,
        "std_ratio_critical_low": 0.25,
        "std_ratio_critical_high": 4.0,
        "missing_frequency_delta_warn": 0.08,
        "missing_frequency_delta_critical": 0.15,
        "categorical_jsd_warn": 0.12,
        "categorical_jsd_critical": 0.22,
    }


def drift_settings(config):
    merged_settings = _default_drift_settings()
    quality_section = config.get("quality") or {}
    raw_drift_section = dict(quality_section.get("drift") or {})
    if (
        "run_check_after_add_data" not in raw_drift_section
        and "run_after_add_data" in raw_drift_section
    ):
        raw_drift_section["run_check_after_add_data"] = raw_drift_section.get(
            "run_after_add_data"
        )
    raw_drift_section.pop("run_after_add_data", None)
    for setting_name, setting_value in raw_drift_section.items():
        if setting_name in merged_settings and setting_value is not None:
            merged_settings[setting_name] = setting_value
    return merged_settings


def _resolve_drift_file_paths(
    resolved_config_path: Path, config: dict
) -> tuple[Path, Path, Path]:
    project_root = resolved_config_path.parent
    data_storage = config["data_storage"]
    current_statistics_path = project_root / data_storage["statistics_path"]
    drift_configuration = drift_settings(config)
    reference_statistics_path = project_root / drift_configuration["reference_path"]
    drift_report_output_path = project_root / drift_configuration["report_path"]
    return current_statistics_path, reference_statistics_path, drift_report_output_path


def load_statistics_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def freeze_drift_reference(config_path: str = "config.yaml") -> Path | None:
    resolved_config_path = Path(config_path).resolve()
    config = load_config(resolved_config_path)
    current_statistics_path, reference_statistics_path, _ = _resolve_drift_file_paths(
        resolved_config_path, config
    )
    if not current_statistics_path.is_file():
        print(
            f"Drift reference: no statistics at {current_statistics_path}; nothing to freeze."
        )
        return None
    reference_statistics_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_statistics_path, reference_statistics_path)
    print(f"Drift reference saved: {reference_statistics_path}")
    return reference_statistics_path


def _kullback_leibler_divergence(p: list[float], q: list[float]) -> float:
    divergence_sum = 0.0
    for p_i, q_i in zip(p, q, strict=True):
        divergence_sum += p_i * math.log((p_i + _EPS) / (q_i + _EPS))
    return divergence_sum


def _jensen_shannon_divergence(
    reference_freq: dict[str, int],
    current_freq: dict[str, int],
) -> float:
    sorted_category_union = sorted(set(reference_freq) | set(current_freq))
    if not sorted_category_union:
        return 0.0
    reference_total_count = sum(reference_freq.get(category, 0) for category in sorted_category_union)
    current_total_count = sum(current_freq.get(category, 0) for category in sorted_category_union)
    if reference_total_count <= 0 or current_total_count <= 0:
        return 0.0
    reference_probabilities = [
        reference_freq.get(category, 0) / reference_total_count for category in sorted_category_union
    ]
    current_probabilities = [
        current_freq.get(category, 0) / current_total_count for category in sorted_category_union
    ]
    mixture_probabilities = [
        (p_ref + p_cur) * 0.5
        for p_ref, p_cur in zip(reference_probabilities, current_probabilities, strict=True)
    ]
    return 0.5 * _kullback_leibler_divergence(reference_probabilities, mixture_probabilities) + 0.5 * _kullback_leibler_divergence(
        current_probabilities, mixture_probabilities
    )


def _drift_severity_label(has_warning: bool, has_critical: bool) -> str:
    if has_critical:
        return "critical"
    if has_warning:
        return "warn"
    return "ok"


def _rate_numeric_column_drift(
    reference_mean: float,
    reference_std: float,
    current_mean: float,
    current_std: float,
    reference_missing_frequency: float,
    current_missing_frequency: float,
    drift_thresholds: dict[str, Any],
    *,
    decimals: int,
) -> dict[str, Any]:
    mean_shift_in_sigmas = abs(current_mean - reference_mean) / max(reference_std, _EPS)
    if reference_std > _EPS:
        std_ratio = current_std / max(reference_std, _EPS)
    else:
        std_ratio = 1.0 if current_std <= _EPS else float("inf")
    missing_frequency_delta = abs(current_missing_frequency - reference_missing_frequency)
    is_critical = (
        mean_shift_in_sigmas >= drift_thresholds["mean_shift_sigma_critical"]
        or std_ratio <= drift_thresholds["std_ratio_critical_low"]
        or std_ratio >= drift_thresholds["std_ratio_critical_high"]
        or missing_frequency_delta >= drift_thresholds["missing_frequency_delta_critical"]
    )
    has_warning = (
        not is_critical
        and (
            mean_shift_in_sigmas >= drift_thresholds["mean_shift_sigma_warn"]
            or std_ratio <= drift_thresholds["std_ratio_warn_low"]
            or std_ratio >= drift_thresholds["std_ratio_warn_high"]
            or missing_frequency_delta >= drift_thresholds["missing_frequency_delta_warn"]
        )
    )
    return {
        "mean_shift_sigmas": round(mean_shift_in_sigmas, decimals),
        "std_ratio": round(std_ratio, decimals) if math.isfinite(std_ratio) else None,
        "missing_frequency_delta": round(missing_frequency_delta, decimals),
        "severity": _drift_severity_label(has_warning, is_critical),
    }


def _compare_reference_and_current_statistics(
    reference_stats: dict[str, Any],
    current_stats: dict[str, Any],
    drift_thresholds: dict[str, Any],
    *,
    excluded_column_names: set[str],
    decimals: int,
) -> dict[str, Any]:
    reference_numeric_features = reference_stats.get("numeric_features") or {}
    current_numeric_features = current_stats.get("numeric_features") or {}
    reference_categorical_features = reference_stats.get("categorical_features") or {}
    current_categorical_features = current_stats.get("categorical_features") or {}

    numeric_metrics_by_column: dict[str, Any] = {}
    for column_name in sorted(
        set(reference_numeric_features) & set(current_numeric_features)
    ):
        if column_name in excluded_column_names:
            continue
        ref_stats = reference_numeric_features[column_name]
        cur_stats = current_numeric_features[column_name]
        try:
            reference_mean = float(ref_stats.get("mean", 0) or 0.0)
            reference_std = float(ref_stats.get("std", 0) or 0.0)
            current_mean = float(cur_stats.get("mean", 0) or 0.0)
            current_std = float(cur_stats.get("std", 0) or 0.0)
            reference_missing_frequency = float(ref_stats.get("missing_frequency", 0) or 0.0)
            current_missing_frequency = float(cur_stats.get("missing_frequency", 0) or 0.0)
        except (TypeError, ValueError):
            continue
        numeric_metrics_by_column[column_name] = _rate_numeric_column_drift(
            reference_mean,
            reference_std,
            current_mean,
            current_std,
            reference_missing_frequency,
            current_missing_frequency,
            drift_thresholds,
            decimals=decimals,
        )

    categorical_metrics_by_column: dict[str, Any] = {}
    for column_name in sorted(
        set(reference_categorical_features) & set(current_categorical_features)
    ):
        if column_name in excluded_column_names:
            continue
        ref_stats = reference_categorical_features[column_name]
        cur_stats = current_categorical_features[column_name]
        reference_freq = {
            str(level_name): int(count)
            for level_name, count in (ref_stats.get("frequency") or {}).items()
        }
        current_freq = {
            str(level_name): int(count)
            for level_name, count in (cur_stats.get("frequency") or {}).items()
        }
        jensen_shannon_divergence = _jensen_shannon_divergence(reference_freq, current_freq)
        reference_missing_frequency = float(ref_stats.get("missing_frequency", 0) or 0.0)
        current_missing_frequency = float(cur_stats.get("missing_frequency", 0) or 0.0)
        missing_frequency_delta = abs(current_missing_frequency - reference_missing_frequency)
        is_critical = (
            jensen_shannon_divergence >= drift_thresholds["categorical_jsd_critical"]
            or missing_frequency_delta
            >= drift_thresholds["missing_frequency_delta_critical"]
        )
        has_warning = (
            not is_critical
            and (
                jensen_shannon_divergence >= drift_thresholds["categorical_jsd_warn"]
                or missing_frequency_delta >= drift_thresholds["missing_frequency_delta_warn"]
            )
        )
        categorical_metrics_by_column[column_name] = {
            "js_divergence": round(jensen_shannon_divergence, decimals),
            "missing_frequency_delta": round(missing_frequency_delta, decimals),
            "severity": _drift_severity_label(has_warning, is_critical),
        }

    return {"numeric": numeric_metrics_by_column, "categorical": categorical_metrics_by_column}


def _normalize_fail_on(value: Any) -> str | None:
    if value in (None, "", False):
        return None
    text = str(value).strip().lower()
    if text in ("none", "off", "false", "no"):
        return None
    if text in ("warn", "warning", "warnings"):
        return "warn"
    if text in ("critical", "crit"):
        return "critical"
    return None


def enforce_drift_policy(
    report: dict[str, Any], drift_configuration: dict[str, Any]
) -> None:
    """
    Raise DataDriftPolicyError if the report violates fail_on / fail_on_incomplete.

    fail_on values (after normalization): warn — block on warn or critical;
    critical — block only on critical. Incomplete runs (no reference/current) are
    controlled separately by fail_on_incomplete.
    """
    status = str(report.get("status") or "ok")
    if status in ("no_reference", "no_current"):
        if drift_configuration.get("fail_on_incomplete"):
            raise DataDriftPolicyError(
                f"Drift check incomplete ({status}): {report.get('message', '')}"
            )
        return
    fail_on = _normalize_fail_on(drift_configuration.get("fail_on"))
    if fail_on is None:
        return
    threshold_rank = _DRIFT_STATUS_RANK[fail_on]
    current_rank = _DRIFT_STATUS_RANK.get(status, 0)
    if current_rank >= threshold_rank:
        raise DataDriftPolicyError(
            f"Data drift status={status!r} violates quality.drift.fail_on={fail_on!r}."
        )


def derive_drift_actions(report: dict[str, Any]) -> list[str]:
    overall_status = report.get("status")
    if overall_status == "no_reference":
        return [
            "freeze_reference: run `python run.py --drift-ref` after a trusted snapshot exists."
        ]
    if overall_status == "no_current":
        return ["reload_data: current statistics file missing."]
    recommended_actions: list[str] = []
    if overall_status == "critical":
        recommended_actions.append(
            "retrain_or_recalibrate: strong distribution shift vs reference."
        )
    elif overall_status == "warn":
        recommended_actions.append(
            "review_features: moderate drift; check EDA and segment splits."
        )

    per_feature_breakdown = report.get("per_feature") or {}
    numeric_severity_by_column = per_feature_breakdown.get("numeric") or {}
    categorical_severity_by_column = per_feature_breakdown.get("categorical") or {}
    if any(
        column_metrics.get("severity") in ("warn", "critical")
        for column_metrics in numeric_severity_by_column.values()
    ):
        recommended_actions.append(
            "refresh_quality_artifacts: re-run stats/association if feature space shifted."
        )
    if any(
        column_metrics.get("severity") in ("warn", "critical")
        for column_metrics in categorical_severity_by_column.values()
    ):
        recommended_actions.append(
            "review_categorical_encoding: cardinality / rare levels may need policy updates."
        )
    if not recommended_actions and overall_status == "ok":
        recommended_actions.append("none: within reference tolerances.")
    return recommended_actions


def _path_relative_to_project_or_absolute(
    filesystem_path: Path, project_root: Path
) -> str:
    try:
        return str(filesystem_path.relative_to(project_root))
    except ValueError:
        return str(filesystem_path)


def run_drift_monitor(config_path: str = "config.yaml") -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    config = load_config(resolved_config_path)
    drift_decimals = quality_round_precision(config)
    drift_thresholds = drift_settings(config)
    raw_fail_on = drift_thresholds.get("fail_on")
    if raw_fail_on not in (None, "", False) and _normalize_fail_on(raw_fail_on) is None:
        print(
            f"Warning: quality.drift.fail_on={raw_fail_on!r} is ignored "
            "(use null, 'warn', or 'critical')."
        )
    _, reference_statistics_path, drift_report_output_path = _resolve_drift_file_paths(
        resolved_config_path, config
    )
    current_statistics_path = resolved_config_path.parent / config["data_storage"][
        "statistics_path"
    ]

    target_column_name = (config.get("columns") or {}).get("target")
    excluded_column_names = {str(target_column_name)} if target_column_name else set()
    quality_section = config.get("quality") or {}
    drift_section = quality_section.get("drift") or {}
    raw_exclude_list = drift_section.get("exclude_columns") or []
    excluded_column_names |= {str(name) for name in raw_exclude_list}

    if not reference_statistics_path.is_file():
        report = {
            "status": "no_reference",
            "message": (
                f"Missing reference {reference_statistics_path}; "
                "run `python run.py --drift-ref` first."
            ),
            "per_feature": {},
            "actions": derive_drift_actions({"status": "no_reference"}),
        }
    elif not current_statistics_path.is_file():
        report = {
            "status": "no_current",
            "message": f"Missing current statistics {current_statistics_path}.",
            "per_feature": {},
            "actions": derive_drift_actions({"status": "no_current"}),
        }
    else:
        reference_stats = load_statistics_yaml(reference_statistics_path)
        current_stats = load_statistics_yaml(current_statistics_path)
        per_feature_breakdown = _compare_reference_and_current_statistics(
            reference_stats,
            current_stats,
            drift_thresholds,
            excluded_column_names=excluded_column_names,
            decimals=drift_decimals,
        )
        severity_rank = {"ok": 0, "warn": 1, "critical": 2}
        all_column_metric_rows = list(
            (per_feature_breakdown["numeric"] or {}).values()
        ) + list((per_feature_breakdown["categorical"] or {}).values())
        worst_severity_rank = max(
            (
                severity_rank.get(column_metrics.get("severity", "ok"), 0)
                for column_metrics in all_column_metric_rows
            ),
            default=0,
        )
        if worst_severity_rank == 2:
            overall_status = "critical"
        elif worst_severity_rank == 1:
            overall_status = "warn"
        else:
            overall_status = "ok"

        report = {
            "status": overall_status,
            "reference_path": _path_relative_to_project_or_absolute(
                reference_statistics_path, resolved_config_path.parent
            ),
            "current_path": _path_relative_to_project_or_absolute(
                current_statistics_path, resolved_config_path.parent
            ),
            "per_feature": per_feature_breakdown,
            "actions": [],
        }
        report["actions"] = derive_drift_actions(report)

    policy_exc: DataDriftPolicyError | None = None
    try:
        enforce_drift_policy(report, drift_thresholds)
    except DataDriftPolicyError as exc:
        policy_exc = exc
    fail_on_raw = drift_thresholds.get("fail_on")
    report["handling"] = {
        "fail_on": _normalize_fail_on(fail_on_raw),
        "fail_on_raw": fail_on_raw,
        "fail_on_incomplete": bool(drift_thresholds.get("fail_on_incomplete")),
        "outcome": "blocked" if policy_exc else "passed",
    }
    if policy_exc is not None:
        report["handling"]["detail"] = str(policy_exc)

    drift_report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with drift_report_output_path.open("w", encoding="utf-8") as yaml_output_file:
        yaml.dump(report, yaml_output_file, allow_unicode=True, sort_keys=False)
    print(
        f"Drift report written: {drift_report_output_path} (status={report['status']})"
    )
    if policy_exc is not None:
        raise policy_exc
    return report
