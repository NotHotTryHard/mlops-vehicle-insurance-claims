from pathlib import Path
from typing import Optional

import yaml

from src.data.utils import load_config

DEFAULT_PREPROCESSING: dict = {
    "target_missing_fill": 0.0,
    "default_variant": "catboost_ord",
    "tune_preprocess_variants": False,
    "preprocess_variant_candidates": None,
    "variants": {
        "catboost_ord": {
            "numeric": {"impute": "median", "scale": False},
            "categorical": {"impute": "most_frequent", "encode": "ordinal"},
        },
        "mlp_ohe": {
            "numeric": {"impute": "median", "scale": True},
            "categorical": {"impute": "most_frequent", "encode": "onehot"},
        },
        "mlp_ord": {
            "numeric": {"impute": "median", "scale": True},
            "categorical": {"impute": "most_frequent", "encode": "ordinal"},
        },
    },
}


def preprocess_block(cfg: dict) -> dict:
    out = dict(DEFAULT_PREPROCESSING)
    user = cfg.get("preprocessing") or {}
    for key, val in user.items():
        if key == "variants":
            continue
        out[key] = val
    out["variants"] = {**out["variants"], **(user.get("variants") or {})}
    return out


def preprocess_tune_variant_keys(cfg: dict, model_family: str) -> list[str]:
    """
    Кандидаты для перебора: только ключи вида ``{model_family}_*`` (например ``catboost_ord`` для
    CatBoost), чтобы не смешивать пресеты MLP с деревом.
    """
    prep = preprocess_block(cfg)
    variants = prep["variants"]
    explicit = prep.get("preprocess_variant_candidates")
    if explicit is not None:
        if not explicit:
            raise ValueError("preprocess_variant_candidates must be a non-empty list when set.")
        unknown = [k for k in explicit if k not in variants]
        if unknown:
            raise KeyError(
                f"Unknown preprocess_variant_candidates keys {unknown}; "
                f"allowed: {sorted(variants)}"
            )
        keys = list(explicit)
    else:
        keys = sorted(variants.keys())
    prefix = f"{model_family}_"
    matched = [k for k in keys if k.startswith(prefix)]
    if not matched:
        raise ValueError(
            f"No preprocessing variants for model_family={model_family!r} "
            f"(need names starting with {prefix!r}). Considered: {keys}."
        )
    return matched


def resolve_variant_key(
    cfg: dict, model_name: str, *, variant_name: Optional[str] = None
) -> str:
    prep = preprocess_block(cfg)
    variants = prep["variants"]
    if variant_name is not None:
        if variant_name not in variants:
            raise KeyError(
                f"Unknown preprocessing variant {variant_name!r}; "
                f"allowed: {sorted(variants)}"
            )
        return variant_name
    vname = prep["default_variant"]
    if model_name == "mlp" and str(vname).startswith("catboost"):
        if "mlp_ohe" in variants:
            vname = "mlp_ohe"
        elif "mlp_ord" in variants:
            vname = "mlp_ord"
    if model_name == "catboost" and str(vname).startswith("mlp"):
        if "catboost_ord" in variants:
            vname = "catboost_ord"
    return str(vname)


def feature_matrix_column_names(config_path: str) -> tuple[list[str], list[str]]:
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    qp = cfg_path.parent / cfg["data_storage"]["quality_path"]
    if not qp.exists():
        raise ValueError(
            f"Missing quality file {qp}. Run the quality pipeline (or run_cleaning_summary) first."
        )
    with qp.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    block = data.get("feature_matrix_columns")
    if not block:
        raise ValueError(
            f"feature_matrix_columns missing in {qp}. Run run_cleaning_summary / quality pipeline first."
        )
    return list(block["numeric"]), list(block["categorical"])
