import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from src.data.database import build_drift_reference, db_add_tables, db_clear, ensure_db
from src.data.quality.drift import DataDriftPolicyError
from src.data.utils import load_config
from src.data.quality.eda import load_eda_rows_from_db, run_automatic_eda
from src.models import CatBoostRegressionModel, MLPRegressionModel
from src.preprocessing import (
    build_train_dataset,
    build_val_dataset,
    cat_features_from_frame,
    load_train_rows_y,
    preprocess_tune_variant_keys,
)

_MODEL_SPECS = {
    "catboost": {
        "family": "catboost",
        "cls": CatBoostRegressionModel,
        "kwargs": {},
    },
    "mlp": {
        "family": "mlp",
        "cls": MLPRegressionModel,
        "kwargs": {"loss": "huber", "huber_delta": 1.0, "lr": 5e-4, "max_epochs": 400},
    },
}

CONFIG_PATH = "config.yaml"
LOG_PATH = Path("session/logs/run.log")
LOGGER = logging.getLogger("mlops_run")


def model_family(model_name: str) -> str:
    return _MODEL_SPECS[model_name]["family"]


def build_model(model_name: str):
    spec = _MODEL_SPECS[model_name]
    return spec["cls"](**spec["kwargs"])


def resolve_incremental_parent_model(cli_old: Optional[str], cfg: dict) -> Optional[str]:
    if cli_old is not None:
        return cli_old.strip()
    block = cfg.get("incremental_training") or {}
    if block.get("enabled") and block.get("parent_model"):
        return str(block["parent_model"]).strip()
    return None


def output_variant_name(cfg: dict, family: str) -> str:
    vname = (cfg.get("preprocessing") or {}).get("default_variant", "default")
    if family == "mlp" and str(vname).startswith("catboost"):
        return "mlp_auto"
    if family == "catboost" and str(vname).startswith("mlp"):
        return "catboost_auto"
    return str(vname)


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


def train_call(
    path_csv: Optional[Path],
    date_until: Optional[str],
    new_model: str,
    models_path: Path,
    cfg: dict,
    *,
    config_path: str,
) -> None:
    LOGGER.info(
        "train_call started path_csv=%s date_until=%s new_model=%s",
        path_csv,
        date_until,
        new_model,
    )
    family = model_family(new_model)
    tune_preprocess = bool(
        (cfg.get("preprocessing") or {}).get("tune_preprocess_variants")
    )

    if tune_preprocess:
        rows, y_raw = load_train_rows_y(
            cfg,
            config_path=config_path,
            path_csv=path_csv,
            date_until=date_until,
        )
        variant_keys = preprocess_tune_variant_keys(cfg, family)
        msg = (
            f"[train] preprocessing variant sweep: {len(variant_keys)} candidate(s) "
            f"for model={family!r}: {variant_keys}"
        )
        print(msg, flush=True)
        LOGGER.info("%s", msg)
        best_rmse = float("inf")
        best_key: str | None = None
        for vk in tqdm(variant_keys, desc="preprocess variants", unit="variant"):
            _, X_try, y_try = build_train_dataset(
                cfg,
                family,
                config_path=config_path,
                rows=rows,
                y=y_raw,
                variant_name=vk,
            )
            trial = build_model(new_model)
            if family == "catboost":
                trial_metrics = trial.train(
                    X_try, y_try, cat_features=cat_features_from_frame(X_try)
                )
            else:
                trial_metrics = trial.train(X_try, y_try)
            rmse = float(trial_metrics["RMSE"])
            line = f"[train]   variant={vk!r} holdout RMSE={rmse:.6f}"
            print(line, flush=True)
            LOGGER.info("preprocess sweep variant=%s holdout_RMSE=%s", vk, rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_key = vk
        assert best_key is not None
        preprocessor, X, y = build_train_dataset(
            cfg,
            family,
            config_path=config_path,
            rows=rows,
            y=y_raw,
            variant_name=best_key,
        )
        LOGGER.info(
            "selected preprocess variant=%s (best holdout RMSE=%s)",
            best_key,
            best_rmse,
        )
        vname = best_key
    else:
        preprocessor, X, y = build_train_dataset(
            cfg,
            family,
            config_path=config_path,
            path_csv=path_csv,
            date_until=date_until,
        )
        vname = preprocessor.variant_key

    model = build_model(new_model)
    if family == "catboost":
        metrics = model.train(X, y, cat_features=cat_features_from_frame(X))
    else:
        metrics = model.train(X, y)

    models_path.mkdir(parents=True, exist_ok=True)
    name = f"{new_model}_{vname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with open(models_path / f"{name}.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "variant": vname,
                "model_name": new_model,
                "metrics": metrics,
                "tune_preprocess_variants": tune_preprocess,
            },
            f,
        )

    LOGGER.info(
        "train_call saved model=%s metrics=%s path=%s",
        name,
        metrics,
        models_path / f"{name}.pkl",
    )
    print(f"Saved: {name}")
    print(metrics)


def update_call(
    path_csv: Optional[Path],
    date_until: Optional[str],
    old_model: str,
    models_path: Path,
    cfg: dict,
    *,
    config_path: str,
    parent_source: str = "unknown",
) -> None:
    data_ref = f"path_csv={path_csv}" if path_csv is not None else f"date_until={date_until}"
    inc_msg = (
        f"Incremental training: parent bundle={old_model!r} "
        f"Applying model.update() on loaded bundle."
    )
    print(inc_msg, flush=True)
    LOGGER.info(inc_msg)
    LOGGER.info(
        "update_call started path_csv=%s date_until=%s old_model=%s",
        path_csv,
        date_until,
        old_model,
    )
    with open(models_path / f"{old_model}.pkl", "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    X, y = build_val_dataset(
        cfg,
        preprocessor=preprocessor,
        config_path=config_path,
        path_csv=path_csv,
        date_until=date_until,
    )
    family = getattr(preprocessor, "model_kind", "catboost")
    if family == "catboost":
        metrics = model.update(X, y, cat_features=cat_features_from_frame(X))
    else:
        metrics = model.update(X, y)

    models_path.mkdir(parents=True, exist_ok=True)
    vname = bundle.get("variant") or output_variant_name(cfg, family)
    mname = bundle.get("model_name") or old_model.split("_")[0]
    name = f"{mname}_{vname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with open(models_path / f"{name}.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "variant": vname,
                "model_name": mname,
                "metrics": metrics,
                "parent_model": old_model,
            },
            f,
        )

    LOGGER.info(
        "update_call saved model=%s parent_model=%s metrics=%s path=%s",
        name,
        old_model,
        metrics,
        models_path / f"{name}.pkl",
    )
    print(f"Updated and saved: {name} (from {old_model})")
    print(metrics)


def val_call(
    path_csv: Optional[Path],
    date_until: Optional[str],
    old_model: str,
    models_path: Path,
    cfg: dict,
    *,
    config_path: str,
) -> None:
    LOGGER.info(
        "val_call started path_csv=%s date_until=%s old_model=%s",
        path_csv,
        date_until,
        old_model,
    )
    with open(models_path / f"{old_model}.pkl", "rb") as f:
        bundle = pickle.load(f)

    X, y = build_val_dataset(
        cfg,
        preprocessor=bundle["preprocessor"],
        config_path=config_path,
        path_csv=path_csv,
        date_until=date_until,
    )
    metrics = bundle["model"].evaluate(X, y)
    LOGGER.info("val_call metrics=%s old_model=%s", metrics, old_model)
    print(metrics)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--mode",
    type=click.Choice(["train", "val", "add_data", "analyse"]),
    default=None,
    help="train | val | add_data | analyse",
)
@click.option(
    "--drift-ref",
    "drift_ref",
    is_flag=True,
    default=False,
    help="Reload data_sources into DB, write meta/statistics, freeze drift reference (no --mode).",
)
@click.option(
    "--path-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="CSV file",
)
@click.option(
    "--date-until",
    "date_until",
    default=None,
    metavar="DATE",
    help="SQLite only: include rows with event_date <= DATE (ISO). Requires --mode train or val without --path-csv.",
)
@click.option(
    "--old",
    "old_model",
    type=str,
    default=None,
    help="Model bundle basename without .pkl, or use incremental_training in config.yaml.",
)
@click.option(
    "--new",
    "new_model",
    type=click.Choice(["catboost", "mlp"], case_sensitive=False),
    default=None,
    help="Model type (catboost | mlp)",
)
@click.option(
    '--clear',
    is_flag=True,
    default=False,
    help="Clear database and model files",
)
def cli(mode, path_csv, date_until, old_model, new_model, clear, drift_ref):
    setup_logging()
    LOGGER.info(
        "cli started mode=%s path_csv=%s date_until=%s old=%s new=%s clear=%s drift_ref=%s",
        mode,
        path_csv,
        date_until,
        old_model,
        new_model,
        clear,
        drift_ref,
    )
    try:
        if clear:
            db_clear()
            LOGGER.info("cli completed clear=true")
            return

        if drift_ref:
            if mode is not None:
                raise click.UsageError("Use --drift-ref without --mode.")
            if path_csv is not None or date_until is not None:
                raise click.UsageError("--drift-ref does not accept --path-csv or --date-until.")
            if old_model is not None or new_model is not None:
                raise click.UsageError("--drift-ref does not accept --old or --new.")
            build_drift_reference(CONFIG_PATH)
            LOGGER.info("cli completed --drift-ref")
            db_clear()
            return

        if mode is None:
            raise click.UsageError("Missing option '--mode' (or use --drift-ref alone).")

        # add_data seeds only the given CSV; db_add_tables creates the DB if missing.
        # Skip ensure_db here so Docker (no bundled datasets/) does not fail before --path-csv is used.
        if mode == "add_data":
            if path_csv is None:
                raise click.UsageError("add_data requires --path-csv.")
            if date_until is not None:
                raise click.UsageError("add_data does not accept --date-until.")
            if old_model is not None or new_model is not None:
                raise click.UsageError("add_data does not accept --old or --new.")
            db_add_tables(config_path=CONFIG_PATH, paths=[path_csv.resolve()])
            LOGGER.info("cli completed mode=add_data path_csv=%s", path_csv.resolve())
            return

        ensure_db()

        if mode == "analyse":
            if path_csv is not None or date_until is not None:
                raise click.UsageError("analyse mode does not use --path-csv or --date-until.")
            if old_model is not None or new_model is not None:
                raise click.UsageError("analyse mode does not use --old or --new.")
            run_automatic_eda(CONFIG_PATH, load_eda_rows_from_db(CONFIG_PATH))
            LOGGER.info("cli completed mode=analyse")
            return

        has_csv = path_csv is not None
        has_dates = date_until is not None
        if has_csv == has_dates:
            raise click.UsageError("Choose either --path-csv or --date-until.")

        config_resolved = str(Path(CONFIG_PATH).resolve())
        cfg = load_config(config_resolved)
        models_path = Path(cfg["model_storage"]["models_path"])

        train_parent: Optional[str] = None
        if mode == "train":
            train_parent = resolve_incremental_parent_model(old_model, cfg)
            if new_model is not None and train_parent is not None:
                raise click.UsageError(
                    "train: use either --new (fresh) or incremental (--old / "
                    "incremental_training.parent_model), not both."
                )
            if new_model is None and train_parent is None:
                raise click.UsageError(
                    "train requires --new for fresh training, or --old / "
                    "incremental_training (enabled: true and parent_model set) for update."
                )
        elif mode == "val":
            if old_model is None or new_model is not None:
                raise click.UsageError("val requires --old and no --new.")

        if mode == "train":
            if new_model is not None:
                train_call(
                    path_csv,
                    date_until,
                    new_model,
                    models_path,
                    cfg,
                    config_path=config_resolved,
                )
            else:
                update_call(
                    path_csv,
                    date_until,
                    train_parent,
                    models_path,
                    cfg,
                    config_path=config_resolved,
                    parent_source=(
                        "CLI --old"
                        if old_model is not None
                        else "config.yaml incremental_training"
                    ),
                )
        elif mode == "val":
            val_call(
                path_csv,
                date_until,
                old_model,
                models_path,
                cfg,
                config_path=config_resolved,
            )

        LOGGER.info("cli completed mode=%s", mode)
    except DataDriftPolicyError as exc:
        LOGGER.error("cli blocked by drift policy mode=%s: %s", mode, exc)
        raise
    except Exception:
        LOGGER.exception("cli failed mode=%s drift_ref=%s", mode, drift_ref)
        raise


if __name__ == "__main__":
    cli()
