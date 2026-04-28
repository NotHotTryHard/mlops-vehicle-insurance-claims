import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from src.data.database import db_add_tables, db_clear, ensure_db
from src.data.utils import load_config
from src.data.quality.eda import load_eda_rows_from_db, run_automatic_eda
from src.models import CatBoostRegressionModel, MLPRegressionModel
from src.preprocessing.stream_train_data import (
    build_train_dataset,
    build_val_dataset,
    cat_features_from_frame,
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
) -> None:
    LOGGER.info(
        "train_call started path_csv=%s date_until=%s new_model=%s",
        path_csv,
        date_until,
        new_model,
    )
    family = model_family(new_model)
    preprocessor, X, y = build_train_dataset(
        cfg,
        family,
        config_path=CONFIG_PATH,
        path_csv=path_csv,
        date_until=date_until,
    )
    model = build_model(new_model)
    if family == "catboost":
        metrics = model.train(X, y, cat_features=cat_features_from_frame(X))
    else:
        metrics = model.train(X, y)

    models_path.mkdir(parents=True, exist_ok=True)
    vname = output_variant_name(cfg, family)
    name = f"{new_model}_{vname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with open(models_path / f"{name}.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "variant": vname,
                "model_name": new_model,
                "metrics": metrics,
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
) -> None:
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
        config_path=CONFIG_PATH,
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
        config_path=CONFIG_PATH,
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
    help="Existing model name (val or train update)",
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
def cli(mode, path_csv, date_until, old_model, new_model, clear):
    setup_logging()
    LOGGER.info(
        "cli started mode=%s path_csv=%s date_until=%s old=%s new=%s clear=%s",
        mode,
        path_csv,
        date_until,
        old_model,
        new_model,
        clear,
    )
    try:
        if clear:
            db_clear()
            LOGGER.info("cli completed clear=true")
            return

        if mode is None:
            raise click.UsageError("Missing option '--mode'.")

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

        if mode == "train":
            if (old_model is None) == (new_model is None):
                raise click.UsageError("train requires exactly one of --new or --old.")
        elif mode == "val":
            if old_model is None or new_model is not None:
                raise click.UsageError("val requires --old and no --new.")

        cfg = load_config(CONFIG_PATH)
        models_path = Path(cfg["model_storage"]["models_path"])

        if mode == "train":
            if new_model is not None:
                train_call(path_csv, date_until, new_model, models_path, cfg)
            else:
                update_call(path_csv, date_until, old_model, models_path, cfg)
        elif mode == "val":
            val_call(path_csv, date_until, old_model, models_path, cfg)

        LOGGER.info("cli completed mode=%s", mode)
    except Exception:
        LOGGER.exception("cli failed mode=%s", mode)
        raise


if __name__ == "__main__":
    cli()
