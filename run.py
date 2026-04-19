import pickle
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

_MODEL_TO_CLASS = {
    "catboost": CatBoostRegressionModel,
    "mlp": MLPRegressionModel,
}

CONFIG_PATH = "config.yaml"


def train_call(
    path_csv: Optional[Path],
    date_until: Optional[str],
    new_model: str,
    models_path: Path,
    cfg: dict,
) -> None:
    preprocessor, X, y = build_train_dataset(
        cfg,
        new_model,
        config_path=CONFIG_PATH,
        path_csv=path_csv,
        date_until=date_until,
    )
    model = _MODEL_TO_CLASS[new_model]()
    if new_model == "catboost":
        metrics = model.train(X, y, cat_features=cat_features_from_frame(X))
    else:
        metrics = model.train(X, y)

    models_path.mkdir(parents=True, exist_ok=True)
    vname = (cfg.get("preprocessing") or {}).get("default_variant", "default")
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

    print(f"Saved: {name}")
    print(metrics)


def val_call(
    path_csv: Optional[Path],
    date_until: Optional[str],
    old_model: str,
    models_path: Path,
    cfg: dict,
) -> None:
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
    help="Existing model name",
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
    if not clear:
        ensure_db()

    if clear:
        db_clear()
        return

    if mode is None:
        raise click.UsageError("Missing option '--mode'.")

    if mode == "add_data":
        if path_csv is None:
            raise click.UsageError("add_data requires --path-csv.")
        if date_until is not None:
            raise click.UsageError("add_data does not accept --date-until.")
        if old_model is not None or new_model is not None:
            raise click.UsageError("add_data does not accept --old or --new.")
        db_add_tables(config_path=CONFIG_PATH, paths=[path_csv.resolve()])
        return

    if mode == "analyse":
        if path_csv is not None or date_until is not None:
            raise click.UsageError("analyse mode does not use --path-csv or --date-until.")
        if old_model is not None or new_model is not None:
            raise click.UsageError("analyse mode does not use --old or --new.")
        run_automatic_eda(CONFIG_PATH, load_eda_rows_from_db(CONFIG_PATH))
        return

    has_csv = path_csv is not None
    has_dates = date_until is not None
    if has_csv == has_dates:
        raise click.UsageError("Choose either --path-csv or --date-until.")

    if mode == "train":
        if old_model is not None:
            raise click.UsageError("train does not use --old.")
        if new_model is None:
            raise click.UsageError("train requires --new.")
    elif mode == "val":
        if old_model is None or new_model is not None:
            raise click.UsageError("val requires --old and no --new.")

    cfg = load_config(CONFIG_PATH)
    models_path = Path(cfg["model_storage"]["models_path"])

    if mode == "train":
        train_call(path_csv, date_until, new_model, models_path, cfg)
    elif mode == "val":
        val_call(path_csv, date_until, old_model, models_path, cfg)


if __name__ == "__main__":
    cli()
