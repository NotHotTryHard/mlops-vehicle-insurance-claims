import pickle
from datetime import datetime
from pathlib import Path

import click

from src.database import db_clear, load_config, load_raw
from src.models import CatBoostRegressionModel, MLPRegressionModel
from src.preprocessing import NumericOnlyPreprocessor

_MODEL_TO_CLASS = {
    "catboost": CatBoostRegressionModel,
    "mlp": MLPRegressionModel,
}


def train_call(path_csv: Path, new_model: str, models_path: Path) -> None:
    X_raw, y = load_raw(path_csv)

    preprocessor = NumericOnlyPreprocessor()
    X = preprocessor.fit_transform(X_raw)
    model = _MODEL_TO_CLASS[new_model]()

    metrics = model.train(X, y)
    models_path.mkdir(parents=True, exist_ok=True)
    name = f"{new_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with open(models_path / f"{name}.pkl", "wb") as f:
        pickle.dump({"model": model, "preprocessor": preprocessor}, f)

    print(f"Saved: {name}")
    print(metrics)


def val_call(path_csv: Path, old_model: str, models_path: Path) -> None:
    X_raw, y = load_raw(path_csv)

    with open(models_path / f"{old_model}.pkl", "rb") as f:
        bundle = pickle.load(f)

    X = bundle["preprocessor"].transform(X_raw)
    metrics = bundle["model"].evaluate(X, y)
    print(metrics)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--mode",
    type=click.Choice(["train", "val", "add_data"]),
    default=None,
    help="train | val | add_data",
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
    help="Use DB data in range [the birth of a fuckin' universe, DATE]",
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
    help="New model type",
)
@click.option(
    '--clear',
    is_flag=True,
    default=False,
    help="Clear database and model files",
)
def cli(mode, path_csv, date_until, old_model, new_model, clear):
    if clear:
        db_clear()
        # model_stash_clear()
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
        return

    has_csv = path_csv is not None
    has_dates = date_until is not None
    if has_csv == has_dates:
        raise click.UsageError("Choose either --path-csv or --date-until.")
    if (old_model is None) == (new_model is None):
        raise click.UsageError("train and val require exactly one of --old or --new.")

    cfg = load_config("config.yaml")
    models_path = Path(cfg["model_storage"]["models_path"])

    if mode == "train":
        train_call(path_csv, new_model, models_path)
    elif mode == "val":
        val_call(path_csv, old_model, models_path)


if __name__ == "__main__":
    cli()
