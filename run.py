from pathlib import Path

import click

from data_collection import db_clear


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--mode",
    type=click.Choice(["train", "val", "add_data"]),
    required=True,
    help="train | val | add_data",
)
@click.option(
    "--path-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="CSV file",
)
@click.option(
    "--date-range",
    nargs=2,
    default=None,
    metavar="START END",
    help="Date range",
)
@click.option("--old",
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
    type=bool,
    default=False,
    help="Clear database and model files",
)
def cli(mode, path_csv, date_range, old_model, new_model, clear):
    if clear:
        db_clear()
        # model_stash_clear()
        return

    if mode == "add_data":
        if path_csv is None:
            raise click.UsageError("add_data requires --path-csv.")
        if date_range is not None:
            raise click.UsageError("add_data does not accept --date-range.")
        if old_model is not None or new_model is not None:
            raise click.UsageError("add_data does not accept --old or --new.")
        return

    has_csv = path_csv is not None
    has_dates = date_range is not None
    if has_csv == has_dates:
        raise click.UsageError(
            "Choose either --path-csv or --date-range."
        )
    if (old_model is None) == (new_model is None):
        raise click.UsageError(
            "train and val require exactly one of --old or --new."
        )



if __name__ == "__main__":
    cli()
