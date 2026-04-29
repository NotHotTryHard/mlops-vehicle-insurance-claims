import pickle
from argparse import ArgumentParser
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args():
    parser = ArgumentParser(description="Lab 2: evaluate pickled sklearn model on pickled data.")
    parser.add_argument("-d", "--data_folder", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-i", "--input_data_filename", required=True)
    parser.add_argument("-a", "--action_type", required=True)
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def main():
    args = parse_args()
    data_folder = Path(args.data_folder)
    data = load_pickle(data_folder / args.input_data_filename)
    model = load_pickle(data_folder / args.model_name)

    X = data["X"]
    y_true = data["y"]
    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted"),
    }

    if args.action_type == "all":
        selected = metrics
    else:
        if args.action_type not in metrics:
            raise SystemExit(
                f"Unknown metric: {args.action_type}. Use one of: all, "
                + ", ".join(sorted(metrics))
            )
        selected = {args.action_type: metrics[args.action_type]}

    for name, value in selected.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
