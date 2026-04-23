import pickle
from argparse import ArgumentParser
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-d', '--data_folder', required=True)
    parser.add_argument('-m', '--model_name', required=True)
    parser.add_argument('-i', '--input_data_filename', required=True)
    parser.add_argument('-a', '--action_type', required=True)
    return parser.parse_args()


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def main():
    args = parse_args()

    data_folder = Path(args.data_folder)
    data = load_pickle(data_folder / args.input_data_filename)
    model = load_pickle(data_folder / args.model_name)

    if isinstance(model, dict):
        model = model['model']

    X = data['X']
    y_true = data['y']
    y_pred = model.predict(X)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted'),
    }

    if args.action_type == 'all':
        selected_metrics = metrics
    else:
        selected_metrics = {args.action_type: metrics[args.action_type]}

    for name, value in selected_metrics.items():
        print(f'{name}: {value:.4f}')


if __name__ == '__main__':
    main()


'''
docker build -t predict-model .

docker run --rm predict-model \
  -d /app \
  -m model_best.pickle \
  -i test_data.pickle \
  -a all
'''