import pickle
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_folder', required=True)
parser.add_argument('-m', '--model_name', required=True)
parser.add_argument('-i', '--input_data_filename', required=True)
parser.add_argument('-a', '--action_type', required=True)
args = parser.parse_args()

data_path = os.path.join(args.data_folder, args.input_data_filename)
with open(data_path, 'rb') as f:
    data = pickle.load(f)
    X = data['X']
    Y_true = data['Y']

model_path = os.path.join(args.data_folder, args.model_name)
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    if isinstance(model, dict):
        model = model['model']

Y_pred = model.predict(X)

if args.action_type == 'all':
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
else:
    print("Используйте -a all для вычисления всех метрик")

