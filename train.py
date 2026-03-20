from data_collection.utils import load_config, load_training_data_ready
from models import CatBoostRegressionModel, MLPRegressionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

cfg = load_config("config.yaml")
X, y = load_training_data_ready(
    cfg["data_sources"][0]["path"],
    cfg["columns"]["features"],
    cfg["columns"]["target"],
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_log = np.log1p(y_train)

for name, model in [("CatBoost", CatBoostRegressionModel()), ("MLP", MLPRegressionModel())]:
    model.fit(X_train, y_train_log)
    preds_log = model.predict(X_test)
    # Log-normal bias correction: training in log-space causes expm1(pred) to
    # estimate E[log y] back-transformed, not E[y]. Adding half the residual
    # variance from train predictions corrects for this systematic downward bias.
    resid_var = float(np.var(y_train_log - model.predict(X_train)))
    preds = np.expm1(preds_log + resid_var / 2)
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    rmsle = np.sqrt(np.mean((np.log1p(preds.clip(0)) - np.log1p(y_test)) ** 2))
    r2 = r2_score(y_test, preds)
    print(f"{name}: RMSE = {rmse:.2f}, RMSLE = {rmsle:.4f}, R2 = {r2:.4f}")
