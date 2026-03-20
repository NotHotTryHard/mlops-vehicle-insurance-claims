from data_collection.utils import load_config, load_training_data_ready
from models import CatBoostRegressionModel, MLPRegressionModel
from sklearn.model_selection import train_test_split
import numpy as np

cfg = load_config("config.yaml")
X, y = load_training_data_ready(
    cfg["data_sources"][0]["path"],
    cfg["columns"]["features"],
    cfg["columns"]["target"],
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in [("CatBoost", CatBoostRegressionModel()), ("MLP", MLPRegressionModel())]:
    model.fit(X_train, y_train)
    rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))
    print(f"{name}: RMSE = {rmse:.2f}")
