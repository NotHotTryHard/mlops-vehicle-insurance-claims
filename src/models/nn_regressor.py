import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .base import BaseRegressor


class _MLP(nn.Module):
    def __init__(self, in_features, hidden_layer_sizes):
        super().__init__()

        layers = []
        prev = in_features
        for h in hidden_layer_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPRegressionModel(BaseRegressor):

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        lr=1e-3,
        max_epochs=300,
        batch_size=256,
        patience=15,
        val_fraction=0.1,
        random_state=42,
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._net = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_scaled = self._scaler.fit_transform(X)

        n = len(X_scaled)
        n_val = max(1, int(n * self.val_fraction))
        perm = np.random.permutation(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_tr = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
        y_tr = torch.tensor(y[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32)

        self._net = _MLP(X_tr.shape[1], self.hidden_layer_sizes)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for _ in range(self.max_epochs):
            self._net.train()
            idx = torch.randperm(len(X_tr))
            for start in range(0, len(X_tr), self.batch_size):
                batch = idx[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self._net(X_tr[batch]), y_tr[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._net(X_val), y_val).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        return self

    def predict(self, X):
        X_scaled = self._scaler.transform(X)
        self._net.eval()
        with torch.no_grad():
            return self._net(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
