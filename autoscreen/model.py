"""Per-objective random-forest surrogates with tree-variance uncertainty.

One independent RandomForestRegressor per objective. Predictions return both the
mean and a std estimate (across trees), which the Pareto/UCB acquisition uses.
CPU-only, no TensorFlow/PyTorch dependency.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class MultiOutputRFSurrogate:
    def __init__(self, n_objectives: int, n_estimators: int = 100, n_jobs: int = -1, seed: int = 0):
        self.n_objectives = n_objectives
        self.models = [
            RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=seed + k,
                max_features="sqrt",
            )
            for k in range(n_objectives)
        ]
        self._fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiOutputRFSurrogate":
        for k, model in enumerate(self.models):
            model.fit(X, Y[:, k])
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.zeros((X.shape[0], self.n_objectives))
        stds = np.zeros((X.shape[0], self.n_objectives))
        for k, model in enumerate(self.models):
            per_tree = np.stack([est.predict(X) for est in model.estimators_], axis=1)
            means[:, k] = per_tree.mean(axis=1)
            stds[:, k] = per_tree.std(axis=1)
        return means, stds
