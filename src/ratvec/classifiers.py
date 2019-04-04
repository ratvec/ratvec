# -*- coding: utf-8 -*-

import numpy as np
from sklearn import neighbors

__all__ = [
    'nearest_neighbor_classifier',
]


class nearest_neighbor_classifier:
    """An efficient classifier for K-nearest neighbors when K is 1."""

    def fit(self, X, Y):
        """Fit."""
        self.tree = neighbors.KDTree(X)
        self.Y_rep = np.array(Y)
        return self

    def score(self, X, Y):
        """Score."""
        _, idx = self.tree.query(X, k=1)
        Y_pred = self.Y_rep[idx]
        return np.sum([Y_pred[i] == Y[i] for i in np.arange(Y_pred.shape[0])]) / len(Y)

    def predict(self, X):
        """Predict."""
        _, idx = self.tree.query(X, k=1)
        return np.hstack(self.Y_rep[idx])
