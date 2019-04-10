# -*- coding: utf-8 -*-

"""Classifier models."""

import numpy as np
from sklearn import neighbors

__all__ = [
    'nearest_neighbor_classifier',
]


class nearest_neighbor_classifier:
    """An efficient classifier for K-nearest neighbors when K is 1."""

    def __init__(self):
        self.tree = None
        self.y_rep = None

    def fit(self, x, y):
        """Fit."""
        self.tree = neighbors.KDTree(x)
        self.y_rep = np.array(y)
        return self

    def score(self, x, y):
        """Score."""
        _, idx = self.tree.query(x, k=1)
        y_pred = self.y_rep[idx]
        return np.sum([y_pred[i] == y[i] for i in np.arange(y_pred.shape[0])]) / len(y)

    def predict(self, x):
        """Predict."""
        _, idx = self.tree.query(x, k=1)
        return np.hstack(self.y_rep[idx])
