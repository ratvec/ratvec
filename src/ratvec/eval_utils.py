# -*- coding: utf-8 -*-

"""Utilities for evaluation of machine learning methodology."""

import random

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

__all__ = [
    'plos_cross_val_score',
]


def plos_cross_val_score(n_components: int, n_neighbors: int, x, y):
    idx = np.arange(len(y))
    random.shuffle(idx)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv_scores = cross_val_score(clf, x[idx, :n_components], y[idx], cv=10)
    return cv_scores.mean()
