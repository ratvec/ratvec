# -*- coding: utf-8 -*-

"""Utilities for evaluation of machine learning methodology."""

import random

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import  make_scorer

__all__ = [
    'knn_cross_val_score',
]


mcc_scorer = make_scorer(matthews_corrcoef)


def knn_cross_val_score(n_components: int, n_neighbors: int, x, y):
    idx = np.arange(len(y))
    random.shuffle(idx)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv_scores = cross_val_score(clf, x[idx, :n_components], y[idx], cv=10, scoring="f1")
    return cv_scores.mean()
