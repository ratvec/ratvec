# -*- coding: utf-8 -*-

"""Pure python implementations of similarity functions."""

from functools import partial

import Bio.SubsMat.MatrixInfo
import numpy as np
from Bio import pairwise2

from ratvec.utils import ngrams

__all__ = [
    'n_gram_sim_list',
    'ngram_sim',
    'global_alignment_similarity',
]


def n_gram_sim_list(tuples, n_ngram: int = 2):
    """Computes the binary version of n-gram similarity of a list of tuples	"""
    f = partial(ngram_sim, n_ngram)
    return [
        f(x, y)
        for x, y in tuples
    ]


def ngram_sim(x, y, n: int = 2) -> float:
    """Binary version of n-gram similarity."""
    ng_a = ngrams(x, n)
    ng_b = ngrams(y, n)
    x_len = len(ng_a)
    y_len = len(ng_b)

    np_mem = np.zeros([x_len + 1, y_len + 1], dtype=np.intc)
    mem_table = np_mem

    for i in range(1, x_len + 1):
        for j in range(1, y_len + 1):
            mem_table[i][j] = max(
                mem_table[i][j - 1],
                mem_table[i - 1][j],
                mem_table[i - 1][j - 1] + (ng_a[i - 1] == ng_b[j - 1])
            )

    return float(mem_table[x_len][y_len]) / float(max(x_len, y_len))


def global_alignment_similarity(x: str, y: str, matrix: str = 'blosum62') -> float:
    """Give a score based on global pairwise alignment."""
    m = getattr(Bio.SubsMat.MatrixInfo, matrix)
    return 1 / (1 + pairwise2.align.globaldx(x, y, m, score_only=True))
