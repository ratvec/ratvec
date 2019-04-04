# -*- coding: utf-8 -*-

import distance
import numpy as np

from ratvec.utils import ngrams


# TODO replace with numpy implementation of sorensen dice?

def sorensen_plus(a: str, b: str) -> float:
    length = min(len(a), len(b))
    ng = [
        distance.sorensen(ngrams(a, n), ngrams(b, n))
        for n in range(1, length + 1)
    ]
    return 1 - np.sum(ng) / length
