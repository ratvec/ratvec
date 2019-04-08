# -*- coding: utf-8 -*-

"""Tests."""

import unittest

import numpy as np

from ratvec.utils import normalize_kernel_matrix


class TestUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_normalize_kernel_matrix(self):
        """"""
        m = [
            [1, 2],
            [2, 3],
            [1, 3],
        ]
        m = np.array(m)
        normalized_matrix = normalize_kernel_matrix(m)
