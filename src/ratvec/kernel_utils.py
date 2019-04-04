# -*- coding: utf-8 -*-

"""Distance functions for each kernel."""

import numpy as np


def project_with_rbf_kernel(s, h):
    d = np.ones(len(s)) - s
    return np.exp(-h * (d ** 2))


def project_with_polynomial_kernel(s, h):
    return s ** h


def project_with_linear_kernel(s, h=0):
    return s + h


KERNEL_TO_PROJECTION = {
    'poly': project_with_polynomial_kernel,
    'rbf': project_with_rbf_kernel,
    'linear': project_with_linear_kernel,
}
