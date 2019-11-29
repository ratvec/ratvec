# -*- coding: utf-8 -*-

"""Setup module."""

import setuptools
#TODO Install numpy automatically before setup is run below
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("*", ["./src/ratvec/*.pyx"],
        include_dirs=[numpy.get_include()]
        )
]

if __name__ == '__main__':
    setuptools.setup(ext_modules=cythonize(extensions))
