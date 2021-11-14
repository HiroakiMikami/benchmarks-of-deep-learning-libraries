from setuptools import find_packages, setup
from Cython.Build import cythonize
import numpy

setup(
    name="benchmark",
    packages=find_packages(),
    ext_modules=cythonize("cpu/cython/_longest_common_subsequence.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
