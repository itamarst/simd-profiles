from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name="simd_example",
    package_dir={"": "src"},
    packages=["simd_example"],
    ext_modules=cythonize(
        Extension(
            "simd_example._extension",
            ["src/simd_example/_extension.pyx"],
            include_dirs=[np.get_include()],
        )
    ),
)
