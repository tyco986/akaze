#!/usr/bin/env python3
"""Setup script for akazecu - AKAZE CUDA Python bindings + C++ tools"""
from skbuild import setup

# Use installed pybind11 to avoid FetchContent (which can hang on network/git)
cmake_args = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_VERBOSE_MAKEFILE=ON",
]
try:
    import pybind11
    cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
except ImportError:
    pass  # Fall back to FetchContent if pybind11 not installed

setup(
    name="akazecu",
    version="0.1.0",
    description="AKAZE CUDA: GPU-accelerated A-KAZE features with Python bindings",
    author="AKAZE Contributors",
    license="BSD",
    packages=["akaze"],
    package_dir={"akaze": "akaze"},
    cmake_install_dir="akaze",
    cmake_args=cmake_args,
    python_requires=">=3.7",
    install_requires=["numpy", "pybind11"],
    extras_require={
        "dev": ["pytest", "scikit-build"],
    },
)
