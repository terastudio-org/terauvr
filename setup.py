#!/usr/bin/env python
"""
Setup script for terauvr package.
"""
import os
import sys
from setuptools import setup, find_packages

# Ensure minimum Python version
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required.")

# Read long description from README
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="terauvr",
    version="1.0.0",
    author="terastudio-org/terastudio",
    description="Advanced AI-powered VR/AR audio separation and processing toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terastudio-org/terauvr",
    project_urls={
        "Bug Tracker": "https://github.com/terastudio-org/terauvr/issues",
        "Documentation": "https://terastudio-docs.readthedocs.io/",
        "Source Code": "https://github.com/terastudio-org/terauvr",
    },
    package_dir={"": "terauvr"},
    packages=find_packages(where="terauvr"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Audio",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "numpy>=1.21.0",
        "gradio>=3.30.0",
        "librosa>=0.9.0",
        "scipy>=1.7.0",
        "onnxruntime>=1.12.0",
        
        # Audio processing
        "soundfile>=0.12.0",
        "pydub>=0.25.0",
        "resampy>=0.4.0",
        
        # Utilities
        "Pillow>=8.0.0",
        "click>=8.0.0",
        "psutil>=5.8.0",
        "tqdm>=4.60.0",
        
        # Optional dependencies (will be handled gracefully)
        "torch-directml; platform_system=='Windows'",
        "pytorch-ocl; platform_system=='Linux'",
        "onnxruntime-gpu; platform_system=='Windows'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
        "all": [
            "torch-directml>=1.0.0",
            "pytorch-ocl>=1.0.0",
            "onnxruntime-gpu>=1.12.0",
            "pywin32>=227; platform_system=='Windows'",
        ],
    },
    entry_points={
        "console_scripts": [
            "terauvr=terauvr.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai audio separation vr ar music machine-learning deep-learning",
)