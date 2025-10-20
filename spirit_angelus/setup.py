#!/usr/bin/env python3
"""
Setup script for Spirit Angelus Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="spirit-angelus-framework",
    version="0.1.0",
    author="Spirit Angelus Framework Team",
    author_email="contact@spiritangelus.dev",
    description="A spiritual-tech hybrid framework for personalized spiritual guidance",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/spirit-angelus/framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Religion",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyter>=1.0.0",
        ],
        "web": [
            "flask>=2.2.0",
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "streamlit>=1.15.0",
            "dash>=2.6.0",
        ],
        "quantum": [
            "qutip>=4.7.0",
            "qiskit>=0.40.0",
            "cirq>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spirit-angelus=spirit_angelus.cli:main",
            "spirit-web=spirit_angelus.web.app:main",
            "spirit-meditation=spirit_angelus.quantum.meditation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md", "*.svg", "*.png"],
    },
    zip_safe=False,
)