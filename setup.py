"""
Setup script pour le projet de détection de façades
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lire le README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="detection-facades",
    version="1.1.0",
    author="Nouhaila Berghane",
    description="Système de détection 3D des façades de bâtiments fusionnant LiDAR et images 2D avec YOLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "open3d>=0.17.0",
        "laspy>=2.5.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)


