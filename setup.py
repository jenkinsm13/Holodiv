"""Setup configuration for dividebyzero package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dividebyzero",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A NumPy extension implementing division by zero as dimensional reduction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dividebyzero",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/dividebyzero/issues",
        "Documentation": "https://dividebyzero.readthedocs.io/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "quantum": ["networkx>=2.6.0"],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-asyncio>=0.18.0",
        ],
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pre-commit",
        ],
    },
    entry_points={
        "console_scripts": [
            "dbz-demo=dividebyzero.cli:main",
        ],
    },
)