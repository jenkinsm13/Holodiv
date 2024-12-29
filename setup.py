"""Setup configuration for dividebyzero package."""

from setuptools import setup, find_packages

setup(
    name="dividebyzero",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
        ],
    },
    author="Michael C. Jenkins",
    author_email="jenkinsm@gmail.com",
    description="A quantum tensor network library for dimensional reduction and elevation",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    keywords="quantum, tensor networks, dimensional reduction",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)