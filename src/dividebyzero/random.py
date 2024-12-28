"""Random number generation with DimensionalArray support."""

import numpy as np
from .array import DimensionalArray
from typing import Optional, Tuple, Union

def rand(*args) -> DimensionalArray:
    """Random values in a given shape.
    
    Create an array of the given shape and populate it with random
    samples from a uniform distribution over [0, 1).
    """
    return DimensionalArray(np.random.rand(*args))

def randn(*args) -> DimensionalArray:
    """Return a sample (or samples) from the "standard normal" distribution."""
    return DimensionalArray(np.random.randn(*args))

def randint(low: int, high: Optional[int] = None, size: Optional[Union[int, Tuple[int, ...]]] = None) -> DimensionalArray:
    """Return random integers from low (inclusive) to high (exclusive)."""
    return DimensionalArray(np.random.randint(low, high, size))

def random(size: Optional[Union[int, Tuple[int, ...]]] = None) -> DimensionalArray:
    """Return random floats in the half-open interval [0.0, 1.0)."""
    return DimensionalArray(np.random.random(size))

def normal(loc: float = 0.0, scale: float = 1.0, size: Optional[Union[int, Tuple[int, ...]]] = None) -> DimensionalArray:
    """Draw random samples from a normal (Gaussian) distribution."""
    return DimensionalArray(np.random.normal(loc, scale, size))

def uniform(low: float = 0.0, high: float = 1.0, size: Optional[Union[int, Tuple[int, ...]]] = None) -> DimensionalArray:
    """Draw samples from a uniform distribution."""
    return DimensionalArray(np.random.uniform(low, high, size))

def multivariate_normal(mean, cov, size: Optional[int] = None) -> DimensionalArray:
    """Draw random samples from a multivariate normal distribution.
    
    Parameters
    ----------
    mean : array_like
        Mean of the distribution (1-D array-like)
    cov : array_like
        Covariance matrix of the distribution (2-D array-like)
    size : int, optional
        Number of samples to draw (default: 1)
        
    Returns
    -------
    DimensionalArray
        Drawn samples from the multivariate normal distribution.
    """
    if isinstance(mean, DimensionalArray):
        mean = mean.array
    if isinstance(cov, DimensionalArray):
        cov = cov.array
    return DimensionalArray(np.random.multivariate_normal(mean, cov, size))

# Add seed function for reproducibility
def seed(seed: Optional[int] = None) -> None:
    """Seed the random number generator."""
    np.random.seed(seed) 