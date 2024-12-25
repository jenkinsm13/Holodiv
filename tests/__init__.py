"""Test suite initialization for dividebyzero package."""

import numpy as np
import pytest

# Common test utilities
def generate_test_array(shape=(3, 3), random=False):
    """Generate test arrays for consistent testing."""
    if random:
        return np.random.rand(*shape)
    return np.arange(np.prod(shape)).reshape(shape) + 1

def assert_array_equal_with_tolerance(arr1, arr2, tolerance=1e-10):
    """Compare arrays with numerical tolerance."""
    assert np.all(np.abs(arr1 - arr2) < tolerance)