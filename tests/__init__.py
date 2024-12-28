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
    if arr1.dtype == bool and arr2.dtype == bool:
        assert np.array_equal(arr1, arr2), f"Boolean arrays are not equal: {arr1} != {arr2}"
    else:
        assert np.all(np.abs(arr1 - arr2) < tolerance), f"Arrays differ by more than tolerance: {arr1} != {arr2}"