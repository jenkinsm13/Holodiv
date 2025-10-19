"""Tests for the numpy compatibility layer."""

import numpy as np
from holodiv import array, DimensionalArray
from holodiv.numpy_compat import sin, pi


def test_numpy_compat_exports_functions_and_constants():
    """sin and constants like pi should be available from numpy_compat."""
    arr = array([0, pi / 2])
    result = sin(arr)

    assert isinstance(result, DimensionalArray)
    assert np.allclose(result.array, np.sin(arr.array))
    assert np.isclose(pi, np.pi)

