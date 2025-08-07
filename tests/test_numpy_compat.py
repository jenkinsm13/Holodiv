"""Tests for the numpy compatibility layer."""

import numpy as np
from dividebyzero import array, DimensionalArray
import dividebyzero.numpy_compat as numpy_compat
from dividebyzero.numpy_compat import sin, pi


def test_numpy_compat_exports_functions_and_constants():
    """sin and constants like pi should be available from numpy_compat."""
    arr = array([0, pi / 2])
    result = sin(arr)

    assert isinstance(result, DimensionalArray)
    assert np.allclose(result.array, np.sin(arr.array))
    assert np.isclose(pi, np.pi)


def test_numpy_compat_does_not_expose_internal_helpers():
    """Internal implementation details should not be publicly exported."""

    # None of the helper names should be present in the public API
    assert "np" not in numpy_compat.__all__
    assert "inspect" not in numpy_compat.__all__
    assert "wrap_and_register_numpy_function" not in numpy_compat.__all__

    # These names are also removed from the module namespace
    assert not hasattr(numpy_compat, "np")
    assert not hasattr(numpy_compat, "inspect")
    assert not hasattr(numpy_compat, "wrap_and_register_numpy_function")

