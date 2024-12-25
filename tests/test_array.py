"""Test suite for core array functionality."""

import pytest
import numpy as np
from dividebyzero import array
from dividebyzero.exceptions import DimensionalError, ReconstructionError
from . import generate_test_array, assert_array_equal_with_tolerance

class TestDimensionalArray:
    """Test suite for DimensionalArray class."""
    
    def test_array_creation(self):
        """Test array initialization."""
        # Test basic creation
        data = generate_test_array()
        arr = array(data)
        assert_array_equal_with_tolerance(arr.array, data)
        
        # Test creation from list
        arr = array([1, 2, 3])
        assert arr.shape == (3,)
        
        # Test creation from another DimensionalArray
        arr2 = array(arr)
        assert_array_equal_with_tolerance(arr2.array, arr.array)

    def test_basic_operations(self):
        """Test standard arithmetic operations."""
        arr = array([1, 2, 3])
        
        # Test multiplication
        result = arr * 2
        assert_array_equal_with_tolerance(result.array, np.array([2, 4, 6]))
        
        # Test addition
        result = arr + 1
        assert_array_equal_with_tolerance(result.array, np.array([2, 3, 4]))
        
        # Test regular division
        result = arr / 2
        assert_array_equal_with_tolerance(result.array, np.array([0.5, 1, 1.5]))

    def test_divide_by_zero_scalar(self):
        """Test division by zero for scalar values."""
        arr = array([1, 2, 3])
        result = arr / 0
        
        # Check dimension reduction
        assert result.ndim < arr.ndim or result.shape[-1] < arr.shape[-1]
        
        # Test reconstruction
        reconstructed = result.elevate()
        assert reconstructed.shape == arr.shape
        
        # Check information preservation (approximate)
        assert np.abs(reconstructed.array.mean() - arr.array.mean()) < 0.1

    def test_divide_by_zero_matrix(self):
        """Test division by zero for matrices."""
        arr = array(generate_test_array((3, 3)))
        result = arr / 0
        
        # Verify dimension reduction
        assert result.ndim < arr.ndim
        
        # Test reconstruction
        reconstructed = result.elevate()
        assert reconstructed.shape == arr.shape
        
        # Check singular values preservation
        original_sv = np.linalg.svd(arr.array, compute_uv=False)[0]
        reconstructed_sv = np.linalg.svd(reconstructed.array, compute_uv=False)[0]
        assert np.abs(original_sv - reconstructed_sv) < 0.1

    def test_partial_division_by_zero(self):
        """Test division where only some elements are zero."""
        arr = array([[1, 2], [3, 4]])
        divisor = array([[0, 2], [3, 0]])
        result = arr / divisor
        
        # Check that non-zero divisions are correct
        assert np.isclose(result.array[0, 1], 1.0)  # 2/2
        assert np.isclose(result.array[1, 0], 1.0)  # 3/3

    def test_error_handling(self):
        """Test error conditions and exception handling."""
        arr = array([1, 2, 3])
        
        # Test reconstruction without division
        with pytest.raises(ReconstructionError):
            arr.elevate()
        
        # Test invalid dimensions
        with pytest.raises(DimensionalError):
            array([[[[[1]]]]])  # Too many dimensions
            
    def test_numpy_compatibility(self):
        """Test NumPy function compatibility."""
        arr = array([[1, 2], [3, 4]])
        
        # Test NumPy functions
        assert np.mean(arr.array) == 2.5
        assert arr.sum() == 10
        assert arr.max() == 4
        
        # Test shape and dimension properties
        assert arr.shape == (2, 2)
        assert arr.ndim == 2