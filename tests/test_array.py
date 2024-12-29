"""Test suite for core array functionality."""

import pytest
import numpy as np
from dividebyzero import array, DimensionalArray
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
        
        # Test SVD failure in dimensional reduction
        bad_matrix = array([[float('nan'), 1], [1, float('nan')]])  # NaN values will definitely cause SVD to fail
        with pytest.raises(DimensionalError):
            bad_matrix._divide_by_zero()

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

    def test_dimensional_reduction_errors(self):
        """Test dimensional reduction errors."""
        # Test dimensional reduction errors
        singular_matrix = array([[float('nan'), 1], [1, float('nan')]])  # Matrix that will fail SVD
        with pytest.raises(DimensionalError):
            singular_matrix._divide_by_zero()  # Should raise error when SVD fails

    def test_complex_operations(self):
        """Test operations with complex numbers."""
        # Test complex array creation and operations
        arr = array([1+1j, 2+2j, 3+3j])
        
        # Test complex multiplication
        result = arr * (1+1j)
        expected = np.array([1+1j, 2+2j, 3+3j]) * (1+1j)
        assert_array_equal_with_tolerance(result.array, expected)
        
        # Test conjugate
        result = arr.conjugate()
        expected = np.array([1-1j, 2-2j, 3-3j])
        assert_array_equal_with_tolerance(result.array, expected)
        
        # Test real and imag properties
        assert_array_equal_with_tolerance(arr.real.array, np.array([1, 2, 3]))
        assert_array_equal_with_tolerance(arr.imag.array, np.array([1, 2, 3]))

    def test_matrix_operations(self):
        """Test matrix multiplication operations."""
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6], [7, 8]])
        
        # Test @ operator
        result = a @ b
        expected = np.array([[19, 22], [43, 50]])
        assert_array_equal_with_tolerance(result.array, expected)
        
        # Test right matrix multiplication
        result = np.array([[1, 2], [3, 4]]) @ a
        assert isinstance(result, DimensionalArray)
        
        # Test matrix-vector multiplication
        c = array([1, 2])
        result = a @ c
        expected = np.array([5, 11])  # [1*1 + 2*2, 3*1 + 4*2]
        assert_array_equal_with_tolerance(result.array, expected)

    def test_comparison_operations(self):
        """Test comparison operations."""
        a = array([1, 2, 3])
        b = array([2, 2, 1])
        
        # Test equality
        eq_result = (a == b)
        assert_array_equal_with_tolerance(eq_result.array, np.array([False, True, False]))
        
        # Test less than
        lt_result = (a < b)
        assert_array_equal_with_tolerance(lt_result.array, np.array([True, False, False]))
        
        # Test greater than
        gt_result = (a > b)
        assert_array_equal_with_tolerance(gt_result.array, np.array([False, False, True]))
        
        # Test comparison with scalar
        scalar_result = (a > 2)
        assert_array_equal_with_tolerance(scalar_result.array, np.array([False, False, True]))

    def test_type_conversions(self):
        """Test type conversion operations."""
        # Test scalar conversions
        scalar = array(3.14)
        assert float(scalar) == 3.14
        assert int(scalar) == 3
        assert complex(scalar) == (3.14 + 0j)
        
        # Test item() method
        assert scalar.item() == 3.14
        
        # Test conversion errors for non-scalar arrays
        arr = array([1, 2, 3])
        with pytest.raises(TypeError):
            float(arr)
        with pytest.raises(TypeError):
            int(arr)
        with pytest.raises(TypeError):
            complex(arr)

    def test_array_reshaping(self):
        """Test array reshaping and transposition operations."""
        # Test reshape
        arr = array([1, 2, 3, 4])
        reshaped = arr.reshape(2, 2)
        assert reshaped.shape == (2, 2)
        assert_array_equal_with_tolerance(reshaped.array, np.array([[1, 2], [3, 4]]))
        
        # Test transpose
        transposed = reshaped.transpose()
        assert transposed.shape == (2, 2)
        assert_array_equal_with_tolerance(transposed.array, np.array([[1, 3], [2, 4]]))
        
        # Test T property
        assert_array_equal_with_tolerance(reshaped.T.array, transposed.array)
        
        # Test flatten
        flattened = reshaped.flatten()
        assert flattened.shape == (4,)
        assert_array_equal_with_tolerance(flattened.array, np.array([1, 2, 3, 4]))

    def test_array_types(self):
        """Test array type casting and dtype operations."""
        # Test dtype property
        arr = array([1, 2, 3])
        assert arr.dtype == np.int64  # or whatever the default int type is
        
        # Test astype conversion
        float_arr = arr.astype(np.float64)
        assert float_arr.dtype == np.float64
        assert_array_equal_with_tolerance(float_arr.array, np.array([1.0, 2.0, 3.0]))
        
        # Test complex type
        complex_arr = arr.astype(np.complex128)
        assert complex_arr.dtype == np.complex128
        assert_array_equal_with_tolerance(complex_arr.array, np.array([1+0j, 2+0j, 3+0j]))
        
        # Test creation with specific dtype
        uint_arr = array([1, 2, 3], dtype=np.uint8)
        assert uint_arr.dtype == np.uint8

    def test_arithmetic_operations(self):
        """Test arithmetic operations not covered by other tests."""
        a = array([1, 2, 3])
        
        # Test negation
        neg = -a
        assert_array_equal_with_tolerance(neg.array, np.array([-1, -2, -3]))
        
        # Test right subtraction
        result = 10 - a
        assert_array_equal_with_tolerance(result.array, np.array([9, 8, 7]))
        
        # Test absolute value
        neg_arr = array([-1, -2, -3])
        abs_result = abs(neg_arr)
        assert_array_equal_with_tolerance(abs_result.array, np.array([1, 2, 3]))
        
        # Test power operation
        pow_result = a ** 2
        assert_array_equal_with_tolerance(pow_result.array, np.array([1, 4, 9]))

    def test_reduction_operations(self):
        """Test array reduction operations with different axis parameters."""
        arr = array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
        
        # Test mean with different axes
        assert_array_equal_with_tolerance(arr.mean(axis=0).array, np.array([4., 5., 6.]))
        assert_array_equal_with_tolerance(arr.mean(axis=1).array, np.array([2., 5., 8.]))
        assert_array_equal_with_tolerance(arr.mean(keepdims=True).array, np.array([[5.]]))
        
        # Test min/max with different axes
        assert_array_equal_with_tolerance(arr.max(axis=0).array, np.array([7, 8, 9]))
        assert_array_equal_with_tolerance(arr.min(axis=1).array, np.array([1, 4, 7]))
        
        # Test sum with keepdims
        assert_array_equal_with_tolerance(arr.sum(axis=0, keepdims=True).array, np.array([[12, 15, 18]]))

    def test_memory_operations(self):
        """Test array memory operations and copying."""
        # Test copy
        original = array([[1, 2], [3, 4]])
        copied = original.copy()
        
        # Verify it's a deep copy
        copied.array[0, 0] = 99
        assert original.array[0, 0] == 1
        
        # Test nbytes property
        assert original.nbytes == original.array.nbytes
        
        # Test array creation from another DimensionalArray
        another = array(original)
        assert_array_equal_with_tolerance(another.array, original.array)
        
        # Verify error registry is preserved
        assert another.error_registry == original.error_registry

    def test_partial_division_higher_dims(self):
        """Test partial division by zero with higher dimensional arrays."""
        # Test 3D array division
        arr = array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape (2, 2, 2)
        divisor = array([[[0, 2], [3, 0]], [[5, 0], [0, 8]]])
        result = arr / divisor
        
        # Check shape preservation
        assert result.shape == arr.shape
        
        # Check non-zero divisions are correct
        assert np.isclose(result.array[0, 0, 1], 1.0)  # 2/2
        assert np.isclose(result.array[0, 1, 0], 1.0)  # 3/3
        assert np.isclose(result.array[1, 0, 0], 1.0)  # 5/5
        assert np.isclose(result.array[1, 1, 1], 1.0)  # 8/8
        
        # Test reconstruction
        elevated = result.elevate()
        assert elevated.shape == arr.shape
        
        # Check information preservation for non-zero elements
        non_zero_mask = divisor.array != 0
        assert np.allclose(
            elevated.array[non_zero_mask],
            arr.array[non_zero_mask] / divisor.array[non_zero_mask],
            rtol=0.1
        )

    def test_broadcasting(self):
        """Test array broadcasting operations."""
        # Create arrays with different shapes
        a = array([[1, 2, 3],
                  [4, 5, 6]])  # Shape (2, 3)
        b = array([10, 20, 30])  # Shape (3,)
        
        # Test broadcasting in addition
        result = a + b
        expected = np.array([[11, 22, 33],
                           [14, 25, 36]])
        assert_array_equal_with_tolerance(result.array, expected)
        
        # Test broadcasting in multiplication
        result = a * b
        expected = np.array([[10, 40, 90],
                           [40, 100, 180]])
        assert_array_equal_with_tolerance(result.array, expected)
        
        # Test broadcasting with scalars in comparison
        mask = a > 3
        expected = np.array([[False, False, False],
                           [True, True, True]])
        assert_array_equal_with_tolerance(mask.array, expected)

    def test_array_function_protocol(self):
        """Test NumPy function protocol implementation."""
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6], [7, 8]])
        
        # Test basic NumPy functions
        result = np.concatenate([a, b])
        assert result.shape == (4, 2)
        
        # Test with mixed inputs
        result = np.concatenate([a, [[9, 10]]])
        assert result.shape == (3, 2)
        
        # Test with keyword arguments
        result = np.concatenate([a, b], axis=1)
        assert result.shape == (2, 4)
        
        # Test function returning tuple
        u, s, vh = np.linalg.svd(a)
        assert isinstance(u, DimensionalArray)
        assert isinstance(s, DimensionalArray)
        assert isinstance(vh, DimensionalArray)

    def test_partial_divide_by_zero_edge_cases(self):
        """Test edge cases in partial division by zero."""
        # Test 1D array with all zeros
        arr = array([0, 0, 0])
        divisor = array([0, 1, 0])
        result = arr / divisor
        assert result.shape == (3,)
        
        # Test 2D array with zero row
        arr = array([[1, 2], [0, 0]])
        divisor = array([[1, 0], [0, 0]])
        result = arr / divisor
        assert result.shape == (2, 2)
        
        # Test 3D array with all zeros in one slice
        arr_3d = array([[[1, 2], [3, 4]], [[0, 0], [0, 0]]])
        divisor_3d = array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
        result = arr_3d / divisor_3d
        assert result.shape == (2, 2, 2)
        
        # Test reconstruction
        elevated = result.elevate()
        assert elevated.shape == arr_3d.shape
        # Check information preservation in non-zero slice
        assert np.allclose(elevated.array[0], arr_3d.array[0], rtol=0.1)

    def test_elevation_methods(self):
        """Test elevation with different shapes and noise scales."""
        # Test complete elevation
        arr = array([[1, 2], [3, 4]])
        reduced = arr / 0
        elevated = reduced.elevate(noise_scale=0.0)  # No noise for deterministic test
        assert elevated.shape == arr.shape
        
        # Test partial elevation
        arr = array([1, 2, 3])
        divisor = array([1, 0, 1])
        result = arr / divisor
        elevated = result.elevate(noise_scale=0.0)
        assert elevated.shape == arr.shape
        
        # Test elevation errors
        arr = array([1, 2, 3])
        with pytest.raises(ReconstructionError):
            arr.elevate()  # No error info available

    def test_copy_and_deepcopy(self):
        """Test copy operations."""
        import copy
        original = array([[1, 2], [3, 4]])
        
        # Test shallow copy
        shallow = copy.copy(original)
        shallow.array[0, 0] = 99
        assert original.array[0, 0] == 1
        
        # Test deep copy
        deep = copy.deepcopy(original)
        deep.array[0, 0] = 99
        assert original.array[0, 0] == 1
        
        # Test error registry copying
        assert deep.error_registry is not original.error_registry

    def test_array_conversion(self):
        """Test NumPy array conversion."""
        arr = array([[1, 2], [3, 4]])
        
        # Test direct conversion
        np_arr = np.array(arr)
        assert isinstance(np_arr, np.ndarray)
        assert np.array_equal(np_arr, arr.array)
        
        # Test with dtype
        np_arr = np.array(arr, dtype=np.float32)
        assert np_arr.dtype == np.float32

    def test_type_conversion_errors(self):
        """Test error handling in type conversions."""
        arr = array([[1, 2], [3, 4]])
        
        with pytest.raises(TypeError):
            float(arr)
        
        with pytest.raises(TypeError):
            int(arr)
        
        with pytest.raises(TypeError):
            complex(arr)
        
        # Test scalar conversions
        scalar = array(3.14)
        assert float(scalar) == 3.14
        assert int(scalar) == 3
        assert complex(scalar) == (3.14 + 0j)

    def test_advanced_broadcasting(self):
        """Test advanced broadcasting operations."""
        # Test broadcasting with different dimensions
        a = array([[[1, 2]], [[3, 4]]])  # Shape (2, 1, 2)
        b = array([10, 20])              # Shape (2,)
        
        result = a + b
        assert result.shape == (2, 1, 2)
        
        # Test broadcasting in multiplication
        result = a * b[:, None, None]
        assert result.shape == (2, 1, 2)
        
        # Test broadcasting in comparison
        mask = a > array([2])
        assert mask.shape == (2, 1, 2)

    def test_error_registry_handling(self):
        """Test error registry handling in operations."""
        from dividebyzero import get_registry
        from dividebyzero import DimensionalArray
        
        # Test custom registry
        custom_registry = get_registry()
        arr = DimensionalArray([1, 2, 3], error_registry=custom_registry)  # Use DimensionalArray directly
        assert arr.error_registry is custom_registry
        
        # Test registry propagation
        result = arr + 1
        assert result.error_registry is custom_registry
        
        # Test registry in division by zero
        result = arr / 0
        assert result.error_registry is custom_registry
        assert result._error_id is not None

    def test_comprehensive_matrix_operations(self):
        """Test comprehensive matrix multiplication operations matching NumPy behavior."""
        # 2D @ 2D
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6], [7, 8]])
        result = a @ b
        expected = np.array([[19, 22], [43, 50]])
        assert_array_equal_with_tolerance(result.array, expected)
        
        # 2D @ 1D
        c = array([1, 2])
        result = a @ c
        expected = np.array([5, 11])
        assert_array_equal_with_tolerance(result.array, expected)
        
        # 1D @ 2D
        result = c @ a
        expected = np.array([7, 10])
        assert_array_equal_with_tolerance(result.array, expected)
        
        # Test broadcasting with higher dimensions
        # 3D @ 2D
        d = array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: (2, 2, 2)
        result = d @ b
        expected = np.array([[[19, 22], [43, 50]], [[67, 78], [91, 106]]])
        assert_array_equal_with_tolerance(result.array, expected)
        
        # Test with numpy arrays
        result = np.array([[1, 2], [3, 4]]) @ a
        assert isinstance(result, DimensionalArray)
        assert_array_equal_with_tolerance(result.array, np.array([[7, 10], [15, 22]]))
        
        # Test with mixed types
        result = a @ [1, 2]  # List should work too
        assert_array_equal_with_tolerance(result.array, np.array([5, 11]))
        
        # Ensure divide by zero still works after matmul
        result = (a @ b) / 0
        assert result._error_id is not None  # Should have error info
        elevated = result.elevate()
        assert elevated.shape == (2, 2)  # Should restore original shape

    def test_numpy_ufunc_compatibility(self):
        """Test compatibility with NumPy universal functions."""
        a = array([[1, 2], [3, 4]])
        
        # Test basic ufuncs
        assert_array_equal_with_tolerance(np.sin(a).array, np.sin(a.array))
        assert_array_equal_with_tolerance(np.exp(a).array, np.exp(a.array))
        assert_array_equal_with_tolerance(np.log(a).array, np.log(a.array))
        
        # Test ufuncs with multiple arguments
        b = array([[5, 6], [7, 8]])
        assert_array_equal_with_tolerance(np.maximum(a, b).array, np.maximum(a.array, b.array))
        assert_array_equal_with_tolerance(np.minimum(a, b).array, np.minimum(a.array, b.array))
        
        # Test ufuncs that return multiple arrays
        magnitude, angle = np.frexp(a)
        assert isinstance(magnitude, DimensionalArray)
        assert isinstance(angle, DimensionalArray)
        
        # Test with mixed types
        result = np.add(a, [[1, 2], [3, 4]])
        assert isinstance(result, DimensionalArray)
        
    def test_numpy_function_compatibility(self):
        """Test compatibility with general NumPy functions."""
        a = array([[1, 2], [3, 4]])
        
        # Test array manipulation
        assert_array_equal_with_tolerance(np.transpose(a).array, a.array.T)
        assert_array_equal_with_tolerance(np.flip(a).array, np.flip(a.array))
        assert_array_equal_with_tolerance(np.rot90(a).array, np.rot90(a.array))
        
        # Test shape manipulation
        assert_array_equal_with_tolerance(np.reshape(a, (4,)).array, a.array.reshape(4))
        assert_array_equal_with_tolerance(np.ravel(a).array, a.array.ravel())
        
        # Test stacking operations
        b = array([[5, 6], [7, 8]])
        assert_array_equal_with_tolerance(np.vstack((a, b)).array, np.vstack((a.array, b.array)))
        assert_array_equal_with_tolerance(np.hstack((a, b)).array, np.hstack((a.array, b.array)))
        
        # Test splitting operations
        splits = np.array_split(a, 2, axis=0)  # array_split returns a list of arrays
        assert all(isinstance(split, np.ndarray) for split in splits)
        # Verify the content is correct
        expected_splits = np.array_split(a.array, 2, axis=0)
        for split, expected in zip(splits, expected_splits):
            assert_array_equal_with_tolerance(split, expected)
        
    def test_numpy_linalg_compatibility(self):
        """Test compatibility with NumPy linear algebra operations."""
        a = array([[1, 2], [3, 4]])
        
        # Test basic linear algebra
        assert_array_equal_with_tolerance(np.linalg.inv(a).array, np.linalg.inv(a.array))
        
        # Test determinant (returns scalar)
        det_val = np.linalg.det(a)
        assert isinstance(det_val, (float, np.floating))
        assert np.isclose(det_val, np.linalg.det(a.array))
        
        # Test decompositions
        u, s, vh = np.linalg.svd(a)
        assert isinstance(u, DimensionalArray)
        assert isinstance(s, DimensionalArray)
        assert isinstance(vh, DimensionalArray)
        
        # Test eigenvalues
        w, v = np.linalg.eig(a)
        assert isinstance(w, DimensionalArray)
        assert isinstance(v, DimensionalArray)
        
        # Test matrix operations
        assert_array_equal_with_tolerance(np.linalg.matrix_power(a, 2).array, np.linalg.matrix_power(a.array, 2))
        
    def test_mixed_operations(self):
        """Test mixing NumPy operations with divide-by-zero functionality."""
        a = array([[1, 2], [3, 4]])
        
        # Test NumPy op followed by division
        result = np.sin(a) / 0
        assert result._error_id is not None
        elevated = result.elevate()
        assert elevated.shape == (2, 2)
        
        # Test division followed by NumPy op
        result = np.exp(a / 0)
        assert isinstance(result, DimensionalArray)
        
        # Test complex chains
        b = array([[5, 6], [7, 8]])
        # Division by zero should happen first to ensure error info is preserved
        zero_div = a / 0
        assert zero_div._error_id is not None
        result = np.matmul(zero_div, np.sin(b))
        assert isinstance(result, DimensionalArray)
        assert result._error_id is not None  # Error info should be preserved
        
        # Test broadcasting with division
        c = array([0, 1])
        result = a / c  # Broadcasting division by zero
        assert result._error_id is not None
        elevated = result.elevate()
        assert elevated.shape == (2, 2)