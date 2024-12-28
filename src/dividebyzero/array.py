"""
Core array implementation with division by zero support.
"""

import numpy as np
from typing import Union, Optional, Tuple, Any
from .registry import ErrorRegistry, ErrorData
from .exceptions import ReconstructionError, DimensionalError
import copy

class DimensionalArray:
    """
    Array class supporting division by zero through dimensional reduction.
    """
    def __init__(self, 
                 array_like: Union[np.ndarray, list, tuple, 'DimensionalArray'],
                 error_registry: Optional[ErrorRegistry] = None,
                 dtype: Any = None):
        """
        Initialize a DimensionalArray.
        
        Parameters:
            array_like: Input array or array-like object
            error_registry: Optional error registry for tracking dimensional operations
            dtype: Data type for the array (numpy dtype)
        """
        if isinstance(array_like, DimensionalArray):
            self.array = array_like.array.copy()
            self._right_singular_vector = getattr(array_like, '_right_singular_vector', None)
        else:
            self.array = np.array(array_like, dtype=dtype)
            self._right_singular_vector = None
            
        from . import get_registry
        self.error_registry = error_registry or get_registry()
        self._error_id = None

    def __setitem__(self, key, value):
        """Support item assignment."""
        if isinstance(value, DimensionalArray):
            self.array[key] = value.array
        else:
            self.array[key] = value

    def __getitem__(self, key):
        """Support item access."""
        result = self.array[key]
        if isinstance(result, np.ndarray):
            return DimensionalArray(result, self.error_registry)
        return result
    
    def __array_function__(self, func, types, args, kwargs):
        """Implement numpy function protocol."""
        def convert_to_array(obj):
            if isinstance(obj, DimensionalArray):
                return obj.array
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert_to_array(x) for x in obj)
            return obj
        
        # Convert inputs to numpy arrays, handling nested structures
        args = tuple(convert_to_array(arg) for arg in args)
        kwargs = {k: convert_to_array(v) for k, v in kwargs.items()}
        
        # Call the numpy function directly
        result = func(*args, **kwargs)
        
        # Convert result back to DimensionalArray if needed
        if isinstance(result, np.ndarray):
            return DimensionalArray(result, self.error_registry)
        elif isinstance(result, tuple):
            return tuple(DimensionalArray(r, self.error_registry) if isinstance(r, np.ndarray) else r for r in result)
        return result
    
    def __mul__(self, other: Union[int, float, complex, 'DimensionalArray']) -> 'DimensionalArray':
        """Support multiplication with scalars (including complex) or other DimensionalArray instances."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array * other.array, self.error_registry)
        return DimensionalArray(self.array * other, self.error_registry)

    def __rmul__(self, other: Union[int, float, complex]) -> 'DimensionalArray':
        """Support right multiplication with scalars (including complex)."""
        return DimensionalArray(other * self.array, self.error_registry)

    def __rtruediv__(self, other: Union[int, float, complex]) -> 'DimensionalArray':
        """Support right division (other / self)."""
        # Handle division by zero using dimensional reduction if needed
        if np.any(self.array == 0):
            mask = self.array == 0
            result = np.zeros_like(self.array, dtype=float)
            non_zero_mask = ~mask
            result[non_zero_mask] = other / self.array[non_zero_mask]
            # For zero elements, use dimensional reduction
            if np.any(mask):
                reduced = self._partial_divide_by_zero(mask)
                result[mask] = other / reduced.array[mask]
            return DimensionalArray(result, self.error_registry)
        return DimensionalArray(other / self.array, self.error_registry)

    def _divide_by_zero(self) -> 'DimensionalArray':
        """Implement complete division by zero"""
        original_shape = self.array.shape
        ndim = self.array.ndim
        
        if ndim == 0:  # scalar case
            result = np.array([np.abs(self.array)])
            error = self.array - result[0]
        elif ndim == 1:
            # For 1D arrays, reduce to a single value
            result = np.array([np.abs(self.array).mean()])
            error = self.array - result[0]
        else:
            # For higher dimensions, use SVD
            reshaped = self.array.reshape(original_shape[0], -1)
            try:
                U, S, Vt = np.linalg.svd(reshaped, full_matrices=False)
                # Take only the first singular value and vector
                result = U[:, 0] * S[0]
                # Store the first right singular vector for reconstruction
                self._right_singular_vector = Vt[0, :]
                error = self.array - np.outer(result, self._right_singular_vector).reshape(original_shape)
            except np.linalg.LinAlgError:
                raise DimensionalError("SVD failed during division by zero")
        
        # Store error information
        error_data = ErrorData(
            original_shape=original_shape,
            error_tensor=error,
            reduction_type='complete'
        )
        error_id = self.error_registry.store(error_data)
        
        reduced = DimensionalArray(result, self.error_registry)
        reduced._error_id = error_id
        reduced._right_singular_vector = getattr(self, '_right_singular_vector', None)
        return reduced
    
    def _partial_divide_by_zero(self, mask: np.ndarray) -> 'DimensionalArray':
        """Handle partial division by zero with proper dimensional reduction."""
        result = np.zeros_like(self.array, dtype=float)
        non_zero_mask = ~mask
        
        # Perform division where divisor is non-zero
        np.divide(self.array, non_zero_mask, out=result, where=non_zero_mask, casting='unsafe')
        
        # Handle zero-division cases
        if self.array.ndim == 1:
            # For 1D arrays, replace zero-division results with mean of non-zero elements
            non_zero_elements = self.array[non_zero_mask]
            if non_zero_elements.size > 0:
                zero_division_value = non_zero_elements.mean()
            else:
                zero_division_value = 0
            result[mask] = zero_division_value
        elif self.array.ndim == 2:
            # For 2D arrays, perform row-wise or column-wise reduction
            for i in range(self.array.shape[0]):
                row_mask = mask[i, :]
                if np.all(row_mask):
                    # If entire row is divided by zero, use mean of the row
                    result[i, row_mask] = self.array[i, :].mean()
                elif np.any(row_mask):
                    # If some elements in the row are divided by zero, use mean of non-zero elements
                    non_zero_elements = self.array[i, ~row_mask]
                    result[i, row_mask] = non_zero_elements.mean()
        else:
            # For higher dimensions, extend the reduction strategy accordingly
            # Implement tensor decomposition methods if necessary
            raise NotImplementedError("Partial division by zero not implemented for ndim > 2.")
        
        error_tensor = np.where(mask, self.array - result, 0)
        error_id = self.error_registry.store(ErrorData(
            original_shape=self.array.shape,
            error_tensor=error_tensor,
            reduction_type='partial'
        ))
        reduced = DimensionalArray(result, self.error_registry)
        reduced._error_id = error_id
        return reduced
    
    def elevate(self, 
                target_shape: Optional[Tuple[int, ...]] = None,
                noise_scale: float = 1e-6) -> 'DimensionalArray':
        """
        Reconstruct higher dimensional representation.
        
        Parameters:
            target_shape: Optional shape for reconstruction
            noise_scale: Scale of random fluctuations in reconstruction
        """
        if not self._error_id:
            raise ReconstructionError("No error information available for elevation")
            
        error_data = self.error_registry.retrieve(self._error_id)
        if not error_data:
            raise ReconstructionError("Error information has been garbage collected")
        
        if error_data.reduction_type == 'complete':
            return self._complete_elevation(error_data, noise_scale)
        else:
            return self._partial_elevation(error_data, noise_scale)
        
    def elevate_dimension(reduced: 'DimensionalArray', 
                        error: 'DimensionalArray', 
                        original_shape: Tuple[int, ...],
                        noise_scale: float = 1.0) -> 'DimensionalArray':
        """
        Reconstruct the original dimensional array from its reduced form and error tensor.
        """
        reconstructed = reduced.array.reshape(original_shape)
        elevated = reconstructed + error.array * noise_scale
        return DimensionalArray(elevated, reduced.error_registry)
    
    def _complete_elevation(self, 
                          error_data: ErrorData,
                          noise_scale: float) -> 'DimensionalArray':
        """Handle elevation for complete reduction"""
        noise = np.random.normal(
            scale=noise_scale,
            size=error_data.original_shape
        )
        
        if self.array.size == 1:  # scalar case
            reconstructed = np.full(error_data.original_shape, self.array[0])
        elif self.array.ndim == 1 and self._right_singular_vector is not None:
            reconstructed = np.outer(self.array, self._right_singular_vector).reshape(error_data.original_shape)
        else:
            reconstructed = self.array.reshape(error_data.original_shape)
            
        result = reconstructed + error_data.error_tensor * noise
        return DimensionalArray(result, self.error_registry)
    
    def _partial_elevation(self,
                         error_data: ErrorData,
                         noise_scale: float) -> 'DimensionalArray':
        """Handle elevation for partial reduction"""
        result = self.array.copy()
        noise = np.random.normal(
            scale=noise_scale,
            size=error_data.original_shape
        )
        
        # Apply elevation only where reduction occurred
        mask = error_data.mask
        result[mask] += error_data.error_tensor[mask] * noise[mask]
        
        return DimensionalArray(result, self.error_registry)
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from underlying numpy array if not found in DimensionalArray."""
        if name == '_left_singular_vector':
            return self._left_singular_vector
        if name == '_right_singular_vector':
            return self._right_singular_vector
        if name == '_error_registry':
            return self._error_registry
        if name == '_array':
            return self._array
        return getattr(self.array, name)
    
    def __repr__(self) -> str:
        return f"DimensionalArray({self.array.__repr__()})"
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape
    
    @property
    def ndim(self) -> int:
        return self.array.ndim
    
    def __mul__(self, other: Union[int, float, complex, 'DimensionalArray']) -> 'DimensionalArray':
        """Support multiplication with scalars (including complex) or other DimensionalArray instances."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array * other.array, self.error_registry)
        return DimensionalArray(self.array * other, self.error_registry)
    
    def __rmul__(self, other: Union[int, float, complex]) -> 'DimensionalArray':
        """Support right multiplication with scalars (including complex)."""
        return DimensionalArray(other * self.array, self.error_registry)
    
    def __truediv__(self, other: Union[int, float, 'DimensionalArray']) -> 'DimensionalArray':
        """Support division with scalars or other DimensionalArray instances."""
        if isinstance(other, (int, float)):
            if other == 0:
                return self._divide_by_zero()
            return DimensionalArray(self.array / other, self.error_registry)
        if isinstance(other, DimensionalArray):
            if np.any(other.array == 0):
                # Handle division by zero using dimensional reduction
                mask = other.array == 0
                result = self._partial_divide_by_zero(mask)
                # Perform regular division for non-zero elements
                non_zero_mask = ~mask
                result.array[non_zero_mask] = self.array[non_zero_mask] / other.array[non_zero_mask]
                return result
            return DimensionalArray(self.array / other.array, self.error_registry)
        raise TypeError(f"Unsupported type for division: {type(other)}")
    
    def __add__(self, other: Union[int, float, complex, 'DimensionalArray']) -> 'DimensionalArray':
        """Support addition with scalars or other DimensionalArray instances."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array + other.array, self.error_registry)
        return DimensionalArray(self.array + other, self.error_registry)
    
    def __radd__(self, other: Union[int, float, complex]) -> 'DimensionalArray':
        """Support right addition with scalars."""
        # This is called when other + self is invoked
        # Needed for sum() to work properly
        return DimensionalArray(other + self.array, self.error_registry)
        
    def __sub__(self, other: Union[int, float, 'DimensionalArray']) -> 'DimensionalArray':
        """Support subtraction with scalars or other DimensionalArray instances."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array - other.array, self.error_registry)
        return DimensionalArray(self.array - other, self.error_registry)
    
    def __float__(self):
        """Convert to float. Only works for scalar arrays."""
        if self.array.size != 1:
            raise TypeError("Only length-1 arrays can be converted to Python scalars")
        return float(self.array.item())
    
    def __int__(self):
        """Convert to int. Only works for scalar arrays."""
        if self.array.size != 1:
            raise TypeError("Only length-1 arrays can be converted to Python scalars")
        return int(self.array.item())
    
    def __complex__(self):
        """Convert to complex. Only works for scalar arrays."""
        if self.array.size != 1:
            raise TypeError("Only length-1 arrays can be converted to Python scalars")
        return complex(self.array.item())
    
    def item(self):
        """Get the scalar value for single-element arrays."""
        return self.array.item()
    
    def __array__(self, dtype=None):
        """Convert to numpy array."""
        return np.asarray(self.array, dtype=dtype)
    
    def __pow__(self, power, modulo=None):
        """Support power operation."""
        if modulo is None:
            return DimensionalArray(np.power(self.array, power), self.error_registry)
        else:
            return DimensionalArray(pow(self.array, power, modulo), self.error_registry)
    
    def __matmul__(self, other: 'DimensionalArray') -> 'DimensionalArray':
        """Support matrix multiplication with @ operator."""
        if isinstance(other, DimensionalArray):
            other_array = other.array
        else:
            other_array = np.asarray(other)
        return DimensionalArray(self.array @ other_array, self.error_registry)
    
    def __rmatmul__(self, other) -> 'DimensionalArray':
        """Support right matrix multiplication with @ operator."""
        other_array = np.asarray(other)
        return DimensionalArray(other_array @ self.array, self.error_registry)

    def __neg__(self) -> 'DimensionalArray':
        """Support negation (-x)."""
        return DimensionalArray(-self.array, self.error_registry)

    def __rsub__(self, other: Union[int, float, complex]) -> 'DimensionalArray':
        """Support right subtraction."""
        return DimensionalArray(other - self.array, self.error_registry)

    def __abs__(self) -> 'DimensionalArray':
        """Support absolute value."""
        return DimensionalArray(np.abs(self.array), self.error_registry)
    
    def __eq__(self, other) -> 'DimensionalArray':
        """Support equality comparison."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array == other.array, self.error_registry)
        return DimensionalArray(self.array == other, self.error_registry)

    def __lt__(self, other) -> 'DimensionalArray':
        """Support less than comparison."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array < other.array, self.error_registry)
        return DimensionalArray(self.array < other, self.error_registry)

    def __le__(self, other) -> 'DimensionalArray':
        """Support less than or equal comparison."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array <= other.array, self.error_registry)
        return DimensionalArray(self.array <= other, self.error_registry)

    def __gt__(self, other) -> 'DimensionalArray':
        """Support greater than comparison."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array > other.array, self.error_registry)
        return DimensionalArray(self.array > other, self.error_registry)

    def __ge__(self, other) -> 'DimensionalArray':
        """Support greater than or equal comparison."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array >= other.array, self.error_registry)
        return DimensionalArray(self.array >= other, self.error_registry)
    
    def reshape(self, *shape) -> 'DimensionalArray':
        """Reshape array."""
        return DimensionalArray(self.array.reshape(*shape), self.error_registry)

    def transpose(self, *axes) -> 'DimensionalArray':
        """Transpose array."""
        return DimensionalArray(self.array.transpose(*axes), self.error_registry)

    def flatten(self) -> 'DimensionalArray':
        """Flatten array to 1D."""
        return DimensionalArray(self.array.flatten(), self.error_registry)

    @property
    def T(self) -> 'DimensionalArray':
        """Array transpose."""
        return DimensionalArray(self.array.T, self.error_registry)
    
    @property
    def dtype(self):
        """Array data type."""
        return self.array.dtype

    def astype(self, dtype) -> 'DimensionalArray':
        """Cast array to specified type."""
        return DimensionalArray(self.array.astype(dtype), self.error_registry)

    def conjugate(self) -> 'DimensionalArray':
        """Complex conjugate."""
        return DimensionalArray(self.array.conjugate(), self.error_registry)

    @property
    def real(self) -> 'DimensionalArray':
        """Real part of array."""
        return DimensionalArray(self.array.real, self.error_registry)

    @property
    def imag(self) -> 'DimensionalArray':
        """Imaginary part of array."""
        return DimensionalArray(self.array.imag, self.error_registry)
    
    def sum(self, axis=None, keepdims=False):
        """Sum of array elements.
        
        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which to perform the sum
        keepdims : bool, optional
            If True, the axes which are reduced are left as dimensions of size one
            
        Returns
        -------
        sum_along_axis : DimensionalArray or float
            An array with the same shape as self, with specified axes removed if keepdims=False.
            If the sum is performed over the entire array, a float is returned.
        """
        result = self.array.sum(axis=axis, keepdims=keepdims)
        return DimensionalArray(result, self.error_registry) if isinstance(result, np.ndarray) else result

    def mean(self, axis=None, keepdims=False) -> 'DimensionalArray':
        """Mean of array elements."""
        return DimensionalArray(self.array.mean(axis=axis, keepdims=keepdims), self.error_registry)

    def max(self, axis=None, keepdims=False):
        """Maximum of array elements."""
        result = self.array.max(axis=axis, keepdims=keepdims)
        return DimensionalArray(result, self.error_registry) if isinstance(result, np.ndarray) else float(result)

    def min(self, axis=None, keepdims=False):
        """Minimum of array elements."""
        result = self.array.min(axis=axis, keepdims=keepdims)
        return DimensionalArray(result, self.error_registry) if isinstance(result, np.ndarray) else float(result)
    
    def copy(self) -> 'DimensionalArray':
        """Return a copy of the array."""
        return DimensionalArray(self.array.copy(), self.error_registry)

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the array."""
        return self.array.nbytes
    
    def __len__(self) -> int:
        """Support len() function."""
        return len(self.array)
    
    def __copy__(self):
        """Return a shallow copy of the DimensionalArray."""
        return DimensionalArray(self.array.copy(), error_registry=self.error_registry)

    def __deepcopy__(self, memo):
        """Return a deep copy of the DimensionalArray."""
        array_copy = copy.deepcopy(self.array, memo)
        registry_copy = copy.deepcopy(self.error_registry, memo)
        return DimensionalArray(array_copy, error_registry=registry_copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Implement numpy ufunc protocol."""
        if method != '__call__':
            return NotImplemented

        def convert_to_array(obj):
            if isinstance(obj, DimensionalArray):
                return obj.array
            return obj

        inputs = tuple(convert_to_array(x) for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if isinstance(result, np.ndarray):
            return DimensionalArray(result, self.error_registry)
        elif isinstance(result, tuple):
            return tuple(DimensionalArray(x, self.error_registry) if isinstance(x, np.ndarray) else x for x in result)
        return result

def array(array_like: Any, dtype: Any = None, error_registry: Optional[ErrorRegistry] = None) -> DimensionalArray:
    """
    Create a DimensionalArray.
    
    Parameters:
        array_like: Input array or array-like object
        dtype: Data type for the array (numpy dtype)
        error_registry: Optional error registry for tracking dimensional operations
    
    Returns:
        DimensionalArray: A new array instance
    """
    return DimensionalArray(array_like, error_registry=error_registry, dtype=dtype)
