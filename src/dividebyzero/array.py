"""
Core array implementation with division by zero support.
"""

import numpy as np
from typing import Union, Optional, Tuple, Any
from .registry import ErrorRegistry, ErrorData
from .exceptions import ReconstructionError, DimensionalError

class DimensionalArray:
    """
    Array class supporting division by zero through dimensional reduction.
    """
    def __init__(self, 
                 array_like: Union[np.ndarray, list, tuple, 'DimensionalArray'],
                 error_registry: Optional[ErrorRegistry] = None):
        """
        Initialize a DimensionalArray.
        
        Parameters:
            array_like: Input array or array-like object
            error_registry: Optional error registry for tracking dimensional operations
        """
        if isinstance(array_like, DimensionalArray):
            self.array = array_like.array.copy()
            self._right_singular_vector = getattr(array_like, '_right_singular_vector', None)
        else:
            self.array = np.array(array_like)
            if self.array.ndim > 3:  # Example condition for invalid dimensions
                raise DimensionalError("Too many dimensions")
            self._right_singular_vector = None
            
        from . import get_registry
        self.error_registry = error_registry or get_registry()
        self._error_id = None
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle NumPy universal functions"""
        if ufunc == np.true_divide:
            # Check each divisor element for zero
            divisor = inputs[1]
            if isinstance(divisor, (int, float)):
                if divisor == 0:
                    return self._divide_by_zero()
            elif isinstance(divisor, (np.ndarray, DimensionalArray)):
                div_array = divisor.array if isinstance(divisor, DimensionalArray) else divisor
                if np.any(div_array == 0):
                    # Apply dimensional reduction where divisor is zero
                    mask = div_array == 0
                    result = self._partial_divide_by_zero(mask)
                    return result
            
        # For all other operations, pass through to NumPy
        arrays = [(x.array if isinstance(x, DimensionalArray) else x) for x in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)
        return DimensionalArray(result, self.error_registry)
    
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
        """Pass through NumPy attributes"""
        if name == '_right_singular_vector':
            return self._right_singular_vector
        return getattr(self.array, name)
    
    def __repr__(self) -> str:
        return f"DimensionalArray({self.array.__repr__()})"
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape
    
    @property
    def ndim(self) -> int:
        return self.array.ndim
    
    def __mul__(self, other: Union[int, float, 'DimensionalArray']) -> 'DimensionalArray':
        """Support multiplication with scalars or other DimensionalArray instances."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array * other.array, self.error_registry)
        return DimensionalArray(self.array * other, self.error_registry)
    
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
    
    def __add__(self, other: Union[int, float, 'DimensionalArray']) -> 'DimensionalArray':
        """Support addition with scalars or other DimensionalArray instances."""
        if isinstance(other, DimensionalArray):
            return DimensionalArray(self.array + other.array, self.error_registry)
        return DimensionalArray(self.array + other, self.error_registry)

def array(array_like: Union[np.ndarray, list, tuple, DimensionalArray]) -> DimensionalArray:
    """Create a DimensionalArray from array-like input."""
    return DimensionalArray(array_like)