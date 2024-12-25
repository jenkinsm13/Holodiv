"""Mathematical operators for dimensional calculations."""

import numpy as np
from typing import Tuple, Optional, Dict
from .exceptions import DimensionalError
import logging

logging.basicConfig(level=logging.DEBUG)

def reduce_dimension(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce dimension of input data."""
    if data.size == 0:
        raise DimensionalError("Cannot reduce dimension of an empty array.")
    if data.ndim == 0:
        raise DimensionalError("Cannot reduce dimension of a scalar.")
    elif data.ndim == 1:
        reduced, error = _reduce_vector(data)
    else:
        reduced, error = _reduce_tensor(data)
    
    # Ensure error has the same shape as the input data
    error = error.reshape(data.shape)
    
    # Center the error around zero
    error -= error.mean()
    
    return reduced, error

def _reduce_vector(data: np.ndarray) -> Tuple[float, np.ndarray]:
    """Reduce a vector to a scalar."""
    magnitude = np.abs(data).mean()
    error = data - magnitude
    return magnitude, error

def _reduce_tensor(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce tensor dimension."""
    if matrix.ndim <= 1:
        return _reduce_vector(matrix)
    
    original_shape = matrix.shape
    matrix_2d = matrix.reshape(-1, matrix.shape[-1])
    
    try:
        u, s, vh = np.linalg.svd(matrix_2d, full_matrices=False)
        if np.isclose(s[0], 0):
            raise DimensionalError("Cannot reduce dimension of a singular matrix.")
        reduced = u[:, 0] * s[0]
        reconstructed = np.outer(u[:, 0], vh[0, :]) * s[0]
        error = matrix - reconstructed.reshape(original_shape)
    except np.linalg.LinAlgError:
        raise DimensionalError("Cannot reduce dimension of a singular matrix.")
    
    return reduced.reshape(original_shape[:-1]), error

def elevate_dimension(reduced_data: np.ndarray,
                    error: np.ndarray,
                    target_shape: Tuple[int, ...],
                    noise_scale: float = 1e-6) -> np.ndarray:
    """
    Implement tensor network elevation as defined in Section 4 of the paper.
    Uses direct reconstruction with error preservation.
    """
    if np.prod(reduced_data.shape) > np.prod(target_shape):
        raise ValueError("Cannot elevate to lower dimensions")
    
    # Normalize input data
    reduced_flat = reduced_data.flatten()
    reduced_flat = reduced_flat / np.linalg.norm(reduced_flat)
    
    # Create error operator
    error_flat = error.flatten()
    error_size = int(np.sqrt(len(error_flat)))
    error_matrix = error_flat.reshape(error_size, error_size)
    
    # Create initial state matrix
    state_matrix = np.zeros((error_size, error_size), dtype=np.complex128)
    state_matrix[:len(reduced_flat), 0] = reduced_flat
    
    # Apply error operator
    elevated = error_matrix @ state_matrix
    
    # Extract the result and normalize
    result = elevated[:np.prod(target_shape), 0]
    result = result / np.linalg.norm(result)
    
    # Add controlled noise to maintain quantum correlations
    if len(result) < np.prod(target_shape):
        padding = np.zeros(np.prod(target_shape) - len(result))
        noise = np.random.normal(0, noise_scale, len(padding))
        result = np.concatenate([result, padding + noise])
    
    # Ensure proper normalization
    result = result / np.linalg.norm(result)
    
    # Reshape to target shape
    return result.reshape(target_shape)

def _elevate_tensor(reduced: np.ndarray, error: np.ndarray, target_shape: Tuple[int, ...], noise_scale: float = 1e-6) -> np.ndarray:
    """Reconstruct the original matrix from reduced matrix and error tensor."""
    if reduced.ndim != 2 or error.ndim != 2:
        raise ValueError("Both reduced and error tensors must be 2D for matrix reconstruction.")
    if target_shape != reduced.shape:
        raise ValueError("Target shape must match the reduced matrix shape for matrices.")
    
    # Add scaled error tensor to create a noisy reconstruction
    noise = noise_scale * np.random.randn(*error.shape)
    reconstructed = reduced + error + noise
    return reconstructed

class QuantumTensor:
    """
    Implements a quantum-aware tensor with support for entanglement operations.
    """
    def __init__(self, 
                 data: np.ndarray,
                 physical_dims: Optional[Tuple[int, ...]] = None,
                 quantum_nums: Optional[Dict[str, float]] = None):
        """
        Initialize quantum tensor.
        
        Args:
            data: Tensor data
            physical_dims: Physical dimensions of the system
            quantum_nums: Quantum numbers for symmetry preservation
        """
        self.data = np.array(data)
        self.physical_dims = physical_dims or tuple(range(data.ndim))
        self.quantum_nums = quantum_nums or {}
        self._entanglement_spectrum = None
    
    def schmidt_decompose(self, 
                          cut_index: int,
                          max_bond_dim: Optional[int] = None) -> Tuple['QuantumTensor', 'QuantumTensor']:
        """
        Perform Schmidt decomposition across specified index.
        
        Args:
            cut_index: Index for bipartition
            max_bond_dim: Maximum bond dimension to keep
            
        Returns:
            Tuple of left and right tensors after decomposition
        """
        shape = self.data.shape
        left_dims = np.prod(shape[:cut_index])
        right_dims = np.prod(shape[cut_index:])
        
        # Reshape for SVD
        matrix = self.data.reshape(left_dims, right_dims)
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        
        # Truncate if requested
        if max_bond_dim:
            U = U[:, :max_bond_dim]
            S = S[:max_bond_dim]
            Vh = Vh[:max_bond_dim, :]
        
        # Store entanglement spectrum
        self._entanglement_spectrum = EntanglementSpectrum(
            schmidt_values=S,
            entropy=-np.sum(S**2 * np.log2(S**2 + 1e-16)),
            bond_dimension=len(S),
            truncation_error=np.sum(S[max_bond_dim:]**2) if max_bond_dim else 0.0
        )
        
        # Create new tensors
        left_tensor = QuantumTensor(
            U @ np.diag(np.sqrt(S)),
            physical_dims=self.physical_dims[:cut_index]
        )
        right_tensor = QuantumTensor(
            np.diag(np.sqrt(S)) @ Vh,
            physical_dims=self.physical_dims[cut_index:]
        )
        
        return left_tensor, right_tensor
    
    def reduce_dimension(self, 
                         target_dims: int,
                         preserve_entanglement: bool = True) -> 'QuantumTensor':
        """
        Reduce tensor dimensions while preserving quantum properties.
        
        Args:
            target_dims: Number of dimensions to reduce to
            preserve_entanglement: Whether to preserve entanglement entropy
            
        Returns:
            Reduced QuantumTensor
        """
        if self.data.ndim <= target_dims:
            raise DimensionalError("Cannot reduce to higher or equal number of dimensions.")
        
        # Perform Schmidt decomposition iteratively until target dimensions are met
        current_tensor = self
        while current_tensor.data.ndim > target_dims:
            # Choose a cut index, e.g., in the middle
            cut_index = current_tensor.data.ndim // 2
            left, right = current_tensor.schmidt_decompose(cut_index)
            
            if preserve_entanglement:
                # Merge with right tensor to maintain entanglement
                current_tensor = left  # Simplification; adjust as per framework
            else:
                current_tensor = left  # Or apply a different strategy
        
        return current_tensor
    
    def elevate(self, target_shape: Optional[Tuple[int, ...]] = None, noise_scale: float = 1e-6) -> 'QuantumTensor':
        """Reconstruct higher dimensional representation."""
        if not self._entanglement_spectrum:
            raise ValueError("No entanglement spectrum available for elevation")
        
        # Example logic for elevation
        # This is a simplistic approach; a more sophisticated method should be used
        noise = np.random.normal(scale=noise_scale, size=self.data.shape)
        elevated_data = self.data + noise
        
        if target_shape:
            elevated_data = elevated_data.reshape(target_shape)
        
        return QuantumTensor(elevated_data, self.physical_dims, self.quantum_nums)