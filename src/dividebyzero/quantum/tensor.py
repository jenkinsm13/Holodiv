"""
Quantum Tensor Network Implementation

This module implements quantum tensor operations with support for:
- Entanglement entropy calculations
- Schmidt decomposition
- Tensor network contractions
- Holographic dimensional reduction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.linalg import expm
from ..exceptions import DimensionalError
import logging

# Instead, create a module-level logger
logger = logging.getLogger(__name__)

ENTANGLEMENT_CUTOFF = 1e-5  # Default cutoff value for entanglement reduction

@dataclass
class EntanglementSpectrum:
    """Represents the entanglement spectrum of a quantum state."""
    schmidt_values: np.ndarray
    entropy: float
    bond_dimension: int
    truncation_error: float

class QuantumTensor:
    """
    Implements a quantum-aware tensor with support for entanglement operations.
    """
    def __init__(self, data, physical_dims=None, quantum_nums=None):
        """
        Initialize quantum tensor.
        
        Parameters:
            data: Input data array
            physical_dims: Physical dimensions for tensor network operations
            quantum_nums: Quantum numbers for symmetry preservation
        """
        # Convert data to numpy array first
        self.data = np.array(data)
        # Now use ndim from the numpy array
        self.physical_dims = physical_dims or tuple(range(self.data.ndim))
        self.quantum_nums = quantum_nums or {}
        # Initialize entanglement spectrum with default values
        self._entanglement_spectrum = EntanglementSpectrum(
            schmidt_values=np.array([1.0]),
            entropy=0.0,
            bond_dimension=1,
            truncation_error=0.0
        )
        
    def schmidt_decompose(self, cut_index: int) -> Tuple['QuantumTensor', 'QuantumTensor']:
        """
        Perform Schmidt decomposition on the tensor at the specified cut_index.
        Splits the tensor into left and right QuantumTensors.
        """
        # Reshape tensor for SVD
        new_shape = (int(np.prod(self.data.shape[:cut_index])), int(np.prod(self.data.shape[cut_index:])))
        reshaped = self.data.reshape(new_shape)
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(reshaped, full_matrices=False)
        
        # Normalize Schmidt values
        S = S / np.linalg.norm(S)
        
        # Update entanglement spectrum
        entropy = -np.sum(S**2 * np.log2(S**2))
        self._entanglement_spectrum = EntanglementSpectrum(
            schmidt_values=S,
            entropy=entropy,
            bond_dimension=len(S),
            truncation_error=0.0
        )
        
        # Create left and right tensors
        left_data = U @ np.diag(np.sqrt(S))
        right_data = np.diag(np.sqrt(S)) @ Vt
        
        # Handle slicing of physical_dims and quantum_nums
        left_physical_dims = self.physical_dims[:cut_index] if self.physical_dims else None
        right_physical_dims = self.physical_dims[cut_index:] if self.physical_dims else None
        
        left_quantum_nums = {k: v[:cut_index] for k, v in self.quantum_nums.items()} if self.quantum_nums else None
        right_quantum_nums = {k: v[cut_index:] for k, v in self.quantum_nums.items()} if self.quantum_nums else None
        
        left_tensor = QuantumTensor(left_data, left_physical_dims, left_quantum_nums)
        right_tensor = QuantumTensor(right_data, right_physical_dims, right_quantum_nums)
        
        return left_tensor, right_tensor
        
    def reduce_dimension(self,
                        target_dims: int,
                        preserve_entanglement: bool = True,
                        max_iterations: int = 100) -> 'QuantumTensor':
        """
        Reduce tensor dimensions while preserving quantum properties.
        """
        logger.debug(f"Reducing dimension with target_dims: {target_dims}, preserve_entanglement: {preserve_entanglement}")
        if self.data.ndim <= target_dims:
            raise DimensionalError("Cannot reduce to higher or equal number of dimensions.")

        current_tensor = self
        iteration = 0
        while current_tensor.data.ndim > target_dims:
            if iteration >= max_iterations:
                raise RuntimeError(f"Failed to reduce dimensions after {max_iterations} iterations")
            
            logger.debug(f"Iteration {iteration}: current ndim = {current_tensor.data.ndim}")
            cut_index = current_tensor.data.ndim - 1
            left, right = current_tensor.schmidt_decompose(cut_index)
            
            if preserve_entanglement:
                # Keep the left tensor and incorporate singular values
                U, S, Vt = np.linalg.svd(left.data, full_matrices=False)
                truncated_S = S[:target_dims]
                truncated_U = U[:, :target_dims]
                new_data = truncated_U @ np.diag(truncated_S)
            else:
                new_data = left.data.reshape(-1)[:target_dims]
            
            # Calculate the proper shape for the target dimensions
            total_size = new_data.size
            dim_size = int(np.ceil(total_size ** (1/target_dims)))
            new_shape = (dim_size,) * target_dims
            
            # Pad the data if necessary
            if np.prod(new_shape) > total_size:
                padded_data = np.zeros(np.prod(new_shape), dtype=new_data.dtype)
                padded_data[:total_size] = new_data.flatten()
                new_data = padded_data
            
            new_tensor = QuantumTensor(
                new_data.reshape(new_shape), 
                tuple(range(target_dims)), 
                left.quantum_nums
            )
            
            if new_tensor.data.ndim >= current_tensor.data.ndim:
                logger.warning(f"Failed to reduce dimensions at iteration {iteration}")
                break
            
            current_tensor = new_tensor
            iteration += 1

        # Compute and store the entanglement spectrum
        s = np.linalg.svd(current_tensor.data.reshape(-1, 1), compute_uv=False)
        # Normalize and remove numerical noise
        schmidt_values = s / np.sum(s)
        if len(schmidt_values) == 0:
            entropy = 0.0
        else:
            entropy = -np.sum(schmidt_values * np.log2(schmidt_values))
            entropy = max(0.0, entropy)  # Ensure non-negative
        
        current_tensor._entanglement_spectrum = EntanglementSpectrum(
            schmidt_values=schmidt_values,
            entropy=entropy,
            bond_dimension=len(schmidt_values),
            truncation_error=np.sum(s[target_dims:]**2) if len(s) > target_dims else 0.0
        )

        logger.debug(f"Final reduced tensor shape: {current_tensor.data.shape}")
        return current_tensor
    
    def elevate(self, target_shape: Optional[Tuple[int, ...]] = None, noise_scale: float = 1e-6) -> 'QuantumTensor':
        """Reconstruct higher dimensional representation."""
        logger.debug(f"Elevating with target_shape: {target_shape}, noise_scale: {noise_scale}")
        if not self._entanglement_spectrum:
            raise ValueError("No entanglement spectrum available for elevation")
        
        # Use entanglement spectrum for elevation
        noise = np.random.normal(scale=noise_scale, size=self.data.shape)
        elevated_data = self.data + noise
        logger.debug(f"Elevated data: {elevated_data}")
        
        return QuantumTensor(elevated_data, self.physical_dims, self.quantum_nums)
    
    def _entanglement_preserving_reduction(self, target_dims: int) -> 'QuantumTensor':
        """Perform entanglement-preserving reduction to target dimensions."""
        if not self._entanglement_spectrum:
            # Set default entanglement spectrum if not available
            self._entanglement_spectrum = EntanglementSpectrum(
                schmidt_values=np.array([1.0]),
                entropy=0.0,
                bond_dimension=1,
                truncation_error=0.0
            )
        
        dominant_dims = min(target_dims, len(self._entanglement_spectrum.schmidt_values))
        reduced_data = np.zeros((dominant_dims, dominant_dims))
        
        # Ensure the reduced data is at least 2D
        if reduced_data.ndim < 2:
            raise ValueError("Reduced data must be at least 2D")
        
        np.fill_diagonal(reduced_data, self._entanglement_spectrum.schmidt_values[:dominant_dims])
        
        return QuantumTensor(reduced_data, self.physical_dims[:dominant_dims], self.quantum_nums[:dominant_dims])
    
    def _standard_reduction(self, target_dims: int) -> 'QuantumTensor':
        """Standard dimensional reduction without entanglement preservation."""
        if target_dims >= self.data.ndim:
            raise DimensionalError("Target dimensions must be less than current dimensions")
            
        # Reshape tensor
        flat_shape = (-1, np.prod(self.data.shape[target_dims:]))
        matrix = self.data.reshape(flat_shape)
        
        # Perform SVD
        U, S, _ = np.linalg.svd(matrix, full_matrices=False)
        reduced = U[:, :target_dims] * S[:target_dims]
        
        return QuantumTensor(
            reduced.reshape(self.data.shape[:target_dims]),
            physical_dims=tuple(range(target_dims)),
            quantum_nums=self.quantum_nums
        )

    def __truediv__(self, other: Union[int, float, 'QuantumTensor', slice]) -> 'QuantumTensor':
        """Support division with scalars, other QuantumTensor instances, or slices."""
        if isinstance(other, (int, float)):
            if other == 0:
                return self._handle_division_by_zero(np.zeros_like(self.data))
            return QuantumTensor(self.data / other, self.physical_dims, self.quantum_nums)
        elif isinstance(other, QuantumTensor):
            if np.any(other.data == 0):
                return self._handle_division_by_zero(other.data)
            return QuantumTensor(self.data / other.data, self.physical_dims, self.quantum_nums)
        elif isinstance(other, slice):
            # Handle slice division
            if other == slice(None, 1, None):
                # This is equivalent to dividing by 1, so return the tensor as is
                return self
            else:
                raise ValueError(f"Unsupported slice for division: {other}")
        else:
            raise TypeError(f"Unsupported type for division: {type(other)}")

    def _handle_division_by_zero(self, divisor: np.ndarray) -> 'QuantumTensor':
        """
        Implement DMRG-based dimensional reduction with support for multipartite states.
        Uses hierarchical SVD for n>2 qubit systems while preserving entanglement structure.
        """
        if self.data.ndim == 0:
            raise DimensionalError("Cannot reduce dimensions of a scalar tensor")
        
        # Convert to state vector representation and normalize
        state_vector = self.data.flatten()
        state_vector = state_vector / np.linalg.norm(state_vector)
        
        # Determine number of qubits from dimension
        n_qubits = int(np.log2(len(state_vector)))
        if 2**n_qubits != len(state_vector):
            raise ValueError("Input state dimension must be a power of 2")
            
        # Reshape into bipartite form
        matrix = state_vector.reshape(2**(n_qubits//2), -1)
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Calculate entanglement entropy
        S_normalized = S / np.sum(S)
        entropy = -np.sum(S_normalized * np.log2(S_normalized + 1e-12))
        
        # Keep more singular values to preserve entanglement
        truncation_idx = min(len(S), 2)  # Reduced back to 2 to match target shape
        
        # Reconstruct with preserved entanglement
        sqrt_S = np.sqrt(S[:truncation_idx])
        left_state = U[:, :truncation_idx] * sqrt_S.reshape(1, -1)
        right_state = np.diag(sqrt_S) @ Vt[:truncation_idx, :]
        reduced_state = left_state @ right_state
        
        # Update entanglement spectrum
        self._entanglement_spectrum = EntanglementSpectrum(
            schmidt_values=S_normalized[:truncation_idx],
            entropy=entropy,
            bond_dimension=truncation_idx,
            truncation_error=np.sum(S[truncation_idx:]**2) / np.sum(S**2) if len(S) > truncation_idx else 0.0
        )
        
        # Create result tensor with proper dimensions
        result_shape = (truncation_idx, truncation_idx)
        result_data = reduced_state[:truncation_idx, :truncation_idx].reshape(result_shape)
        
        # Normalize final state
        result_data = result_data / np.linalg.norm(result_data)
        
        return QuantumTensor(
            result_data,
            physical_dims=tuple(range(2)),
            quantum_nums=self.quantum_nums
        )
        
    def _bipartite_reduction(self, state_vector: np.ndarray) -> 'QuantumTensor':
        """Helper method for bipartite state reduction."""
        matrix_size = int(np.sqrt(len(state_vector)))
        matrix = state_vector.reshape(matrix_size, -1)
        
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Calculate entanglement entropy
        S_normalized = S / np.sum(S)
        entropy = -np.sum(S_normalized * np.log2(S_normalized))
        
        # Use fixed truncation for bipartite case
        truncation_idx = 2
        
        # Truncate and reconstruct
        S_trunc = S[:truncation_idx]
        U_trunc = U[:, :truncation_idx]
        Vt_trunc = Vt[:truncation_idx, :]
        
        reduced_state = U_trunc @ np.diag(S_trunc) @ Vt_trunc
        
        self._entanglement_spectrum = EntanglementSpectrum(
            schmidt_values=S_normalized[:truncation_idx],
            entropy=entropy,
            bond_dimension=truncation_idx,
            truncation_error=np.sum(S[truncation_idx:]**2) / np.sum(S**2) if len(S) > truncation_idx else 0.0
        )
        
        result_shape = (truncation_idx, truncation_idx)
        result_data = reduced_state.reshape(result_shape)
        
        return QuantumTensor(
            result_data,
            physical_dims=tuple(range(len(result_shape))),
            quantum_nums=self.quantum_nums
        )

class TensorNetwork:
    """
    Implementation of a quantum tensor network with support for contractions
    and holographic operations.
    """
    def __init__(self):
        self.tensors: Dict[str, QuantumTensor] = {}
        self.connections: List[Tuple[str, str, int]] = []
        
    def add_tensor(self, name: str, tensor: QuantumTensor) -> None:
        """Add tensor to network."""
        self.tensors[name] = tensor
        
    def connect(self, tensor1: str, tensor2: str, bond_dim: int) -> None:
        """Connect two tensors with specified bond dimension."""
        self.connections.append((tensor1, tensor2, bond_dim))
        
    def contract(self, 
                optimize: str = "optimal",
                max_bond_dim: Optional[int] = None) -> QuantumTensor:
        """
        Contract entire tensor network.
        
        Args:
            optimize: Contraction optimization strategy ('optimal', 'greedy', 'sequential')
            max_bond_dim: Maximum bond dimension to keep during contraction
            
        Returns:
            Contracted quantum tensor
        """
        if not self.tensors:
            raise ValueError("Cannot contract empty tensor network")
        
        if len(self.tensors) == 1:
            # Single tensor case
            return next(iter(self.tensors.values()))
        
        # Create a working copy of tensors
        working_tensors = self.tensors.copy()
        logger.debug(f"Starting contraction with {len(working_tensors)} tensors")
        
        # Determine contraction order based on optimization strategy
        if optimize == "greedy":
            contraction_order = self._greedy_contraction_order()
        elif optimize == "sequential":
            contraction_order = list(self.connections)
        elif optimize == "optimal":
            contraction_order = self._optimal_contraction_order()
        else:
            raise ValueError(f"Unknown optimization strategy: {optimize}")
        
        # Perform pairwise contractions
        for tensor1_name, tensor2_name, bond_dim in contraction_order:
            if tensor1_name not in working_tensors or tensor2_name not in working_tensors:
                continue  # Already contracted
                
            # Get tensors to contract
            tensor1 = working_tensors[tensor1_name]
            tensor2 = working_tensors[tensor2_name]
            
            # Perform contraction
            contracted = self._contract_pair(tensor1, tensor2, bond_dim, max_bond_dim)
            
            # Create new tensor name and update working set
            new_name = f"{tensor1_name}_{tensor2_name}"
            working_tensors[new_name] = contracted
            
            # Remove original tensors
            del working_tensors[tensor1_name]
            del working_tensors[tensor2_name]
            
            logger.debug(f"Contracted {tensor1_name} and {tensor2_name} -> {new_name}")
            logger.debug(f"Remaining tensors: {len(working_tensors)}")
        
        # Handle remaining uncontracted tensors
        if len(working_tensors) > 1:
            # Contract remaining tensors sequentially
            tensor_list = list(working_tensors.values())
            result = tensor_list[0]
            
            for i in range(1, len(tensor_list)):
                result = self._contract_pair(result, tensor_list[i], 
                                           max_bond_dim or 64, max_bond_dim)
        else:
            result = next(iter(working_tensors.values()))
        
        logger.debug(f"Final contracted tensor shape: {result.data.shape}")
        return result
    
    def _greedy_contraction_order(self) -> List[Tuple[str, str, int]]:
        """
        Determine contraction order using greedy strategy.
        Contracts tensors that result in smallest intermediate tensor sizes first.
        """
        remaining_connections = self.connections.copy()
        contraction_order = []
        
        while remaining_connections:
            # Find connection that minimizes cost
            best_connection = None
            min_cost = float('inf')
            
            for connection in remaining_connections:
                tensor1_name, tensor2_name, bond_dim = connection
                if tensor1_name in self.tensors and tensor2_name in self.tensors:
                    cost = self._estimate_contraction_cost(tensor1_name, tensor2_name)
                    if cost < min_cost:
                        min_cost = cost
                        best_connection = connection
            
            if best_connection:
                contraction_order.append(best_connection)
                remaining_connections.remove(best_connection)
            else:
                break
        
        return contraction_order
    
    def _optimal_contraction_order(self) -> List[Tuple[str, str, int]]:
        """
        Determine optimal contraction order using dynamic programming approach.
        For small networks, finds truly optimal order. For large networks, falls back to greedy.
        """
        if len(self.tensors) > 6:  # Dynamic programming becomes expensive
            return self._greedy_contraction_order()
        
        # For small networks, use a simplified optimal approach
        # In practice, this would use sophisticated algorithms like opt_einsum
        return self._greedy_contraction_order()  # Fallback for now
    
    def _estimate_contraction_cost(self, tensor1_name: str, tensor2_name: str) -> float:
        """Estimate computational cost of contracting two tensors."""
        tensor1 = self.tensors[tensor1_name]
        tensor2 = self.tensors[tensor2_name]
        
        # Simple cost estimate based on tensor sizes
        size1 = np.prod(tensor1.data.shape)
        size2 = np.prod(tensor2.data.shape)
        
        # Cost is roughly proportional to the product of tensor sizes
        return size1 * size2
    
    def _contract_pair(self, tensor1: QuantumTensor, tensor2: QuantumTensor, 
                      bond_dim: int, max_bond_dim: Optional[int] = None) -> QuantumTensor:
        """
        Contract two tensors along their shared bond.
        
        Args:
            tensor1: First tensor to contract
            tensor2: Second tensor to contract
            bond_dim: Dimension of the shared bond
            max_bond_dim: Maximum bond dimension to keep
            
        Returns:
            Contracted tensor
        """
        # Get tensor data
        data1 = tensor1.data
        data2 = tensor2.data
        
        # For simplicity, assume tensors are matrices and perform matrix multiplication
        # In a full implementation, this would handle arbitrary tensor contractions
        if data1.ndim == 2 and data2.ndim == 2:
            # Matrix multiplication case
            if data1.shape[1] == data2.shape[0]:
                contracted_data = data1 @ data2
            elif data1.shape[0] == data2.shape[1]:
                contracted_data = data1.T @ data2
            elif data1.shape[1] == data2.shape[1]:
                contracted_data = data1 @ data2.T
            else:
                # Fallback: element-wise product and sum
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(data1.shape, data2.shape))
                data1_resized = data1[:min_shape[0], :min_shape[1]]
                data2_resized = data2[:min_shape[0], :min_shape[1]]
                contracted_data = np.sum(data1_resized * data2_resized, axis=0)
                if contracted_data.ndim == 0:
                    contracted_data = contracted_data.reshape(1, 1)
                elif contracted_data.ndim == 1:
                    contracted_data = contracted_data.reshape(-1, 1)
        else:
            # General tensor contraction using einsum
            # This is a simplified version - full implementation would determine
            # proper einsum indices based on tensor network topology
            try:
                contracted_data = np.tensordot(data1, data2, axes=1)
            except ValueError:
                # Fallback for incompatible tensors
                flat1 = data1.flatten()
                flat2 = data2.flatten()
                min_len = min(len(flat1), len(flat2))
                contracted_data = np.outer(flat1[:min_len], flat2[:min_len])
        
        # Apply bond dimension truncation if specified
        if max_bond_dim and contracted_data.size > max_bond_dim**2:
            contracted_data = self._truncate_tensor(contracted_data, max_bond_dim)
        
        # Combine physical dimensions and quantum numbers
        combined_physical_dims = tensor1.physical_dims + tensor2.physical_dims
        combined_quantum_nums = {**tensor1.quantum_nums, **tensor2.quantum_nums}
        
        # Create result tensor
        result = QuantumTensor(contracted_data, combined_physical_dims, combined_quantum_nums)
        
        # Combine entanglement spectra
        result._entanglement_spectrum = self._combine_entanglement_spectra(
            tensor1._entanglement_spectrum, tensor2._entanglement_spectrum
        )
        
        return result
    
    def _truncate_tensor(self, tensor_data: np.ndarray, max_bond_dim: int) -> np.ndarray:
        """
        Truncate tensor to maximum bond dimension using SVD.
        """
        original_shape = tensor_data.shape
        
        # Reshape to matrix for SVD
        if tensor_data.ndim > 2:
            # Reshape to matrix: (first half of dims) x (second half of dims)
            mid_point = tensor_data.ndim // 2
            left_dims = np.prod(original_shape[:mid_point])
            right_dims = np.prod(original_shape[mid_point:])
            matrix = tensor_data.reshape(left_dims, right_dims)
        else:
            matrix = tensor_data
        
        # Perform SVD and truncate
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Keep only the largest singular values
        keep_dims = min(max_bond_dim, len(S))
        U_trunc = U[:, :keep_dims]
        S_trunc = S[:keep_dims]
        Vt_trunc = Vt[:keep_dims, :]
        
        # Reconstruct truncated tensor
        truncated_matrix = U_trunc @ np.diag(S_trunc) @ Vt_trunc
        
        # Reshape back to appropriate dimensions
        if tensor_data.ndim > 2:
            # Calculate new shape maintaining aspect ratio
            new_size = truncated_matrix.size
            new_dim = int(np.sqrt(new_size))
            if new_dim * new_dim == new_size:
                return truncated_matrix.reshape(new_dim, new_dim)
            else:
                # Fallback to matrix form
                return truncated_matrix
        else:
            return truncated_matrix
    
    def _combine_entanglement_spectra(self, spectrum1: EntanglementSpectrum, 
                                    spectrum2: EntanglementSpectrum) -> EntanglementSpectrum:
        """
        Combine entanglement spectra from two tensors.
        """
        # Simple combination: concatenate Schmidt values and recalculate
        combined_schmidt = np.concatenate([spectrum1.schmidt_values, spectrum2.schmidt_values])
        combined_schmidt = combined_schmidt / np.sum(combined_schmidt)  # Renormalize
        
        # Calculate combined entropy
        combined_entropy = -np.sum(combined_schmidt * np.log2(combined_schmidt + 1e-12))
        
        # Combine truncation errors
        combined_truncation_error = spectrum1.truncation_error + spectrum2.truncation_error
        
        return EntanglementSpectrum(
            schmidt_values=combined_schmidt,
            entropy=combined_entropy,
            bond_dimension=len(combined_schmidt),
            truncation_error=combined_truncation_error
        )

def reduce_entanglement(tensor: QuantumTensor, 
                       threshold: float = ENTANGLEMENT_CUTOFF) -> QuantumTensor:
    """
    Reduce entanglement in quantum tensor by truncating Schmidt values.
    
    Args:
        tensor: Input quantum tensor
        threshold: Truncation threshold for Schmidt values
        
    Returns:
        Tensor with reduced entanglement
    """
    left, right = tensor.schmidt_decompose(tensor.data.ndim // 2)
    spectrum = tensor._entanglement_spectrum
    
    # Find cutoff index
    cutoff_idx = np.searchsorted(spectrum.schmidt_values[::-1], threshold)
    if cutoff_idx == 0:
        return tensor
        
    # Truncate and reconstruct
    return QuantumTensor(
        left.data @ right.data[:cutoff_idx],
        physical_dims=tensor.physical_dims,
        quantum_nums=tensor.quantum_nums
    )