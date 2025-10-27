import numpy as np
import pytest
from holodiv.quantum.tensor import QuantumTensor, TensorNetwork, reduce_entanglement
from holodiv.exceptions import DimensionalError

def test_quantum_tensor_init():
    """Test QuantumTensor initialization."""
    data = np.random.rand(2, 2)
    tensor = QuantumTensor(data)
    assert np.allclose(tensor.data, data)
    assert tensor.physical_dims == (0, 1)

    # Test with quantum numbers
    qnums = {'charge': [1, -1]}
    tensor_q = QuantumTensor(data, quantum_nums=qnums)
    assert tensor_q.quantum_nums == qnums

def test_schmidt_decomposition():
    """Test the schmidt_decompose method."""
    data = np.random.rand(4, 4)
    tensor = QuantumTensor(data)
    left, right = tensor.schmidt_decompose(1)
    assert left.data.shape[0] == 4
    assert right.data.shape[1] == 4

def test_reduce_dimension():
    """Test the reduce_dimension method."""
    data = np.random.rand(4, 4, 4)
    tensor = QuantumTensor(data)

    # Test reduction to 2 dimensions
    reduced_tensor = tensor.reduce_dimension(2)
    assert reduced_tensor.data.ndim == 2

    # Test without preserving entanglement
    reduced_tensor_no_ent = tensor.reduce_dimension(2, preserve_entanglement=False)
    assert reduced_tensor_no_ent.data.ndim == 2

    # Test error on higher or equal dimensions
    with pytest.raises(DimensionalError):
        tensor.reduce_dimension(3)

    # Test max iterations error
    with pytest.raises(RuntimeError):
        tensor.reduce_dimension(2, max_iterations=0)

def test_elevate():
    """Test the elevate method."""
    data = np.random.rand(2, 2)
    tensor = QuantumTensor(data)
    tensor.schmidt_decompose(1) # to populate entanglement spectrum
    elevated_tensor = tensor.elevate()
    assert elevated_tensor.data.shape == (2, 2)

def test_division():
    """Test the __truediv__ method."""
    data = np.random.rand(4, 4)
    tensor = QuantumTensor(data)

    # Division by scalar
    result = tensor / 2.0
    assert np.allclose(result.data, data / 2.0)

    # Division by zero
    result_zero = tensor / 0
    assert result_zero.data.shape == (2, 2)

    # Division by another tensor
    divisor_data = np.random.rand(4, 4) + 0.1 # avoid division by zero
    divisor = QuantumTensor(divisor_data)
    result_tensor = tensor / divisor
    assert np.allclose(result_tensor.data, data / divisor_data)

    # Division by slice
    result_slice = tensor / slice(None, 1, None)
    assert np.allclose(result_slice.data, data)

    # Unsupported slice
    with pytest.raises(ValueError):
        tensor / slice(None, 2, None)

    # Unsupported type
    with pytest.raises(TypeError):
        tensor / "unsupported"

def test_tensor_network():
    """Test the TensorNetwork class."""
    tn = TensorNetwork()
    tensor1 = QuantumTensor(np.random.rand(2, 2))
    tensor2 = QuantumTensor(np.random.rand(2, 2))

    tn.add_tensor("A", tensor1)
    tn.add_tensor("B", tensor2)
    assert "A" in tn.tensors

    tn.connect("A", "B", 2)
    assert len(tn.connections) == 1

    # Test contraction (currently raises NotImplementedError)
    with pytest.raises(NotImplementedError):
        tn.contract()

def test_reduce_entanglement():
    """Test the reduce_entanglement function."""
    data = np.random.rand(4, 4)
    tensor = QuantumTensor(data)
    reduced_tensor = reduce_entanglement(tensor)
    assert reduced_tensor.data.shape[1] <= 4

    # Test with zero threshold
    reduced_tensor_zero = reduce_entanglement(tensor, threshold=0.0)
    assert np.allclose(reduced_tensor_zero.data, tensor.data)

def test_internal_reductions():
    """Test the internal reduction methods."""
    data = np.random.rand(4, 4)
    tensor = QuantumTensor(data)
    tensor.schmidt_decompose(1)

    # Test entanglement preserving reduction
    reduced_ent = tensor._entanglement_preserving_reduction(2)
    assert reduced_ent.data.ndim == 2

    # Test standard reduction
    reduced_std = tensor._standard_reduction(1)
    assert reduced_std.data.ndim == 1

    # Test bipartite reduction
    state_vector = np.random.rand(16)
    reduced_bipartite = tensor._bipartite_reduction(state_vector)
    assert reduced_bipartite.data.shape == (2, 2)
from holodiv.exceptions import DimensionalError
def test_tensor_full_coverage():
    """Tests for full coverage of tensor.py."""
    import numpy as np
    from holodiv.quantum.tensor import QuantumTensor
    import pytest

    # Test reduce_dimension with empty schmidt values
    qt = QuantumTensor(np.array([1, 0, 0, 0]))
    qt._entanglement_spectrum.schmidt_values = np.array([])
    with pytest.raises(DimensionalError):
        qt.reduce_dimension(1)

    # Test _handle_division_by_zero with non-power of 2
    qt_invalid = QuantumTensor(np.random.rand(6))
    with pytest.raises(ValueError):
        qt_invalid / 0
