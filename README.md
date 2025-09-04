# DivideByZero: Dimensional Reduction Through Mathematical Singularities

## Foundational Framework for Computational Singularity Analysis

DivideByZero (`dividebyzero`) implements a novel mathematical framework that reconceptualizes division by zero as dimensional reduction operations. This paradigm shift transforms traditionally undefined mathematical operations into well-defined dimensional transformations, enabling new approaches to numerical analysis and quantum computation.

## Core Mathematical Principles

### Dimensional Division Operator
The framework defines division by zero through the dimensional reduction operator $\oslash$:

For tensor $T \in \mathcal{D}_n$:
```
T ∅ 0 = π(T) + ε(T)
```
Where:
- $\pi(T)$: Projection to lower dimension
- $\epsilon(T)$: Quantized error preservation
- $\mathcal{D}_n$: n-dimensional tensor space

### Installation

```bash
pip install dividebyzero
```

## Fundamental Usage Patterns

### Basic Operations
```python
import dividebyzero as dbz

# Create dimensional array
x = dbz.array([[1, 2, 3],
               [4, 5, 6]])

# Divide by zero - reduces dimension
result = x / 0

# Reconstruct original dimensions
reconstructed = result.elevate()
```

### Key Features

#### 1. Transparent NumPy Integration
- Use ``import dividebyzero as dbz`` and call NumPy-like functions directly
  (``dbz.array``, ``dbz.zeros``)
- For rare drop-in scenarios, ``dbz.numpy_compat`` mirrors the NumPy API
- Preserves standard numerical behavior
- Extends functionality to handle singularities

#### 2. Information Preservation
- Maintains core data characteristics through reduction
- Tracks quantum error information
- Enables dimensional reconstruction

#### 3. Advanced Mathematical Operations
```python
import dividebyzero as dbz

# Quantum tensor operations
q_tensor = dbz.quantum.QuantumTensor(data, physical_dims=(2, 2, 2))

# Perform gauge-invariant reduction
reduced = q_tensor.reduce_dimension(
    target_dims=2,
    preserve_entanglement=True
)
```

## Theoretical Framework

### Mathematical Foundations

The framework builds on several key mathematical concepts:

1. **Dimensional Reduction**
   - Singular Value Decomposition (SVD)
   - Information-preserving projections
   - Error quantization mechanisms

2. **Quantum Extensions**
   - Tensor network operations
   - Gauge field computations
   - Holonomy calculations

3. **Error Tracking**
   - Holographic error encoding
   - Dimensional reconstruction algorithms
   - Quantum state preservation

## Advanced Applications

### 1. Quantum Computing
```python
# Quantum state manipulation
state = dbz.quantum.QuantumTensor([
    [1, 0],
    [0, 1]
])

# Preserve quantum information during reduction
reduced_state = state / 0
```

### 2. Numerical Analysis
```python
# Handle singularities in numerical computations
def stable_computation(x):
    return dbz.array(x) / 0  # Returns dimensional reduction instead of error
```

### 3. Data Processing
```python
# Dimensionality reduction with information preservation
reduced_data = dbz.array(high_dim_data) / 0
reconstructed = reduced_data.elevate()
```

## Technical Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0

### Optional Dependencies
- networkx ≥ 2.6.0 (for quantum features)
- pytest ≥ 6.0 (for testing)

## Development and Extension

### Contributing
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

### Testing
```bash
pytest tests/
```

## Demonstration Suite

Standalone examples are available in the [`examples/`](examples/) directory to
showcase typical usage patterns and advanced features:

```bash
python examples/basic_reduction.py      # Dimensional reduction and registry
python examples/numpy_integration.py    # NumPy-compatible operations
python examples/quantum_tensor.py       # Quantum tensor utilities
python examples/financial_returns.py    # Market returns with zero-price days
python examples/web_pagerank.py         # PageRank transition matrix without NaNs
python examples/image_normalization.py  # Image normalization with dark pixels
```

Each script prints intermediate results illustrating how the library safely
handles division by zero while preserving recoverable information.

## Mathematical Documentation

Detailed mathematical foundations are available in the [Technical Documentation](docs/theory.md), including:

- Formal proofs of dimensional preservation
- Quantum mechanical extensions
- Gauge field implementations
- Error quantization theorems

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{dividebyzero2024,
  title={DivideByZero: Dimensional Reduction Through Mathematical Singularities},
  author={Michael C. Jenkins},
  year={2024},
  url={https://github.com/jenkinsm13/dividebyzero}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Note:** This framework reimagines fundamental mathematical operations. While it provides practical solutions for handling mathematical singularities, users should understand the underlying theoretical principles for appropriate application.