# DivideByZero Demonstration Suite

This directory contains standalone examples showing how to use the `dividebyzero` library and illustrating its unique dimensional–reduction features.

## Available Examples

- **`basic_reduction.py`** – Introduction to dimensional reduction and reconstruction using the global error registry.
- **`numpy_integration.py`** – Using DivideByZero as a drop‑in replacement for NumPy functions and performing safe division by zero.
- **`quantum_tensor.py`** – Demonstrates the quantum tensor utilities and entanglement‑preserving dimensional reduction.
- **`financial_returns.py`** – Computes stock returns when prior prices hit zero, preserving market information instead of producing `inf`.
- **`web_pagerank.py`** – Builds a transition matrix for a small web graph with dangling nodes, enabling PageRank-style algorithms without NaNs.
- **`image_normalization.py`** – Normalizes an image by an illumination map containing dark pixels while retaining reconstruction data.

Run any script directly with Python:

```bash
python examples/basic_reduction.py
python examples/numpy_integration.py
python examples/quantum_tensor.py
python examples/financial_returns.py
python examples/web_pagerank.py
python examples/image_normalization.py
```

Each script prints intermediate values to illustrate how DivideByZero handles singular operations while preserving information.
