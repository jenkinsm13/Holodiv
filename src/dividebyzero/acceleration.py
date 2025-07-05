"""
GPU Acceleration Module for DivideByZero

This module provides GPU-accelerated implementations of dimensional reduction
operations using CuPy as a drop-in NumPy replacement.

Example of implementing new opportunities for the DivideByZero project.
"""

import warnings
from typing import Optional, Union
import numpy as np

# Optional CuPy import for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration disabled.")

from .array import DimensionalArray
from .exceptions import DimensionalError


class GPUDimensionalArray(DimensionalArray):
    """
    GPU-accelerated version of DimensionalArray using CuPy.
    
    This demonstrates how to extend the existing framework with new capabilities
    while maintaining compatibility with the existing API.
    """
    
    def __init__(self, array_like, error_registry=None, dtype=None, device='auto'):
        """
        Initialize a GPU-accelerated DimensionalArray.
        
        Parameters:
            array_like: Input array or array-like object
            error_registry: Optional error registry for tracking operations
            dtype: Data type for the array
            device: 'gpu', 'cpu', or 'auto' for automatic selection
        """
        if not GPU_AVAILABLE and device == 'gpu':
            raise RuntimeError("GPU requested but CuPy not available")
        
        self.device = self._select_device(device)
        
        if self.device == 'gpu':
            if isinstance(array_like, DimensionalArray):
                self.array = cp.array(array_like.array)
            else:
                self.array = cp.array(array_like, dtype=dtype)
        else:
            # Fall back to CPU implementation
            super().__init__(array_like, error_registry, dtype)
            return
            
        from . import get_registry
        self.error_registry = error_registry or get_registry()
        self._error_id = None
        self._right_singular_vector = None
    
    def _select_device(self, device: str) -> str:
        """Intelligently select computation device."""
        if device == 'auto':
            return 'gpu' if GPU_AVAILABLE else 'cpu'
        elif device == 'gpu' and not GPU_AVAILABLE:
            warnings.warn("GPU requested but not available, falling back to CPU")
            return 'cpu'
        return device
    
    def _gpu_divide_by_zero(self) -> 'GPUDimensionalArray':
        """
        GPU-accelerated division by zero using CuPy.
        
        This demonstrates how GPU acceleration can significantly speed up
        mathematical operations, especially SVD computations.
        """
        if self.device == 'cpu':
            return super()._divide_by_zero()
            
        original_shape = self.array.shape
        ndim = self.array.ndim
        
        if ndim == 0:  # scalar case
            result = cp.array([cp.abs(self.array)])
            error = cp.array([self.array - result[0]])
            original_shape = (1,)
        elif ndim == 1:
            # GPU-accelerated mean calculation
            mean_val = cp.abs(self.array).mean()
            result = cp.array([mean_val])
            error = self.array - mean_val
        else:
            # GPU-accelerated SVD - this is where major speedups occur
            reshaped = self.array.reshape(original_shape[0], -1)
            try:
                # CuPy's SVD runs on GPU and can be much faster
                U, S, Vt = cp.linalg.svd(reshaped, full_matrices=False)
                threshold = 0.1 * S.max()
                result = U[:, 0] * (S[0] if S[0] > threshold else threshold)
                self._right_singular_vector = cp.asnumpy(Vt[0, :])  # Convert back for compatibility
                
                # GPU-accelerated reconstruction
                reconstructed = cp.outer(result, Vt[0, :]).reshape(original_shape)
                error = self.array - reconstructed
            except cp.linalg.LinAlgError:
                raise DimensionalError("GPU SVD failed during division by zero")
        
        # Store error information (convert to NumPy for compatibility)
        from .registry import ErrorData
        error_data = ErrorData(
            original_shape=original_shape,
            error_tensor=cp.asnumpy(error) if self.device == 'gpu' else error,
            reduction_type='complete'
        )
        error_id = self.error_registry.store(error_data)
        
        reduced = GPUDimensionalArray(result, self.error_registry, device=self.device)
        reduced._error_id = error_id
        reduced._right_singular_vector = self._right_singular_vector
        return reduced
    
    def __truediv__(self, other):
        """GPU-accelerated division operation."""
        if self.device == 'cpu':
            return super().__truediv__(other)
            
        print(f"GPU dividing {self.array} by {other}")
        
        if isinstance(other, (int, float)):
            if other == 0:
                return self._gpu_divide_by_zero()
            return GPUDimensionalArray(self.array / other, self.error_registry, device=self.device)
        
        if isinstance(other, (DimensionalArray, GPUDimensionalArray)):
            other_array = other.array if hasattr(other, 'array') else other
            if hasattr(other_array, 'get'):  # CuPy array
                other_array = cp.asarray(other_array)
            
            if cp.ndim(other_array) == 0:
                if float(other_array) == 0:
                    return self._gpu_divide_by_zero()
                return GPUDimensionalArray(self.array / other_array, self.error_registry, device=self.device)
            
            if cp.any(other_array == 0):
                # Handle partial division by zero on GPU
                mask = other_array == 0
                result = self._gpu_partial_divide_by_zero(mask)
                non_zero_mask = ~mask
                result.array[non_zero_mask] = self.array[non_zero_mask] / other_array[non_zero_mask]
                return result
                
            return GPUDimensionalArray(self.array / other_array, self.error_registry, device=self.device)
        
        raise TypeError(f"Unsupported type for GPU division: {type(other)}")
    
    def _gpu_partial_divide_by_zero(self, mask):
        """GPU-accelerated partial division by zero."""
        result = cp.zeros_like(self.array, dtype=float)
        non_zero_mask = ~mask
        
        if self.array.ndim == 1:
            non_zero_elements = self.array[non_zero_mask]
            if non_zero_elements.size > 0:
                zero_division_value = non_zero_elements.mean()
            else:
                zero_division_value = 0
            result[mask] = zero_division_value
        else:
            # GPU-optimized broadcasting operations
            broadcast_shape = cp.broadcast_shapes(self.array.shape, mask.shape)
            expanded_mask = cp.broadcast_to(mask, broadcast_shape)
            
            # Vectorized operations on GPU
            slice_means = cp.nanmean(cp.where(~expanded_mask, self.array, cp.nan), axis=-1)
            result = cp.where(expanded_mask, slice_means[..., cp.newaxis], self.array)
        
        from .registry import ErrorData
        error_data = ErrorData(
            original_shape=self.array.shape,
            error_tensor=cp.asnumpy(self.array - result),
            reduction_type='partial'
        )
        error_id = self.error_registry.store(error_data)
        
        reduced = GPUDimensionalArray(result, self.error_registry, device=self.device)
        reduced._error_id = error_id
        return reduced
    
    def to_cpu(self) -> DimensionalArray:
        """Convert GPU array back to CPU."""
        if self.device == 'cpu':
            return self
        
        cpu_array = cp.asnumpy(self.array)
        return DimensionalArray(cpu_array, self.error_registry)
    
    def to_gpu(self) -> 'GPUDimensionalArray':
        """Ensure array is on GPU."""
        if self.device == 'gpu':
            return self
        
        return GPUDimensionalArray(self.array, self.error_registry, device='gpu')


class PerformanceBenchmark:
    """
    Benchmarking tools for comparing CPU vs GPU performance.
    
    This demonstrates another new opportunity: performance monitoring and analysis.
    """
    
    def __init__(self):
        self.results = {}
    
    def benchmark_division_by_zero(self, sizes, devices=['cpu', 'gpu'], iterations=5):
        """
        Benchmark division by zero operations across different sizes and devices.
        
        Parameters:
            sizes: List of array sizes to test (e.g., [100, 1000, 10000])
            devices: List of devices to test ['cpu', 'gpu']
            iterations: Number of iterations for averaging
            
        Returns:
            Dictionary with timing results
        """
        import time
        
        results = {}
        
        for size in sizes:
            results[size] = {}
            # Create test data
            test_data = np.random.randn(size, size)
            
            for device in devices:
                if device == 'gpu' and not GPU_AVAILABLE:
                    continue
                    
                times = []
                for _ in range(iterations):
                    if device == 'gpu':
                        arr = GPUDimensionalArray(test_data, device='gpu')
                    else:
                        arr = DimensionalArray(test_data)
                    
                    start_time = time.perf_counter()
                    result = arr / 0  # Division by zero
                    if device == 'gpu':
                        cp.cuda.Device().synchronize()  # Ensure GPU operation completes
                    end_time = time.perf_counter()
                    
                    times.append(end_time - start_time)
                
                results[size][device] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'speedup': None  # Will be calculated later
                }
        
        # Calculate speedups
        for size in sizes:
            if 'cpu' in results[size] and 'gpu' in results[size]:
                cpu_time = results[size]['cpu']['mean_time']
                gpu_time = results[size]['gpu']['mean_time']
                results[size]['gpu']['speedup'] = cpu_time / gpu_time
        
        self.results = results
        return results
    
    def generate_report(self):
        """Generate a performance report."""
        if not self.results:
            return "No benchmark results available. Run benchmark_division_by_zero() first."
        
        report = ["Performance Benchmark Report", "=" * 30, ""]
        
        for size, device_results in self.results.items():
            report.append(f"Array size: {size}x{size}")
            for device, metrics in device_results.items():
                report.append(f"  {device.upper()}:")
                report.append(f"    Mean time: {metrics['mean_time']:.4f}s")
                report.append(f"    Std time: {metrics['std_time']:.4f}s")
                if metrics['speedup']:
                    report.append(f"    Speedup vs CPU: {metrics['speedup']:.2f}x")
            report.append("")
        
        return "\n".join(report)


def create_accelerated_array(array_like, device='auto', **kwargs):
    """
    Factory function for creating accelerated arrays.
    
    This provides a clean API for users to access GPU acceleration
    without needing to know the implementation details.
    """
    if device == 'auto':
        device = 'gpu' if GPU_AVAILABLE else 'cpu'
    
    if device == 'gpu':
        return GPUDimensionalArray(array_like, device='gpu', **kwargs)
    else:
        return DimensionalArray(array_like, **kwargs)


# Convenience functions
def benchmark_performance(sizes=[100, 500, 1000], devices=['cpu', 'gpu']):
    """Quick performance benchmark."""
    benchmark = PerformanceBenchmark()
    results = benchmark.benchmark_division_by_zero(sizes, devices)
    print(benchmark.generate_report())
    return results


def gpu_available():
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE


if __name__ == "__main__":
    # Example usage and demonstration
    print("DivideByZero GPU Acceleration Demo")
    print("=" * 40)
    
    if gpu_available():
        print("✓ GPU acceleration available")
        
        # Create test data
        test_array = np.random.randn(100, 100)
        
        # CPU version
        print("\nCPU Division by Zero:")
        cpu_arr = DimensionalArray(test_array)
        cpu_result = cpu_arr / 0
        print(f"Result shape: {cpu_result.shape}")
        
        # GPU version
        print("\nGPU Division by Zero:")
        gpu_arr = GPUDimensionalArray(test_array, device='gpu')
        gpu_result = gpu_arr / 0
        print(f"Result shape: {gpu_result.shape}")
        
        # Quick benchmark
        print("\nQuick Performance Benchmark:")
        benchmark_performance(sizes=[100, 500], devices=['cpu', 'gpu'])
        
    else:
        print("✗ GPU acceleration not available (CuPy not installed)")
        print("Install CuPy with: pip install cupy-cuda11x  # or appropriate CUDA version")