"""
Visualization Module for DivideByZero

This module provides comprehensive visualization capabilities for dimensional reduction,
quantum operations, and tensor networks. Supports both 2D and 3D plotting with
interactive features.
"""

import warnings
from typing import Optional, Tuple, List, Union, Dict, Any
import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    animation = None
    Axes3D = None
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

from .array import DimensionalArray
from .quantum.tensor import QuantumTensor, EntanglementSpectrum


class DimensionalVisualizer:
    """
    Main visualization class for dimensional reduction operations.
    """
    
    def __init__(self, backend='matplotlib'):
        """
        Initialize visualizer with specified backend.
        
        Parameters:
            backend: 'matplotlib' or 'plotly'
        """
        self.backend = backend
        
        if backend == 'matplotlib' and not MATPLOTLIB_AVAILABLE:
            warnings.warn("Matplotlib not available, falling back to plotly")
            self.backend = 'plotly'
        elif backend == 'plotly' and not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available, falling back to matplotlib")
            self.backend = 'matplotlib'
        
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            raise ImportError("No visualization backend available. Install matplotlib or plotly.")
    
    def plot_reduction_pathway(self, original: DimensionalArray, 
                             reduced: DimensionalArray, 
                             reconstructed: Optional[DimensionalArray] = None,
                             title: str = "Dimensional Reduction Pathway") -> Any:
        """
        Visualize the pathway from original → reduced → reconstructed data.
        
        Parameters:
            original: Original high-dimensional data
            reduced: Dimensionally reduced data
            reconstructed: Reconstructed data (optional)
            title: Plot title
            
        Returns:
            Figure object
        """
        if self.backend == 'matplotlib':
            return self._plot_reduction_pathway_mpl(original, reduced, reconstructed, title)
        else:
            return self._plot_reduction_pathway_plotly(original, reduced, reconstructed, title)
    
    def _plot_reduction_pathway_mpl(self, original, reduced, reconstructed, title):
        """Matplotlib implementation of reduction pathway."""
        fig, axes = plt.subplots(1, 3 if reconstructed else 2, figsize=(15, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Original data
        if original.array.ndim >= 2:
            im1 = axes[0].imshow(original.array, cmap='viridis', aspect='auto')
            axes[0].set_title('Original Data')
            axes[0].set_xlabel(f'Shape: {original.shape}')
            plt.colorbar(im1, ax=axes[0])
        else:
            axes[0].plot(original.array)
            axes[0].set_title('Original Data')
            axes[0].set_xlabel('Index')
            axes[0].set_ylabel('Value')
        
        # Reduced data
        if reduced.array.ndim >= 2:
            im2 = axes[1].imshow(reduced.array, cmap='viridis', aspect='auto')
            axes[1].set_title('Reduced Data')
            axes[1].set_xlabel(f'Shape: {reduced.shape}')
            plt.colorbar(im2, ax=axes[1])
        else:
            axes[1].plot(reduced.array)
            axes[1].set_title('Reduced Data')
            axes[1].set_xlabel('Index')
            axes[1].set_ylabel('Value')
        
        # Reconstructed data (if provided)
        if reconstructed and len(axes) > 2:
            if reconstructed.array.ndim >= 2:
                im3 = axes[2].imshow(reconstructed.array, cmap='viridis', aspect='auto')
                axes[2].set_title('Reconstructed Data')
                axes[2].set_xlabel(f'Shape: {reconstructed.shape}')
                plt.colorbar(im3, ax=axes[2])
            else:
                axes[2].plot(reconstructed.array)
                axes[2].set_title('Reconstructed Data')
                axes[2].set_xlabel('Index')
                axes[2].set_ylabel('Value')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def _plot_reduction_pathway_plotly(self, original, reduced, reconstructed, title):
        """Plotly implementation of reduction pathway."""
        cols = 3 if reconstructed else 2
        fig = make_subplots(rows=1, cols=cols, 
                           subplot_titles=['Original Data', 'Reduced Data', 'Reconstructed Data'][:cols])
        
        # Original data
        if original.array.ndim >= 2:
            fig.add_trace(go.Heatmap(z=original.array, colorscale='Viridis', name='Original'), 
                         row=1, col=1)
        else:
            fig.add_trace(go.Scatter(y=original.array, mode='lines', name='Original'), 
                         row=1, col=1)
        
        # Reduced data
        if reduced.array.ndim >= 2:
            fig.add_trace(go.Heatmap(z=reduced.array, colorscale='Viridis', name='Reduced'), 
                         row=1, col=2)
        else:
            fig.add_trace(go.Scatter(y=reduced.array, mode='lines', name='Reduced'), 
                         row=1, col=2)
        
        # Reconstructed data
        if reconstructed:
            if reconstructed.array.ndim >= 2:
                fig.add_trace(go.Heatmap(z=reconstructed.array, colorscale='Viridis', name='Reconstructed'), 
                             row=1, col=3)
            else:
                fig.add_trace(go.Scatter(y=reconstructed.array, mode='lines', name='Reconstructed'), 
                             row=1, col=3)
        
        fig.update_layout(title_text=title, showlegend=False)
        return fig
    
    def plot_error_landscape(self, error_tensor: np.ndarray, 
                           title: str = "Error Landscape") -> Any:
        """
        Create a 3D surface plot of reconstruction errors.
        
        Parameters:
            error_tensor: 2D array of reconstruction errors
            title: Plot title
            
        Returns:
            Figure object
        """
        if error_tensor.ndim != 2:
            raise ValueError("Error tensor must be 2D for landscape visualization")
        
        if self.backend == 'matplotlib':
            return self._plot_error_landscape_mpl(error_tensor, title)
        else:
            return self._plot_error_landscape_plotly(error_tensor, title)
    
    def _plot_error_landscape_mpl(self, error_tensor, title):
        """Matplotlib 3D error landscape."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        x = np.arange(error_tensor.shape[1])
        y = np.arange(error_tensor.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, error_tensor, cmap='plasma', alpha=0.8)
        
        # Add contour lines
        contours = ax.contour(X, Y, error_tensor, zdir='z', offset=error_tensor.min(), cmap='plasma')
        
        ax.set_xlabel('X Dimension')
        ax.set_ylabel('Y Dimension')
        ax.set_zlabel('Error Magnitude')
        ax.set_title(title)
        
        plt.colorbar(surf, ax=ax, shrink=0.5)
        return fig
    
    def _plot_error_landscape_plotly(self, error_tensor, title):
        """Plotly 3D error landscape."""
        fig = go.Figure(data=[go.Surface(z=error_tensor, colorscale='Plasma')])
        
        fig.update_layout(
            title=title,
            autosize=False,
            width=800,
            height=600,
            scene=dict(
                xaxis_title='X Dimension',
                yaxis_title='Y Dimension',
                zaxis_title='Error Magnitude'
            )
        )
        return fig
    
    def animate_dimensional_flow(self, tensor_sequence: List[DimensionalArray],
                               title: str = "Dimensional Flow Animation") -> Any:
        """
        Create animated visualization of dimensional reduction over time.
        
        Parameters:
            tensor_sequence: List of tensors showing evolution
            title: Animation title
            
        Returns:
            Animation object
        """
        if self.backend == 'matplotlib':
            return self._animate_dimensional_flow_mpl(tensor_sequence, title)
        else:
            # Plotly animation would be more complex to implement
            warnings.warn("Plotly animation not implemented, using static plot")
            return self.plot_reduction_pathway(tensor_sequence[0], tensor_sequence[-1])
    
    def _animate_dimensional_flow_mpl(self, tensor_sequence, title):
        """Matplotlib animation of dimensional flow."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine common scale
        all_data = [t.array for t in tensor_sequence]
        vmin = min(np.min(data) for data in all_data)
        vmax = max(np.max(data) for data in all_data)
        
        def update(frame):
            ax.clear()
            data = tensor_sequence[frame].array
            
            if data.ndim >= 2:
                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                ax.set_title(f'{title} - Frame {frame+1}/{len(tensor_sequence)}')
                ax.set_xlabel(f'Shape: {data.shape}')
            else:
                ax.plot(data)
                ax.set_title(f'{title} - Frame {frame+1}/{len(tensor_sequence)}')
                ax.set_ylim(vmin, vmax)
            
            return ax.get_children()
        
        ani = animation.FuncAnimation(fig, update, frames=len(tensor_sequence), 
                                    interval=500, blit=False, repeat=True)
        return ani


class QuantumVisualizer:
    """
    Specialized visualizer for quantum tensor operations.
    """
    
    def __init__(self, backend='matplotlib'):
        self.visualizer = DimensionalVisualizer(backend)
        self.backend = backend
    
    def plot_entanglement_spectrum(self, spectrum: EntanglementSpectrum,
                                 title: str = "Entanglement Spectrum") -> Any:
        """
        Plot the entanglement spectrum (Schmidt values).
        
        Parameters:
            spectrum: EntanglementSpectrum object
            title: Plot title
            
        Returns:
            Figure object
        """
        if self.backend == 'matplotlib':
            return self._plot_entanglement_spectrum_mpl(spectrum, title)
        else:
            return self._plot_entanglement_spectrum_plotly(spectrum, title)
    
    def _plot_entanglement_spectrum_mpl(self, spectrum, title):
        """Matplotlib entanglement spectrum plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Schmidt values
        ax1.semilogy(spectrum.schmidt_values, 'o-')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Schmidt Value')
        ax1.set_title('Schmidt Values')
        ax1.grid(True)
        
        # Entanglement entropy info
        ax2.bar(['Entropy', 'Bond Dim', 'Trunc Error'], 
                [spectrum.entropy, spectrum.bond_dimension, spectrum.truncation_error])
        ax2.set_title('Entanglement Properties')
        ax2.set_ylabel('Value')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def _plot_entanglement_spectrum_plotly(self, spectrum, title):
        """Plotly entanglement spectrum plot."""
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=['Schmidt Values', 'Entanglement Properties'])
        
        # Schmidt values
        fig.add_trace(go.Scatter(y=spectrum.schmidt_values, mode='lines+markers',
                                name='Schmidt Values'), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        
        # Properties
        fig.add_trace(go.Bar(x=['Entropy', 'Bond Dim', 'Trunc Error'],
                            y=[spectrum.entropy, spectrum.bond_dimension, spectrum.truncation_error],
                            name='Properties'), row=1, col=2)
        
        fig.update_layout(title_text=title, showlegend=False)
        return fig
    
    def plot_quantum_state(self, quantum_tensor: QuantumTensor,
                          representation: str = 'amplitude',
                          title: str = "Quantum State") -> Any:
        """
        Visualize quantum state in various representations.
        
        Parameters:
            quantum_tensor: QuantumTensor to visualize
            representation: 'amplitude', 'phase', 'probability'
            title: Plot title
            
        Returns:
            Figure object
        """
        data = quantum_tensor.data
        
        if representation == 'amplitude':
            plot_data = np.abs(data)
        elif representation == 'phase':
            plot_data = np.angle(data)
        elif representation == 'probability':
            plot_data = np.abs(data)**2
        else:
            raise ValueError(f"Unknown representation: {representation}")
        
        return self.visualizer.plot_reduction_pathway(
            DimensionalArray(plot_data), 
            DimensionalArray(plot_data),
            title=f"{title} ({representation})"
        )


class PerformanceVisualizer:
    """
    Visualizer for performance metrics and benchmarks.
    """
    
    def __init__(self, backend='matplotlib'):
        self.backend = backend
    
    def plot_speedup_comparison(self, benchmark_results: Dict[str, Dict],
                              title: str = "Performance Speedup") -> Any:
        """
        Visualize performance speedup results.
        
        Parameters:
            benchmark_results: Results from performance benchmarking
            title: Plot title
            
        Returns:
            Figure object
        """
        sizes = list(benchmark_results.keys())
        speedups = []
        
        for size in sizes:
            if 'gpu' in benchmark_results[size] and benchmark_results[size]['gpu']['speedup']:
                speedups.append(benchmark_results[size]['gpu']['speedup'])
            else:
                speedups.append(1.0)  # No speedup
        
        if self.backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(sizes)), speedups)
            ax.set_xlabel('Array Size')
            ax.set_ylabel('Speedup (x)')
            ax.set_title(title)
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels([f"{s}x{s}" for s in sizes])
            ax.grid(True, alpha=0.3)
            
            # Add speedup labels on bars
            for i, speedup in enumerate(speedups):
                ax.text(i, speedup + 0.1, f'{speedup:.1f}x', 
                       ha='center', va='bottom')
            
            return fig
        
        elif PLOTLY_AVAILABLE:
            fig = go.Figure([go.Bar(x=[f"{s}x{s}" for s in sizes], y=speedups)])
            fig.update_layout(title=title, xaxis_title="Array Size", 
                            yaxis_title="Speedup (x)")
            return fig
        
        else:
            raise ImportError("No visualization backend available")


# Convenience functions
def plot_dimensional_reduction(original: DimensionalArray, reduced: DimensionalArray,
                             reconstructed: Optional[DimensionalArray] = None,
                             backend: str = 'matplotlib') -> Any:
    """Quick plot of dimensional reduction results."""
    viz = DimensionalVisualizer(backend)
    return viz.plot_reduction_pathway(original, reduced, reconstructed)


def plot_quantum_entanglement(quantum_tensor: QuantumTensor,
                            backend: str = 'matplotlib') -> Any:
    """Quick plot of quantum entanglement spectrum."""
    viz = QuantumVisualizer(backend)
    return viz.plot_entanglement_spectrum(quantum_tensor._entanglement_spectrum)


def animate_reduction_process(tensor_sequence: List[DimensionalArray],
                            backend: str = 'matplotlib') -> Any:
    """Quick animation of reduction process."""
    viz = DimensionalVisualizer(backend)
    return viz.animate_dimensional_flow(tensor_sequence)


# Check what visualization backends are available
def available_backends() -> List[str]:
    """Return list of available visualization backends."""
    backends = []
    if MATPLOTLIB_AVAILABLE:
        backends.append('matplotlib')
    if PLOTLY_AVAILABLE:
        backends.append('plotly')
    return backends


def recommend_backend() -> str:
    """Recommend the best available backend."""
    if PLOTLY_AVAILABLE:
        return 'plotly'  # More interactive
    elif MATPLOTLIB_AVAILABLE:
        return 'matplotlib'  # More stable
    else:
        raise ImportError("No visualization backends available. Install matplotlib or plotly.")


if __name__ == "__main__":
    # Demo visualization capabilities
    print("DivideByZero Visualization Demo")
    print("=" * 40)
    
    print(f"Available backends: {available_backends()}")
    print(f"Recommended backend: {recommend_backend()}")
    
    # Create test data
    original_data = np.random.randn(50, 50)
    original = DimensionalArray(original_data)
    reduced = original / 0  # Division by zero reduction
    
    # Create visualizations
    try:
        backend = recommend_backend()
        
        # Basic reduction visualization
        fig1 = plot_dimensional_reduction(original, reduced, backend=backend)
        print("✓ Created dimensional reduction plot")
        
        # Quantum visualization (if quantum tensor available)
        try:
            from .quantum.tensor import QuantumTensor
            quantum_state = np.random.randn(16) + 1j * np.random.randn(16)
            quantum_state = quantum_state / np.linalg.norm(quantum_state)
            qtensor = QuantumTensor(quantum_state)
            
            fig2 = plot_quantum_entanglement(qtensor, backend=backend)
            print("✓ Created quantum entanglement plot")
        except Exception as e:
            print(f"⚠️  Quantum visualization failed: {e}")
        
        print("Visualization demo completed successfully!")
        
    except Exception as e:
        print(f"Visualization demo failed: {e}")
        print("Install matplotlib or plotly to enable visualizations")