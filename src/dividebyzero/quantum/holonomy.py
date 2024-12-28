"""
Advanced Holonomy and Topological Calculations

This module implements sophisticated holonomy calculations including:
- Non-abelian Wilson loops
- Topological invariants
- Berry phase computations
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Union
from scipy.linalg import expm
from .gauge_groups import GaugeGroup, SU2Group, SU3Group, U1Group

class HolonomyCalculator:
    """
    Implements advanced holonomy calculations for gauge fields.
    """
    def __init__(self, gauge_group: GaugeGroup):
        self.gauge_group = gauge_group
        self._path_ordering_samples = 100
    
    def wilson_loop(self, 
                   connection: np.ndarray, 
                   loop: List[Tuple[float, ...]], 
                   method: str = 'adaptive') -> np.ndarray:
        """
        Compute Wilson loop with sophisticated path ordering.
        
        Args:
            connection: Gauge connection
            loop: Points defining closed loop
            method: Integration method ('adaptive' or 'fixed')
            
        Returns:
            Wilson loop matrix
        """
        if method == 'adaptive':
            return self._adaptive_path_ordering(connection, loop)
        return self._fixed_path_ordering(connection, loop)
    
    def _adaptive_path_ordering(self,
                              connection: np.ndarray,
                              loop: List[Tuple[float, ...]]) -> np.ndarray:
        """Implement adaptive path ordering based on curvature."""
        # Initialize transport matrix
        transport = np.eye(self.gauge_group.dimension, dtype=complex)
        
        # Compute local curvature to determine sampling
        for i in range(len(loop)-1):
            dx = np.array(loop[i+1]) - np.array(loop[i])
            local_curvature = self._compute_local_curvature(connection, loop[i])
            
            # Adjust sampling based on curvature
            n_samples = max(
                2,
                int(self._path_ordering_samples * local_curvature)
            )
            
            # Compute local transport
            local_transport = self._compute_local_transport(
                connection, loop[i], dx, n_samples
            )
            transport = local_transport @ transport
            
        return transport
    
    def _compute_local_curvature(self,
                                connection: np.ndarray,
                                point: Tuple[float, ...]) -> float:
        """Compute local field strength as measure of curvature."""
        F_μν = np.zeros((len(point), len(point)))
        for μ in range(len(point)):
            for ν in range(μ):
                # F_μν = ∂_μA_ν - ∂_νA_μ + [A_μ, A_ν]
                F_μν[μ,ν] = np.linalg.norm(
                    connection[μ] @ connection[ν] -
                    connection[ν] @ connection[μ]
                )
        return np.max(np.abs(F_μν))
        
    def berry_phase(self, 
                hamiltonian: Callable[[Tuple[float, ...]], np.ndarray],
                loop: List[Union[float, Tuple[float, ...]]]) -> float:
        """
        Calculate Berry phase for a given loop in parameter space.
        
        Args:
            hamiltonian: Function that returns the Hamiltonian matrix for given parameters
            loop: List of parameter values defining the loop. Can be single floats or tuples.
        
        Returns:
            Berry phase in units of π
        """
        phase = 0.0
        
        # Convert single parameters to tuples if needed
        loop = [(x,) if isinstance(x, (int, float)) else x for x in loop]
        
        # Compute eigenvectors at each point
        for i in range(len(loop)):
            t1, t2 = loop[i], loop[(i+1) % len(loop)]
            
            # Get Hamiltonians at adjacent points
            H1 = hamiltonian(t1)
            H2 = hamiltonian(t2)
            
            # Compute eigenvectors (get ground state)
            _, v1 = np.linalg.eigh(H1)
            _, v2 = np.linalg.eigh(H2)
            
            # Compute overlap between adjacent states
            overlap = np.vdot(v1[:, 0], v2[:, 0])
            
            # Compute the phase difference using the complex argument
            local_phase = np.angle(overlap) / np.pi
            
            # Accumulate the phase
            phase -= local_phase
        
        # Normalize to [-1, 1] range
        phase = ((phase + 1) % 2) - 1
        return phase
    
    def compute_chern_number(self,
                           berry_curvature: Callable[[float, float], float],
                           surface: List[Tuple[float, float]]) -> int:
        """
        Compute first Chern number (topological invariant).
        
        Args:
            berry_curvature: Function computing Berry curvature
            surface: Surface points for integration
            
        Returns:
            Chern number (integer)
        """
        total_curvature = 0.0
        
        # Discretize surface into triangles
        triangles = self._triangulate_surface(surface)
        
        # Integrate Berry curvature
        for triangle in triangles:
            area = self._triangle_area(triangle)
            center = np.mean(triangle, axis=0)
            total_curvature += berry_curvature(*center) * area
            
        return int(round(total_curvature / (2 * np.pi)))
    
    def _triangulate_surface(self,
                           points: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Implement Delaunay triangulation of surface."""
        from scipy.spatial import Delaunay
        points_array = np.array(points)
        triangulation = Delaunay(points_array)
        return [points_array[simplex] for simplex in triangulation.simplices]
    
    def _triangle_area(self, vertices: np.ndarray) -> float:
        """Compute area of triangle using cross product."""
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        return abs(np.cross(v1, v2)) / 2
    
    def parallel_propagator(self,
                          connection: np.ndarray,
                          path: List[Tuple[float, ...]],
                          reference_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute parallel propagator along path with reference frame.
        
        Args:
            connection: Gauge connection
            path: Path points
            reference_frame: Initial reference frame
            
        Returns:
            Parallel propagator matrix
        """
        if reference_frame is None:
            reference_frame = np.eye(self.gauge_group.dimension)
            
        propagator = reference_frame.copy()
        
        for i in range(len(path)-1):
            dx = np.array(path[i+1]) - np.array(path[i])
            connection_term = sum(
                dx[μ] * connection[μ] for μ in range(len(dx))
            )
            transport = expm(-1j * connection_term)
            propagator = transport @ propagator
            
        return propagator