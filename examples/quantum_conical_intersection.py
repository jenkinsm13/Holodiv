import matplotlib.pyplot as plt
import dividebyzero as dbz
from dividebyzero.quantum import QuantumTensor
from dividebyzero.quantum.holonomy import HolonomyCalculator
from dividebyzero.quantum.gauge_groups import U1Group
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def create_conical_hamiltonian(t, x, y, regularization=1e-6):
    """
    Create a Hamiltonian with a conical intersection at (x,y) = (0,0).
    This type of Hamiltonian appears in molecular systems and exhibits
    interesting geometric phase effects.
    
    Added regularization to handle numerical instabilities.
    """
    # Distance from conical intersection with regularization
    r = dbz.sqrt(x**2 + y**2 + regularization**2)
    theta = dbz.arctan2(y, x) if r > regularization else 0
    
    # Create the Hamiltonian matrix with regularization
    H = dbz.array([
        [t * x + regularization, t * y - 1j*r],
        [t * y + 1j*r, -t * x - regularization]
    ])
    return dbz.array(H)

def traditional_evolution(t_values, x, y):
    """
    Traditional quantum evolution that fails at the conical intersection.
    """
    states = []
    psi0 = dbz.array([1, 0]) / dbz.sqrt(2)  # Initial state
    current_state = psi0
    
    try:
        for t in t_values:
            H = create_conical_hamiltonian(t, x, y).data
            # This will fail near the conical intersection
            eigenvals, eigenvecs = dbz.linalg.eigh(H)
            # Evolution operator
            U = eigenvecs @ dbz.diag(dbz.exp(-1j * eigenvals)) @ eigenvecs.conj().T
            current_state = U @ current_state
            states.append(current_state)
    except dbz.linalg.LinAlgError:
        # Fill with NaN when traditional method fails
        states.extend([dbz.array([dbz.nan, dbz.nan])] * (len(t_values) - len(states)))
    
    return dbz.array(states)

def quantum_holonomy_evolution(t_values, x, y):
    """
    Evolution using dividebyzero's holonomy-based approach.
    This handles the conical intersection gracefully through dimensional reduction.
    """
    # Initialize holonomy calculator with U(1) gauge group
    calculator = HolonomyCalculator(U1Group())
    
    states = []
    # Initial state as quantum tensor with proper physical dimensions
    psi0 = QuantumTensor(dbz.array([1, 0]) / dbz.sqrt(2), physical_dims=(2,))
    current_state = psi0
    
    for t in t_values:
        # Create Hamiltonian as quantum tensor
        H = create_conical_hamiltonian(t, x, y)
        # Normalize Hamiltonian
        H_norm = dbz.linalg.norm(dbz.array(H).flatten())
        H = H / H_norm
        H_tensor = QuantumTensor(H, physical_dims=(2,))
        
        # Compute geometric phase
        delta = max(0.01, abs(t) * 0.1)
        loop = [(t-delta, x, y), (t, x, y), (t+delta, x, y)]
        phase = calculator.berry_phase(
            lambda params: dbz.array(create_conical_hamiltonian(*params).data) / dbz.linalg.norm(dbz.array(create_conical_hamiltonian(*params).data).flatten()),
            loop
        )
        
        # Create effective Hamiltonian including geometric phase
        H_effective = H + phase * dbz.eye(2)
        # Normalize effective Hamiltonian
        H_effective_norm = dbz.linalg.norm(dbz.array(H_effective).flatten())
        H_effective = H_effective / H_effective_norm
        H_effective_tensor = QuantumTensor(H_effective, physical_dims=(2,))
        
        # Apply dimensional reduction at/near the conical intersection
        r = dbz.sqrt(x**2 + y**2)
        if r < 0.5:  # Near intersection region
            # Use framework's division by zero for dimensional reduction
            reduced_state = current_state / H_effective_tensor
            current_state = reduced_state
        else:
            # Far from intersection, use standard evolution with geometric phase
            U = dbz.array([
                [dbz.exp(-1j * phase), 0],
                [0, dbz.exp(1j * phase)]
            ])
            # Apply geometric phase correction
            U = U @ dbz.array([
                [dbz.cos(phase * dbz.pi), -dbz.sin(phase * dbz.pi)],
                [dbz.sin(phase * dbz.pi), dbz.cos(phase * dbz.pi)]
            ])
            # Normalize unitary
            U = U / dbz.linalg.norm(U.flatten())
            evolved_data = U @ current_state.data
            # Normalize evolved state
            evolved_data = evolved_data / dbz.linalg.norm(evolved_data)
            current_state = QuantumTensor(evolved_data, physical_dims=(2,))
        
        # Track the quantum state with normalization
        state_data = dbz.array(current_state.data).flatten()
        state_data = state_data / dbz.linalg.norm(state_data)
        states.append(state_data)
    
    return dbz.array(states)

def plot_results(t_values, trad_states, holo_states, x, y):
    """Plot the evolution of quantum states."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Quantum Evolution at (x={x:.1f}, y={y:.1f})')
    
    # Plot traditional evolution
    for i in range(2):
        # Ensure proper normalization for probabilities
        prob = dbz.clip(dbz.abs(trad_states[:, i])**2, 0, 1)
        axes[i, 0].plot(t_values, prob, 'r-', label=f'|{i}⟩')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel(f'P(|{i}⟩)')
        axes[i, 0].set_title(f'Traditional Evolution - State |{i}⟩')
        axes[i, 0].grid(True)
        axes[i, 0].legend()
    
    # Plot holonomy evolution
    for i in range(2):
        # Ensure proper normalization for probabilities
        prob = dbz.clip(dbz.abs(holo_states[:, i])**2, 0, 1)
        axes[i, 1].plot(t_values, prob, 'b-', label=f'|{i}⟩')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel(f'P(|{i}⟩)')
        axes[i, 1].set_title(f'Holonomy Evolution - State |{i}⟩')
        axes[i, 1].grid(True)
        axes[i, 1].legend()
    
    plt.tight_layout()
    filename = f'quantum_conical_intersection_x_{x:.1f}_y_{y:.1f}.png'
    plt.savefig(filename)
    plt.close()

def main():
    # Create time values
    t_values = dbz.linspace(-2, 2, 200)
    
    # Points to evaluate (including conical intersection)
    points = [(0.0, 0.0), (0.1, 0.0), (1.0, 0.0)]
    
    for x, y in points:
        # Traditional evolution
        trad_states = traditional_evolution(t_values, x, y)
        # Normalize states
        trad_states = trad_states / dbz.sqrt(dbz.sum(dbz.abs(trad_states)**2, axis=1))[:, dbz.newaxis]
        
        # Holonomy evolution
        holo_states = quantum_holonomy_evolution(t_values, x, y)
        # Normalize states
        holo_states = holo_states / dbz.sqrt(dbz.sum(dbz.abs(holo_states)**2, axis=1))[:, dbz.newaxis]
        
        # Plot results
        plot_results(t_values, trad_states, holo_states, x, y)

if __name__ == '__main__':
    main() 