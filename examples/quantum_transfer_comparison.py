import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from dividebyzero.quantum import QuantumTensor
from dividebyzero.quantum.holonomy import HolonomyCalculator
from dividebyzero.quantum.gauge_groups import U1Group

def state_transfer_qutip(distance, noise_strength=0.01):
    """Implement quantum state transfer using QuTiP with adiabatic evolution."""
    # Initial state |1⟩ in first qubit
    psi0 = qt.tensor(qt.basis([2], 1), qt.basis([2], 0))
    
    # Create time points - use more points for adiabatic evolution
    times = np.linspace(0, 20.0, 400)  # Longer evolution time for adiabatic transfer
    
    # Create Pauli operators for both qubits
    sx1 = qt.tensor(qt.sigmax(), qt.identity(2))
    sy1 = qt.tensor(qt.sigmay(), qt.identity(2))
    sx2 = qt.tensor(qt.identity(2), qt.sigmax())
    sy2 = qt.tensor(qt.identity(2), qt.sigmay())
    
    # Create coupling operators with adiabatic turn-on/off
    Jx = 0.5 * (sx1 * sx2 + sy1 * sy2)
    Jy = 0.5 * (sx1 * sy2 - sy1 * sx2)
    
    # Base coupling strength scaled by distance with minimum strength
    base_coupling = 1.0 / (distance + 0.5)  # Avoid divergence at small distances
    
    # Add noise to the coupling strengths
    noise_x = noise_strength * np.random.randn()
    noise_y = noise_strength * np.random.randn()
    
    def H1_coeff(t, args):
        # Smooth turn-on/off for adiabatic evolution
        envelope = np.sin(np.pi * t / times[-1])**2
        return base_coupling * (1.0 + noise_x) * envelope * np.cos(np.pi * t / times[-1])
        
    def H2_coeff(t, args):
        # Smooth turn-on/off for adiabatic evolution
        envelope = np.sin(np.pi * t / times[-1])**2
        return base_coupling * (1.0 + noise_y) * envelope * np.sin(np.pi * t / times[-1])
    
    H = [[Jx, H1_coeff], [Jy, H2_coeff]]
    
    # Evolve state
    result = qt.sesolve(H, psi0, times)
    final_state = result.states[-1]
    
    # Target state: |0⟩|1⟩
    target = qt.tensor(qt.basis([2], 0), qt.basis([2], 1))
    
    # Calculate fidelity
    fidelity = abs(target.overlap(final_state))**2
    return fidelity

def state_transfer_holonomy(distance, noise_strength=0.01):
    """Implement quantum state transfer using dividebyzero's holonomy."""
    # Initial state |1⟩|0⟩
    psi0 = QuantumTensor(np.array([0, 1, 0, 0]))
    
    # Initialize holonomy calculator
    gauge_group = U1Group()
    holonomy_calc = HolonomyCalculator(gauge_group)
    
    # Create path in parameter space
    num_points = 20
    theta = np.linspace(0, np.pi/2 * distance, num_points)
    
    # Create noisy path
    loop = []
    for th in theta:
        noise_x = noise_strength * np.random.randn()
        noise_y = noise_strength * np.random.randn()
        x = np.cos(th) + noise_x
        y = np.sin(th) + noise_y
        loop.append((x, y))
    
    # Define the Hamiltonian function
    def H_transfer(params):
        x, y = params
        return np.array([[0, 0, 0, 0],
                        [0, x, y, 0],
                        [0, y, -x, 0],
                        [0, 0, 0, 0]])
    
    # Calculate and apply evolution
    phase = holonomy_calc.berry_phase(H_transfer, loop)
    U = np.array([[1, 0, 0, 0],
                  [0, np.cos(phase), -np.sin(phase), 0],
                  [0, np.sin(phase), np.cos(phase), 0],
                  [0, 0, 0, 1]])
    final_state = QuantumTensor(U @ psi0.data)
    
    # Target state |0⟩|1⟩
    target = QuantumTensor(np.array([0, 0, 1, 0]))
    
    # Calculate fidelity
    fidelity = np.abs(np.vdot(target.data.flatten(), final_state.data.flatten()))**2
    return fidelity

def main():
    # Test state transfer with different noise levels and distances
    distances = np.linspace(0.5, 3.0, 50)
    noise_levels = [0.01, 0.05, 0.1]
    
    plt.figure(figsize=(12, 8))
    
    for noise in noise_levels:
        qutip_fidelities = np.zeros_like(distances)
        holonomy_fidelities = np.zeros_like(distances)
        
        for i, dist in enumerate(distances):
            # Run multiple trials
            n_trials = 20
            qutip_trial_fids = []
            holonomy_trial_fids = []
            
            for _ in range(n_trials):
                qutip_trial_fids.append(state_transfer_qutip(dist, noise))
                holonomy_trial_fids.append(state_transfer_holonomy(dist, noise))
            
            qutip_fidelities[i] = np.mean(qutip_trial_fids)
            holonomy_fidelities[i] = np.mean(holonomy_trial_fids)
        
        plt.plot(distances, qutip_fidelities, 
                label=f'QuTiP (noise={noise:.2f})',
                linestyle='--')
        plt.plot(distances, holonomy_fidelities,
                label=f'Holonomy (U(1)) (noise={noise:.2f})',
                linestyle='-')
    
    plt.xlabel('Transfer Distance (arbitrary units)')
    plt.ylabel('Transfer Fidelity')
    plt.title('Quantum State Transfer Fidelity vs Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig('transfer_comparison-u1.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main() 