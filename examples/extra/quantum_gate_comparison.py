import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from dividebyzero.quantum import QuantumTensor
from dividebyzero.quantum.holonomy import HolonomyCalculator
from dividebyzero.quantum.gauge_groups import U1Group

def implement_phase_gate_qutip(phase, noise_strength=0.01):
    """Implement a phase gate using QuTiP with realistic noise."""
    # Initial state |+⟩ = (|0⟩ + |1⟩)/√2
    psi0 = qt.Qobj(np.array([1/np.sqrt(2), 1/np.sqrt(2)]))
    
    # Ideal phase gate
    phase_gate = qt.Qobj(np.array([[1, 0], [0, np.exp(1j*phase)]], dtype=complex))
    
    # Add realistic noise
    noise = noise_strength * (np.random.randn(2,2) + 1j * np.random.randn(2,2))
    noisy_gate = qt.Qobj(phase_gate.full() + noise)
    
    # Normalize the noisy gate
    noisy_gate = noisy_gate.unit()
    
    # Apply gate
    final_state = noisy_gate * psi0
    
    # Calculate fidelity with ideal output
    ideal_output = phase_gate * psi0
    fidelity = abs(ideal_output.overlap(final_state))**2
    
    return fidelity

def implement_phase_gate_holonomy(phase, noise_strength=0.01):
    """Implement a phase gate using dividebyzero's holonomy approach."""
    # Initial state |+⟩ = (|0⟩ + |1⟩)/√2
    psi0 = QuantumTensor(np.array([1/np.sqrt(2), 1/np.sqrt(2)]))
    
    # Initialize holonomy calculator
    gauge_group = U1Group()
    holonomy_calc = HolonomyCalculator(gauge_group)
    
    # Create a loop in parameter space that will generate desired phase
    num_points = 16
    radius = abs(phase)/(2*np.pi)  # Radius determines the phase accumulated
    theta = np.linspace(0, 2*np.pi, num_points)
    
    # Create a circular loop with noise
    loop = []
    for th in theta:
        # Add noise to both coordinates
        noise_x = noise_strength * np.random.randn()
        noise_y = noise_strength * np.random.randn()
        x = radius * np.cos(th) + noise_x
        y = radius * np.sin(th) + noise_y
        loop.append((x, y))
    
    # Define the Hamiltonian function for the loop
    def H_loop(params):
        x, y = params
        return np.array([[x, y], [y, -x]])
    
    # Calculate and apply geometric phase
    acquired_phase = holonomy_calc.berry_phase(H_loop, loop)
    U = np.array([[1, 0], [0, np.exp(1j*acquired_phase)]])
    
    # Apply gate
    final_state = QuantumTensor(U @ psi0.data)
    
    # Calculate fidelity with ideal output
    ideal_U = np.array([[1, 0], [0, np.exp(1j*phase)]])
    ideal_output = QuantumTensor(ideal_U @ psi0.data)
    fidelity = np.abs(np.vdot(ideal_output.data.flatten(), final_state.data.flatten()))**2
    
    return fidelity

def main():
    # Test phase gates with different noise levels
    phases = np.linspace(0, 2*np.pi, 100)  # Use 100 points for both arrays
    noise_levels = [0.01, 0.05, 0.1]
    
    plt.figure(figsize=(12, 8))
    
    for noise in noise_levels:
        qutip_fidelities = np.zeros_like(phases)  # Pre-allocate arrays
        holonomy_fidelities = np.zeros_like(phases)
        
        for i, phase in enumerate(phases):
            # Run multiple trials for statistics
            n_trials = 20
            qutip_trial_fids = []
            holonomy_trial_fids = []
            
            for _ in range(n_trials):
                qutip_trial_fids.append(implement_phase_gate_qutip(phase, noise))
                holonomy_trial_fids.append(implement_phase_gate_holonomy(phase, noise))
            
            qutip_fidelities[i] = np.mean(qutip_trial_fids)
            holonomy_fidelities[i] = np.mean(holonomy_trial_fids)
        
        plt.plot(phases, qutip_fidelities, 
                label=f'QuTiP (noise={noise:.2f})', 
                linestyle='--')
        plt.plot(phases, holonomy_fidelities, 
                label=f'Holonomy (noise={noise:.2f})', 
                linestyle='-')
    
    plt.xlabel('Target Phase (radians)')
    plt.ylabel('Gate Fidelity')
    plt.title('Phase Gate Fidelity Under Different Noise Levels')
    plt.legend()
    plt.grid(True)
    plt.savefig('phase_gate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main() 