import numpy as np
import matplotlib.pyplot as plt
import dividebyzero as dbz
import seaborn as sns
from scipy.linalg import logm as scipy_logm
from scipy.linalg import expm
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def create_test_matrix(kind='nilpotent', size=4):
    """Create various test matrices that are challenging for matrix logarithm."""
    if kind == 'nilpotent':
        # Nilpotent matrix (all eigenvalues zero)
        A = np.zeros((size, size))
        for i in range(size-1):
            A[i, i+1] = 1
        return dbz.array(A)
    
    elif kind == 'jordan':
        # Jordan block with zero eigenvalue
        A = np.eye(size)
        for i in range(size-1):
            A[i, i+1] = 2
        A[0, 0] = 0
        return dbz.array(A)
    
    elif kind == 'degenerate':
        # Matrix with degenerate eigenvalues
        A = np.eye(size)
        A[0, 1] = A[1, 0] = 1
        return dbz.array(A)
    
    elif kind == 'singular_continuous':
        # Family of increasingly singular matrices
        def matrix(t):
            A = np.array([[np.cos(t), -np.sin(t)],
                         [np.sin(t), np.cos(t)]]) * np.exp(-t)
            return dbz.array(A)
        return matrix
    
    else:
        raise ValueError(f"Unknown matrix kind: {kind}")

def compare_logarithms(A, title):
    """Compare our logm with standard scipy implementation."""
    try:
        dbz_log = dbz.logm(A)
        scipy_log = dbz.array(scipy_logm(A.array))
        
        diff = dbz.linalg.norm(dbz_log - scipy_log)
        print(f"\n{title}")
        print("[DBZ] Matrix logarithm eigenvalues:", dbz.linalg.eigvals(dbz_log))
        print("[SciPy] Matrix logarithm eigenvalues:", dbz.linalg.eigvals(scipy_log))
        print("[Comparison] Norm of difference between DBZ and SciPy:", diff)
        
        # Verify log properties
        exp_dbz_log = dbz.array(expm(dbz_log.array))
        reconstruction_error = dbz.linalg.norm(exp_dbz_log - A)
        print("[DBZ] Reconstruction error:", reconstruction_error)
        
        return dbz_log, scipy_log
        
    except Exception as e:
        print(f"\n{title}")
        print("[Error] Exception occurred:", str(e))
        return None, None

# Test cases
print("=== Testing Various Singular Cases ===")

# Test 1: Nilpotent matrix
A_nil = create_test_matrix('nilpotent')
dbz_nil, scipy_nil = compare_logarithms(A_nil, "Nilpotent Matrix")

# Test 2: Jordan block
A_jordan = create_test_matrix('jordan')
dbz_jordan, scipy_jordan = compare_logarithms(A_jordan, "Jordan Block")

# Test 3: Degenerate eigenvalues
A_degen = create_test_matrix('degenerate')
dbz_degen, scipy_degen = compare_logarithms(A_degen, "Degenerate Matrix")

# Test 4: Continuous family of singular matrices
print("\n=== Testing Continuous Singular Family ===")
matrix_family = create_test_matrix('singular_continuous')
t_values = np.linspace(0, 5, 50)
dbz_norms = []
scipy_norms = []

for t in t_values:
    A_t = matrix_family(t)
    try:
        dbz_log = dbz.logm(A_t)
        scipy_log = dbz.array(scipy_logm(A_t.array))
        dbz_norms.append(float(dbz.linalg.norm(dbz_log)))  # Convert to float for plotting
        scipy_norms.append(float(dbz.linalg.norm(scipy_log)))
    except Exception as e:
        print(f"[Error] At t={t}: {str(e)}")
        dbz_norms.append(np.nan)
        scipy_norms.append(np.nan)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t_values, dbz_norms, label='DBZ logm', linewidth=2)
plt.plot(t_values, scipy_norms, '--', label='Scipy logm', linewidth=2)
plt.xlabel('Parameter t')
plt.ylabel('Norm of matrix logarithm')
plt.title('Behavior of Matrix Logarithm for Singular Continuous Family')
plt.legend()
plt.grid(True)
plt.show()

# Visualize differences in matrix structure
def plot_matrix_comparison(dbz_result, scipy_result, title):
    if dbz_result is None or scipy_result is None:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(dbz_result.array.real, ax=ax1, cmap='viridis')
    ax1.set_title('DBZ logm')
    
    sns.heatmap(scipy_result.array.real, ax=ax2, cmap='viridis')
    ax2.set_title('Scipy logm')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_matrix_comparison(dbz_nil, scipy_nil, "Nilpotent Matrix Logarithm")
plot_matrix_comparison(dbz_jordan, scipy_jordan, "Jordan Block Logarithm")
plot_matrix_comparison(dbz_degen, scipy_degen, "Degenerate Matrix Logarithm") 