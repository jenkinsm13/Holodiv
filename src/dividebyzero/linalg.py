import numpy as np
from .array import DimensionalArray

def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """
    Singular Value Decomposition.
    
    This function wraps numpy's svd function to work with DimensionalArray objects.
    """
    if isinstance(a, DimensionalArray):
        a = a.array
    
    result = np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    
    if compute_uv:
        u, s, vh = result
        return DimensionalArray(u), DimensionalArray(s), DimensionalArray(vh)
    else:
        return DimensionalArray(result)

def eigh(a, UPLO='L'):
    """
    Eigenvalue decomposition for Hermitian (symmetric) arrays.
    
    This function wraps numpy's eigh function to work with DimensionalArray objects.
    """
    if isinstance(a, DimensionalArray):
        a = a.array
    
    w, v = np.linalg.eigh(a, UPLO=UPLO)
    return DimensionalArray(w), DimensionalArray(v)

def eigvalsh(a, UPLO='L'):
    """
    Compute eigenvalues of a Hermitian (symmetric) array.
    """
    if isinstance(a, DimensionalArray):
        a = a.array
    
    w = np.linalg.eigvalsh(a, UPLO=UPLO)
    return DimensionalArray(w)

def norm(x, ord=None, axis=None, keepdims=False):
    """
    Matrix or vector norm.
    """
    if isinstance(x, DimensionalArray):
        x = x.array
    
    result = np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return DimensionalArray(result) if isinstance(result, np.ndarray) else result

def det(a):
    """
    Compute the determinant of an array.
    """
    if isinstance(a, DimensionalArray):
        a = a.array
    
    return DimensionalArray(np.linalg.det(a))

def inv(a):
    """
    Compute the inverse of a matrix.
    """
    if isinstance(a, DimensionalArray):
        a = a.array
    
    return DimensionalArray(np.linalg.inv(a))

def eigvals(a):
    """
    Compute eigenvalues of a general matrix.
    
    Parameters
    ----------
    a : array_like
        A complex or real matrix (2-D array) whose eigenvalues will be computed.
    
    Returns
    -------
    w : DimensionalArray
        The computed eigenvalues.
    """
    if isinstance(a, DimensionalArray):
        a = a.array
    
    w = np.linalg.eigvals(a)
    return DimensionalArray(w)

def logm(a):
    """
    Compute matrix logarithm.
    
    Parameters
    ----------
    a : array_like
        Matrix whose logarithm is to be computed
    
    Returns
    -------
    logm : DimensionalArray
        Matrix logarithm of a
    """
    if isinstance(a, DimensionalArray):
        a = a.array
    
    return DimensionalArray(np.linalg.logm(a))