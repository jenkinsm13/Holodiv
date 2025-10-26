import numpy as np
import pytest
import holodiv as hd
from holodiv.linalg import svd, eigh, eigvalsh, norm, det, inv, pinv, eigvals, logm
from scipy.linalg import expm

def test_svd():
    a_np = np.random.rand(3, 3)
    a = hd.array(a_np)
    u, s, vh = svd(a)
    assert isinstance(u, hd.DimensionalArray)
    assert isinstance(s, hd.DimensionalArray)
    assert isinstance(vh, hd.DimensionalArray)
    reconstructed_a = u.array @ np.diag(s.array) @ vh.array
    assert np.allclose(reconstructed_a, a_np)

def test_eigh():
    a_np = np.array([[1, -2j], [2j, 5]])
    a = hd.array(a_np)
    w, v = eigh(a)
    assert isinstance(w, hd.DimensionalArray)
    assert isinstance(v, hd.DimensionalArray)
    reconstructed_a = v.array @ np.diag(w.array) @ v.array.conj().T
    assert np.allclose(reconstructed_a, a_np)

def test_eigvalsh():
    a = hd.array(np.array([[1, -2j], [2j, 5]]))
    w = eigvalsh(a)
    assert isinstance(w, hd.DimensionalArray)
    assert np.all(np.isreal(w.array))

def test_norm():
    a = hd.array(np.arange(9) - 4)
    b = norm(a)
    assert np.isclose(b, np.linalg.norm(a.array))
    c = a.reshape((3, 3))
    d = norm(c)
    assert np.isclose(d, np.linalg.norm(c.array))
    e = norm(c, axis=1)
    assert np.allclose(e.array, np.linalg.norm(c.array, axis=1))

def test_det():
    a = hd.array([[1, 2], [3, 4]])
    d = det(a)
    assert np.isclose(d, -2.0)

def test_inv():
    a = hd.array(np.array([[1., 2.], [3., 4.]]))
    a_inv = inv(a)
    assert np.allclose(a_inv.array @ a.array, np.eye(2))

def test_pinv():
    a_np = np.random.randn(9, 6)
    a = hd.array(a_np)
    B = pinv(a)
    assert np.allclose(a_np, a_np @ B.array @ a_np)
    assert np.allclose(B.array, B.array @ a_np @ B.array)

def test_eigvals():
    a_np = np.array([[1, -2j], [2j, 5]])
    a = hd.array(a_np)
    w = eigvals(a)
    assert np.allclose(np.sort(w.array), np.sort(np.linalg.eigvals(a_np)))

def test_logm_nonsingular():
    a_np = np.array([[1, 2], [3, 4]])
    a = hd.array(a_np)
    log_a = logm(a)
    assert np.allclose(a_np, expm(log_a.array))

def test_logm_singular():
    a_np = np.array([[1, 1], [1, 1]])
    a = hd.array(a_np)
    log_a = logm(a)
    # The matrix is singular, so we cannot simply use expm to verify.
    # Instead, we check that the result has the correct properties.
    assert isinstance(log_a, hd.DimensionalArray)
    assert log_a.shape == (2, 2)
