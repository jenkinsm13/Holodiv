import numpy as np
import pytest
import holodiv as hd
from holodiv.random import rand, randn, randint, random, normal, uniform, multivariate_normal, seed

def test_rand():
    a = rand(3, 3)
    assert isinstance(a, hd.DimensionalArray)
    assert a.shape == (3, 3)
    assert np.all(a.array >= 0) and np.all(a.array < 1)

def test_randn():
    a = randn(1000)
    assert isinstance(a, hd.DimensionalArray)
    assert a.shape == (1000,)
    assert np.abs(np.mean(a.array)) < 0.1
    assert np.abs(np.std(a.array) - 1) < 0.1

def test_randint():
    a = randint(0, 10, (3, 3))
    assert isinstance(a, hd.DimensionalArray)
    assert a.shape == (3, 3)
    assert np.all(a.array >= 0) and np.all(a.array < 10)

    # Test with only low
    b = randint(5)
    assert isinstance(b, hd.DimensionalArray)
    assert b.array >= 0 and b.array < 5

def test_random():
    a = random((3, 3))
    assert isinstance(a, hd.DimensionalArray)
    assert a.shape == (3, 3)
    assert np.all(a.array >= 0) and np.all(a.array < 1)

def test_normal():
    a = normal(5, 2, 1000)
    assert isinstance(a, hd.DimensionalArray)
    assert a.shape == (1000,)
    assert np.abs(np.mean(a.array) - 5) < 0.2
    assert np.abs(np.std(a.array) - 2) < 0.2

def test_uniform():
    a = uniform(5, 10, (3, 3))
    assert isinstance(a, hd.DimensionalArray)
    assert a.shape == (3, 3)
    assert np.all(a.array >= 5) and np.all(a.array < 10)

def test_multivariate_normal():
    mean = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    a = multivariate_normal(mean, cov, 1000)
    assert isinstance(a, hd.DimensionalArray)
    assert a.shape == (1000, 2)
    assert np.all(np.abs(np.mean(a.array, axis=0)) < 0.2)
    assert np.all(np.abs(np.cov(a.array.T) - cov) < 0.2)


def test_seed():
    seed(0)
    a = random((3, 3))
    seed(0)
    b = random((3, 3))
    assert np.allclose(a.array, b.array)
