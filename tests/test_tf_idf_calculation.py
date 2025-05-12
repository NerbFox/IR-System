"""Test module for TF-IDF calculation functions.

Tests TF schemes (raw, log, binary, augmented), IDF schemes (raw, log),
and TF-IDF computation with and without normalization.
"""

import pytest
from utils import xp, compute_tf, compute_idf, compute_tfidf

@pytest.fixture
def test_matrix():
    """Test matrix for TF-IDF calculations."""
    return xp.array([
        [3, 0, 1],
        [0, 2, 4],
        [5, 5, 0]
    ])

def test_compute_tf_raw(test_matrix):
    """Test raw term frequency calculation."""
    expected = xp.array([
        [3, 0, 1],
        [0, 2, 4],
        [5, 5, 0]
    ])
    result = compute_tf(test_matrix, "raw")
    xp.testing.assert_array_equal(result, expected)

def test_compute_tf_log(test_matrix):
    """Test log term frequency calculation."""
    expected = xp.array([
        [2.58496250, float('-inf'), 1.0],
        [float('-inf'), 2.0, 3.0],
        [3.32192809, 3.32192809, float('-inf')]
    ])
    result = compute_tf(test_matrix, "log")
    xp.testing.assert_array_almost_equal(result, expected, decimal=5)

def test_compute_tf_binary(test_matrix):
    """Test binary term frequency calculation."""
    expected = xp.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0]
    ])
    result = compute_tf(test_matrix, "binary")
    xp.testing.assert_array_equal(result, expected)

def test_compute_tf_augmented(test_matrix):
    """Test augmented term frequency calculation."""
    expected = xp.array([
        [1.0, 0.5, 0.66666667],
        [0.5, 0.75, 1.0],
        [1.0, 1.0, 0.5]
    ])
    result = compute_tf(test_matrix, "augmented")
    xp.testing.assert_array_almost_equal(result, expected, decimal=5)

def test_compute_tf_invalid_scheme(test_matrix):
    """Test invalid term frequency scheme raises ValueError."""
    with pytest.raises(ValueError):
        compute_tf(test_matrix, "invalid")

def test_compute_idf_raw(test_matrix):
    """Test raw inverse document frequency calculation."""
    expected = xp.array([1.0, 1.0, 1.0])
    result = compute_idf(test_matrix, "raw")
    xp.testing.assert_array_equal(result, expected)

def test_compute_idf_log(test_matrix):
    """Test log inverse document frequency calculation."""
    expected = xp.array([0.58496250, 0.58496250, 0.58496250])
    result = compute_idf(test_matrix, "log")
    xp.testing.assert_array_almost_equal(result, expected, decimal=5)

def test_compute_idf_invalid_scheme(test_matrix):
    """Test invalid inverse document frequency scheme raises ValueError."""
    with pytest.raises(ValueError):
        compute_idf(test_matrix, "invalid")

def test_compute_tfidf_normalized(test_matrix):
    """Test TF-IDF calculation with normalization."""
    expected = xp.array([
        [0.9486833, 0.0,      0.31622777],
        [0.0,      0.4472136, 0.89442719],
        [0.70710678, 0.70710678, 0.0]
    ])
    result = compute_tfidf(test_matrix, scheme_tf="raw", scheme_idf="log", normalize=True)
    xp.testing.assert_array_almost_equal(result, expected, decimal=5)

def test_compute_tfidf_non_normalized(test_matrix):
    """Test TF-IDF calculation without normalization."""
    expected = xp.array([
        [1.75488750, 0.0, 0.58496250],
        [0.0, 1.16992500, 2.33985000],
        [2.92481250, 2.92481250, 0.0]
    ])
    result = compute_tfidf(test_matrix, scheme_tf="raw", scheme_idf="log", normalize=False)
    xp.testing.assert_array_almost_equal(result, expected, decimal=5)
