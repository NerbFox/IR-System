"""Term frequency (TF) and inverse document frequency (IDF) calculation module.

This module provides functions for computing various TF-IDF schemes:
- Term Frequency (TF) schemes: raw, log, binary, augmented
- Inverse Document Frequency (IDF) schemes: raw, log
"""

from .cuda_utils import xp

def compute_tf(tf_matrix, scheme="raw"):
    """
    Compute the term frequency (TF) for a given term frequency matrix.

    Args:
        tf_matrix (numpy.ndarray): The term frequency matrix.
        scheme (str): The scheme to use for computing TF. Options are "raw", "log", "binary", and "augmented".

    Returns:
        numpy.ndarray or cupy.ndarray: The computed TF matrix.
    """
    if scheme == "raw":
        return tf_matrix
    if scheme == "log":
        return xp.where(tf_matrix > 0, 1 + xp.log2(tf_matrix), 0)
    if scheme == "binary":
        return xp.where(tf_matrix > 0, 1, 0)
    if scheme == "augmented":
        # Avoid division by zero
        max_tf = xp.maximum(xp.max(tf_matrix, axis=1, keepdims=True), 1e-10)
        return 0.5 + 0.5 * (tf_matrix / max_tf)
    raise ValueError("Invalid scheme. Choose from 'raw', 'log', 'binary', 'augmented'.")

def compute_idf(tf_matrix, scheme="raw"):
    """
    Compute the inverse document frequency (IDF) for a given term frequency matrix.

    Args:
        tf_matrix (numpy.ndarray or cupy.ndarray): The term frequency matrix.
        scheme (str): The scheme to use for computing IDF. Options are "raw" and "log".

    Returns:
        numpy.ndarray or cupy.ndarray: The computed IDF vector (1D).
    """
    n_docs = tf_matrix.shape[0]
    doc_freq = xp.sum(tf_matrix > 0, axis=0)
    if scheme == "raw":
        return xp.ones_like(doc_freq, dtype=tf_matrix.dtype)
    if scheme == "log":
        # Add small epsilon to avoid division by 0
        return xp.log2(n_docs / (doc_freq + 1e-10))
    raise ValueError("Invalid scheme. Choose from 'raw', 'log'.")

def compute_tfidf(tf_matrix, scheme_tf="raw", scheme_idf="raw", normalize=True):
    """
    Compute the TF-IDF matrix.

    Args:
        tf_matrix (numpy.ndarray or cupy.ndarray): The term frequency matrix.
        scheme_tf (str): TF scheme: "raw", "log", "binary", or "augmented".
        scheme_idf (str): IDF scheme: "raw" or "log".
        normalize (bool): Whether to apply cosine normalization to the resulting matrix.

    Returns:
        numpy.ndarray or cupy.ndarray: The normalized TF-IDF matrix.
    """
    tf = compute_tf(tf_matrix, scheme_tf)
    idf = compute_idf(tf_matrix, scheme_idf)
    tfidf = tf * idf

    if normalize:
        norm = xp.linalg.norm(tfidf, axis=0, keepdims=True)
        norm = xp.maximum(norm, 1e-10)  # Avoid division by zero
        return tfidf / norm
    return tfidf

def compute_tfidf_input(tf_matrix, scheme_tf="raw", scheme_idf="raw", normalize=True, source_idf=None):
    """
    Compute the TF-IDF matrix.

    Args:
        tf_matrix (numpy.ndarray or cupy.ndarray): The term frequency matrix.
        scheme_tf (str): TF scheme: "raw", "log", "binary", or "augmented".
        scheme_idf (str): IDF scheme: "raw" or "log".
        normalize (bool): Whether to apply cosine normalization to the resulting matrix.

    Returns:
        numpy.ndarray or cupy.ndarray: The normalized TF-IDF matrix.
    """
    tf = compute_tf(tf_matrix, scheme_tf)
    idf = source_idf
    tfidf = tf * idf

    if normalize:
        norm = xp.linalg.norm(tfidf, axis=0, keepdims=True)
        norm = xp.maximum(norm, 1e-10)  # Avoid division by zero
        return tfidf / norm
    return tfidf