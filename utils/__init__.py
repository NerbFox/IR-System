"""Utility module for IR-System that handles array computation configuration.

This module automatically selects the appropriate array computation library
(CuPy for GPU or NumPy for CPU) based on hardware availability. It exports:
    - xp: The array computation module (either CuPy or NumPy)
    - GPU_AVAILABLE: Boolean indicating if GPU computation is available
"""

from .cuda_utils import xp
from .tf_idf_calculation import compute_tf, compute_idf, compute_tfidf
from .map_calculation import compute_average_precision, compute_mean_average_precision
from .bert_calculation import compute_bert, compute_bert_document_embeddings, rank_documents_by_similarity, get_bert_model_and_tokenizer
from .OriginalProcess import OriginalProcess
from .ExpandProcess import ExpandProcess
from .process import calculate_inverted

__all__ = ['xp', 'compute_tf', 'compute_idf', 'compute_tfidf',
           'compute_average_precision', 'compute_mean_average_precision',
           'compute_bert', 'compute_bert_document_embeddings', 'rank_documents_by_similarity',
           'get_bert_model_and_tokenizer', 'OriginalProcess', 'calculate_inverted', 'ExpandProcess']
