"""Pytest module for BERT calculation.
This module tests the BERT embedding and similarity calculation functions.
"""

import pytest
import numpy as np
from typing import Tuple
from utils import compute_bert, compute_bert_document_embeddings, rank_documents_by_similarity, get_bert_model_and_tokenizer
from transformers import BertTokenizer, BertModel

@pytest.fixture(scope="module")
def bert_model_and_tokenizer() -> Tuple[BertModel, BertTokenizer]:
    """
    Fixture to load the BERT model and tokenizer for testing.  Loaded once per test session.
    """
    model_name = 'bert-base-uncased'
    model, tokenizer = get_bert_model_and_tokenizer(model_name)
    return model, tokenizer

# --- Tests for BERT calculation functions ---

def test_compute_bert(bert_model_and_tokenizer: Tuple[BertModel, BertTokenizer]):
    """
    Test the compute_bert function.
    """
    model, tokenizer = bert_model_and_tokenizer
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "The dog is lazy.",
        "The fox is quick and brown."
    ]
    embeddings = compute_bert(sentences, model, tokenizer)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(sentences), 768)  # Check the shape of the embeddings (768 is the default hidden size for bert-base-uncased)

def test_rank_documents_by_similarity(bert_model_and_tokenizer: Tuple[BertModel, BertTokenizer]):
    """
    Test the rank_documents_by_similarity function.
    """
    model, tokenizer = bert_model_and_tokenizer
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A lazy dog sleeps in the sun.",
        "The fox is very quick and brown."
    ]
    embeddings = compute_bert(sentences, model, tokenizer)
    query_embedding = compute_bert(["quick brown fox jumps"], model, tokenizer)
    ranked_indices, similarities = rank_documents_by_similarity(embeddings, query_embedding)

    assert isinstance(ranked_indices, np.ndarray)
    assert isinstance(similarities, np.ndarray)
    assert ranked_indices.shape == (len(sentences),)
    assert similarities.shape == (len(sentences),)
    assert all(0 <= sim <= 1 for sim in similarities) #cosine similarity
    # the first document should be the most similar to the query 
    assert ranked_indices[0] == 0  # The first document should be ranked highest
    assert similarities[0] > 0.5  # The similarity should be greater than 0.5 for the first document

def test_rank_documents_by_similarity_same_embedding(bert_model_and_tokenizer: Tuple[BertModel, BertTokenizer]):
    """
    Test rank_documents_by_similarity when the query embedding is the same as one of the document embeddings.
    """
    model, tokenizer = bert_model_and_tokenizer
    sentences = [
        "This is a test sentence.",
        "That is another one of the word.",
        "There is a different word."
    ]
    embeddings = compute_bert(sentences, model, tokenizer)
    query = sentences[0] 
    query_embedding = compute_bert(query, model, tokenizer)
    ranked_indices, similarities = rank_documents_by_similarity(embeddings, query_embedding)
    print("Ranked Indices:", ranked_indices)
    print("Similarities:", similarities)
    assert ranked_indices[0] == 0  # The first document should be ranked highest
    # similarities should be near 1 for the first document
    assert similarities[0] > 0.95

