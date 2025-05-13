"""Pytest module for Average Precision (AP) and Mean Average Precision (MAP) with TF-IDF.
This module tests the MAP calculation using TF-IDF to generate predictions.
"""

import pytest
from utils import compute_average_precision, compute_mean_average_precision
from utils import xp, compute_tfidf

# --- Tests for MAP using TF-IDF to generate predictions ---

@pytest.fixture
def sample_documents():
    """Sample documents for TF-IDF testing."""
    return [
        # "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the river",
        "the dog is lazy",
        "the fox is quick and brown"
    ]

def simple_tfidf_ranker(query, documents, scheme_tf="raw", scheme_idf="raw", normalize=True):
    """A basic TF-IDF ranker using the provided TF-IDF functions."""
    # 1. Create a vocabulary
    vocabulary = set()
    for doc in documents:
        vocabulary.update(doc.lower().split())
    vocabulary = list(vocabulary)
    term_to_index = {term: i for i, term in enumerate(vocabulary)}

    # 2. Create a term-document matrix
    term_document_matrix = xp.zeros((len(documents), len(vocabulary)))
    for i, doc in enumerate(documents):
        for term in doc.lower().split():
            if term in term_to_index:
                term_document_matrix[i, term_to_index[term]] += 1

    # 3. Compute TF-IDF
    tfidf_matrix = compute_tfidf(term_document_matrix, scheme_tf=scheme_tf, scheme_idf=scheme_idf, normalize=normalize)

    # 4. Vectorize the query
    query_vector = xp.zeros(len(vocabulary))
    for term in query.lower().split():
        if term in term_to_index:
            query_vector[term_to_index[term]] += 1
    query_tfidf = compute_tfidf(xp.array([query_vector]), scheme_tf=scheme_tf, scheme_idf=scheme_idf, normalize=normalize)[0]

    # 5. Compute cosine similarity
    similarities = []
    for doc_tfidf in tfidf_matrix:
        # print("Doc TF-IDF:", doc_tfidf)
        # print("Query TF-IDF:", query_tfidf)
        norm_query = xp.linalg.norm(query_tfidf)
        norm_doc = xp.linalg.norm(doc_tfidf)
        if norm_query > 0 and norm_doc > 0:
            similarity = xp.dot(query_tfidf, doc_tfidf) / (norm_query * norm_doc)
        else:
            similarity = 0.0
        similarities.append(similarity)

    # 6. Rank documents by similarity
    ranked_indices = xp.argsort(xp.array(similarities))[::-1]
    ranked_documents = [documents[i] for i in ranked_indices.tolist()]  # Convert to list here
    print("Documents:", documents)
    print("Similarities:", similarities)
    return ranked_documents

def test_compute_ap_with_tfidf(sample_documents):
    """Test the Average Precision (AP) calculation using TF-IDF to generate ranked predictions."""
    query = "quick brown fox"
    predictions = simple_tfidf_ranker(query, sample_documents, scheme_tf="raw", scheme_idf="raw", normalize=True)
    print("\nRanked predictions:", predictions)
    ground_truth = ["the quick brown fox jumps over the river", "the fox is quick and brown"]

    # Compute the Average Precision (AP) score
    ap_score = compute_average_precision(ground_truth, predictions)

    # Ensure the AP score is within the valid range [0, 1]
    assert 0.0 <= ap_score <= 1.0

    # Expected AP calculation: all relevant documents are in the top 2 ranks
    expected_ap = (1/1 + 2/2) / 2

    # Verify that the computed AP matches the expected value
    assert ap_score == pytest.approx(expected_ap, rel=1e-2)

def test_compute_map_with_tfidf_predictions(sample_documents):
    """Test the Mean Average Precision (MAP) calculation using TF-IDF to generate ranked predictions."""
    queries = ["quick brown fox", "lazy dog", "the fox is quick"]
    predictions = [
        simple_tfidf_ranker(q, sample_documents) for q in queries
    ]
    print("\nRanked predictions:", predictions)
    ground_truths = [
        ["the quick brown fox jumps over the river", "the fox is quick and brown"],
        ["the dog is lazy"],
        ["the fox is quick and brown", "the quick brown fox jumps over the river"]
    ]

    # Compute the Mean Average Precision (MAP) score
    map_score = compute_mean_average_precision(ground_truths, predictions)

    # Ensure the MAP score is within the valid range [0, 1]
    assert 0.0 <= map_score <= 1.0

    # Expected MAP calculation: all relevant documents are in the top 3 ranks
    expected_map = (1 + 1 + 1) / 3

    # Verify that the computed MAP matches the expected value
    assert map_score == pytest.approx(expected_map, rel=1e-2)

def test_compute_map_with_tfidf_predictions_2(sample_documents):
    """Test the MAP calculation with a case where some queries have no relevant documents."""
    queries = ["quick brown fox", "lazy dog", "the fox is quick"]
    predictions = [
        simple_tfidf_ranker(q, sample_documents) for q in queries
    ]
    print("\nRanked predictions:", predictions)
    ground_truths = [
        ["the quick brown fox jumps over the river", "the fox is quick and brown"],
        ["the dog is lazy"],
        ["the dog is lazy"]  # Simulate a case where the query is not relevant
    ]

    # Compute the MAP score
    map_score = compute_mean_average_precision(ground_truths, predictions)

    # Ensure the MAP score is within the valid range [0, 1]
    assert 0.0 <= map_score <= 1.0

    # Expected MAP calculation with one query having no relevant documents
    ap_3 = (0/1 + 0/2 + 1/3) / 1
    expected_map = (1 + 1 + ap_3) / 3

    # Verify that the computed MAP matches the expected value
    assert map_score == pytest.approx(expected_map, rel=1e-2)

def test_compute_map_with_tfidf_and_k(sample_documents):
    """Test the MAP calculation with a cutoff parameter k using TF-IDF predictions."""
    queries = ["quick brown fox", "lazy dog"]
    predictions = [
        simple_tfidf_ranker(q, sample_documents) for q in queries
    ]
    ground_truths = [
        ["the quick brown fox jumps over the river", "the fox is quick and brown"],
        ["the dog is lazy"]
    ]
    k = 2  # Cutoff parameter for MAP calculation

    # Compute the MAP score at k
    map_score_at_k = compute_mean_average_precision(ground_truths, predictions, k=k)

    # Ensure the MAP score at k is within the valid range [0, 1]
    assert 0.0 <= map_score_at_k <= 1.0

    # Expected MAP calculation with a cutoff at k=2
    expected_map_at_k = (1 + 1) / 2

    # Verify that the computed MAP at k matches the expected value
    assert map_score_at_k == pytest.approx(expected_map_at_k, rel=1e-2)
    
