"""Mean Average Precision (MAP) calculation module.
This module provides functions for computing Mean Average Precision (MAP)
"""
"""Term frequency (TF) and inverse document frequency (IDF) calculation module.

This module provides functions for computing various TF-IDF schemes:
- Term Frequency (TF) schemes: raw, log, binary, augmented
- Inverse Document Frequency (IDF) schemes: raw, log
"""

def compute_average_precision(relevant_documents, retrieved_documents, k=None):
    """
    Compute Average Precision (AP) for a single query.

    Args:
        relevant_documents (list): List of truly relevant documents for the query.
        retrieved_documents (list): List of documents retrieved by the system.
        k (int, optional): Number of top retrieved documents to consider. If None, consider all.

    Returns:
        float: The computed AP score.
    """
    if k is not None:
        retrieved_documents = retrieved_documents[:k]

    relevant_document_set = set(relevant_documents)
    if not relevant_document_set or not retrieved_documents:
        return 0.0

    number_of_relevant_documents = len(relevant_documents) if k is None else min(k, len(relevant_documents))
    cumulative_precision = 0.0
    relevant_retrieved_count = 0

    for i, document in enumerate(retrieved_documents):
        if document in relevant_document_set:
            relevant_retrieved_count += 1
            cumulative_precision += relevant_retrieved_count / (i + 1)  # Precision at each relevant retrieval

    return cumulative_precision / number_of_relevant_documents if number_of_relevant_documents > 0 else 0.0

def compute_mean_average_precision(ground_truth_relevance, system_predictions, k=None):
    """
    Compute Mean Average Precision (MAP) across multiple queries.

    Args:
        ground_truth_relevance (list of list): List of relevant documents per query.
        system_predictions (list of list): List of retrieved documents per query.
        k (int, optional): Number of top predictions to consider. If None, consider all.

    Returns:
        float: The computed MAP score.
    """
    if not ground_truth_relevance or not system_predictions or len(ground_truth_relevance) != len(system_predictions):
        return 0.0

    ap_scores = [compute_average_precision(gt, pred, k) for gt, pred in zip(ground_truth_relevance, system_predictions)]
    return sum(ap_scores) / len(ground_truth_relevance) if ground_truth_relevance else 0.0