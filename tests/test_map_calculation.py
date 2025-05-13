"""Pytest module for Average Precision (AP) and Mean Average Precision (MAP) calculation functions.
"""

import pytest
from utils import compute_average_precision, compute_mean_average_precision

# --- Tests for compute_average_precision ---

def test_compute_ap_basic():
    """Test basic Average Precision calculation."""
    ground_truth = [1, 2, 3]
    predictions = [1, 4, 2, 5, 3]
    expected_ap = (1/1 + 2/3 + 3/5) / 3
    assert compute_average_precision(ground_truth, predictions) == pytest.approx(expected_ap)

def test_compute_ap_empty_ground_truth():
    """Test AP with empty ground truth."""
    ground_truth = []
    predictions = [1, 2, 3]
    assert compute_average_precision(ground_truth, predictions) == 0.0

def test_compute_ap_empty_predictions():
    """Test AP with empty predictions."""
    ground_truth = [1, 2, 3]
    predictions = []
    assert compute_average_precision(ground_truth, predictions) == 0.0

def test_compute_ap_no_relevant_in_predictions():
    """Test AP when no relevant items are in predictions."""
    ground_truth = [1, 2, 3]
    predictions = [4, 5, 6]
    assert compute_average_precision(ground_truth, predictions) == 0.0

def test_compute_ap_k_parameter():
    """Test AP with the k parameter."""
    ground_truth = [1, 2, 3]
    predictions = [1, 4, 2, 5, 3]
    k = 3
    expected_ap_at_k = (1/1 + 2/3) / len(ground_truth)
    assert compute_average_precision(ground_truth, predictions, k=k) == pytest.approx(expected_ap_at_k)

def test_compute_ap_k_greater_than_predictions():
    """Test AP with k greater than the number of predictions."""
    ground_truth = [1, 2]
    predictions = [1]
    k = 5
    expected_ap = 1/1 / len(ground_truth)
    assert compute_average_precision(ground_truth, predictions, k=k) == pytest.approx(expected_ap)

def test_compute_ap_different_data_types():
    """Test AP with different data types in ground truth and predictions."""
    ground_truth = ["apple", "banana"]
    predictions = ["apple", "orange", "banana"]
    expected_ap = (1/1 + 2/3) / 2
    assert compute_average_precision(ground_truth, predictions) == pytest.approx(expected_ap)

# --- Tests for compute_mean_average_precision ---

def test_compute_map_basic():
    """Test basic Mean Average Precision calculation."""
    ground_truths = [[1, 2], [3]]
    predictions = [[1, 3, 2], [3, 4]]
    ap1 = (1/1 + 2/3) / 2
    ap2 = 1/1 / 1
    expected_map = (ap1 + ap2) / 2
    assert compute_mean_average_precision(ground_truths, predictions) == pytest.approx(expected_map)

def test_compute_map_empty_ground_truths():
    """Test MAP with empty ground truths."""
    ground_truths = []
    predictions = [[]]
    assert compute_mean_average_precision(ground_truths, predictions) == 0.0

def test_compute_map_empty_predictions():
    """Test MAP with empty predictions."""
    ground_truths = [[1], [2]]
    predictions = [[], []]
    assert compute_mean_average_precision(ground_truths, predictions) == 0.0

def test_compute_map_unequal_number_of_queries():
    """Test MAP with unequal number of ground truths and predictions."""
    ground_truths = [[1]]
    predictions = [[1], [2]]
    assert compute_mean_average_precision(ground_truths, predictions) == 0.0

def test_compute_map_k_parameter_1():
    """Test MAP with the k parameter."""
    ground_truths = [[1, 2], [3]]
    predictions = [[1, 3, 2], [3, 4]]
    k = 2
    ap1_at_k = (1/1) / len(ground_truths[0])  
    ap2_at_k = 1/1 / len(ground_truths[1])
    expected_map_at_k = (ap1_at_k + ap2_at_k) / 2
    assert compute_mean_average_precision(ground_truths, predictions, k=k) == pytest.approx(expected_map_at_k)
    
def test_compute_map_k_parameter_2():
    """Test MAP with the k parameter."""
    ground_truths = [[1, 2], [3]]
    predictions = [[1, 3, 2], [3, 4]]
    ap1_at_k = (1/1 + 2/3) / len(ground_truths[0])  
    ap2_at_k = 1/1 / len(ground_truths[1])
    expected_map_at_k = (ap1_at_k + ap2_at_k) / 2
    assert compute_mean_average_precision(ground_truths, predictions) == pytest.approx(expected_map_at_k)

def test_compute_map_with_empty_query_results():
    """Test MAP with some empty query results."""
    ground_truths = [[1], [2], [3]]
    predictions = [[1], [], [3, 4]]
    ap1 = 1/1 / 1
    ap2 = 0.0
    ap3 = 1/1 / 1
    expected_map = (ap1 + ap2 + ap3) / 3
    assert compute_mean_average_precision(ground_truths, predictions) == pytest.approx(expected_map)
    
# --- Tests with expected results (manual calculation for simple cases) ---

def test_compute_ap_expected_result_simple():
    """Test AP with a manually calculated expected result."""
    ground_truth = [1, 2]
    predictions = [1, 3, 2]
    # Rank 1: 1 is relevant (precision = 1/1 = 1)
    # Rank 2: 3 is not relevant
    # Rank 3: 2 is relevant (precision = 2/3)
    # AP = (1 + 2/3) / 2 = 5/6
    expected_ap = 5/6
    assert compute_average_precision(ground_truth, predictions) == pytest.approx(expected_ap)

def test_compute_map_expected_result_simple():
    """Test MAP with a manually calculated expected result for two queries."""
    ground_truths = [[1], [2, 3]]
    predictions = [[1, 4], [3, 5, 2]]
    # Query 1: GT=[1], Pred=[1, 4] -> AP = 1/1 / 1 = 1.0
    # Query 2: GT=[2, 3], Pred=[3, 5, 2] ->
    #   Rank 1: 3 is relevant (precision = 1/1 = 1)
    #   Rank 2: 5 is not relevant
    #   Rank 3: 2 is relevant (precision = 2/3)
    #   AP = (1 + 2/3) / 2 = 5/6
    expected_map = (1.0 + 5/6) / 2
    assert compute_mean_average_precision(ground_truths, predictions) == pytest.approx(expected_map)