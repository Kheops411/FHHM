import numpy as np
import pytest
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import evaluate

def test_evaluation_metrics_match_cpp():
    # These are the values you recorded from running the C++ binary
    expected_metrics = {
        "General": 0.653518,
        "Highest": 0.703625,
        "Soft": 0.759062,
        "Recomb": 0.722814
    }

    gt_files = ["scores/001-1_fingering.txt", "scores/001-2_fingering.txt"]
    est_file = "python/tests/ref_outputs/ref_001.txt"

    actual_metrics = evaluate.calculate_metrics(gt_files, est_file)

    for key in expected_metrics:
        assert abs(actual_metrics[key] - expected_metrics[key]) < 1e-3
