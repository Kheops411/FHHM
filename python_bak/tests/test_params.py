import numpy as np
import pytest
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python.model import HMMParameters

# Epsilon for comparing floating point numbers, should be tight
LOG_PROB_EPSILON = 1e-6

def load_reference_data(filepath):
    """Loads the C++ reference data into a dictionary."""
    ref_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            key = parts[0]
            value = float(parts[1])
            ref_data[key] = value
    return ref_data

@pytest.fixture(scope="module")
def hmm_params():
    """Fixture to load the HMM parameters once for all tests in this module."""
    # Build a robust, absolute path to make test execution location-independent
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM2.txt'))
    return HMMParameters(param_file)

@pytest.fixture(scope="module")
def ref_params():
    """Fixture to load the C++ reference parameters once."""
    # Build a robust, absolute path
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ref_file = os.path.realpath(os.path.join(test_dir, 'ref_outputs', 'param_reference.txt'))
    return load_reference_data(ref_file)

def test_initial_prob(hmm_params, ref_params):
    """Test a specific initial probability (Right Hand, finger 2)."""
    # Corresponds to hand=0, finger=1
    py_log_prob = hmm_params.log_initial_prob[0, 1]
    ref_log_prob = np.log(ref_params["INIT_R_2"])
    assert np.isclose(py_log_prob, ref_log_prob, atol=LOG_PROB_EPSILON)

def test_transition1_prob(hmm_params, ref_params):
    """Test a specific 1st order transition (Right Hand, 2 -> 3)."""
    # Corresponds to hand=0, prev_f=1, curr_f=2
    py_log_prob = hmm_params.log_transition1_prob[0, 1, 2]
    ref_log_prob = np.log(ref_params["TR1_R_2_3"])
    assert np.isclose(py_log_prob, ref_log_prob, atol=LOG_PROB_EPSILON)

def test_transition2_prob(hmm_params, ref_params):
    """Test a specific 2nd order transition (Right Hand, 1 -> 2 -> 3)."""
    # Corresponds to hand=0, f_n-2=0, f_n-1=1, f_n=2
    py_log_prob = hmm_params.log_transition2_prob[0, 0, 1, 2]
    ref_log_prob = np.log(ref_params["TR2_R_1_2_3"])
    assert np.isclose(py_log_prob, ref_log_prob, atol=LOG_PROB_EPSILON)

def test_output1_prob(hmm_params, ref_params):
    """Test a specific 1st order output prob (RH, 1->2, C4->E4)."""
    # Corresponds to hand=0, prev_f=0, curr_f=1, idx=52
    idx_C4E4 = 52
    py_log_prob = hmm_params.log_output1_prob[0, 0, 1, idx_C4E4]
    ref_log_prob = np.log(ref_params["OUT1_R_1_2_C4E4"])
    assert np.isclose(py_log_prob, ref_log_prob, atol=LOG_PROB_EPSILON)

def test_output2_prob(hmm_params, ref_params):
    """Test a specific 2nd order output prob (LH, 1->2, C4->G3)."""
    # Corresponds to hand=1, f_n-2=0, f_n=1, idx=37
    idx_C4G3 = 37
    py_log_prob = hmm_params.log_output2_prob[1, 0, 1, idx_C4G3]
    ref_log_prob = np.log(ref_params["OUT2_L_1_2_C4G3"])
    assert np.isclose(py_log_prob, ref_log_prob, atol=LOG_PROB_EPSILON)
