import numpy as np
import pytest
import sys
import os
import subprocess
import re

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import evaluate

def parse_cpp_evaluation_output(output: str) -> dict:
    """
    Parses the stdout of the C++ evaluation binary.
    The C++ binary outputs a header line and then a tab-separated value line.
    Example: General,Highest,Soft,Recomb: 0.653518	0.703625	0.759062	0.722814
    """
    metrics = {}
    lines = output.strip().split('\\n')
    # The last non-empty line contains the data
    data_line = [line for line in lines if line][-1]

    parts = data_line.split(':')
    if len(parts) != 2:
        pytest.fail(f"Unexpected C++ output format. Line: '{data_line}'")

    keys_str, vals_str = parts
    keys = keys_str.split(',')
    vals = [float(v) for v in vals_str.strip().split()]

    if len(keys) != len(vals):
        pytest.fail("Mismatch between number of metric keys and values in C++ output.")

    metrics = dict(zip(keys, vals))
    return metrics

def test_evaluation_metrics_match_live_cpp():
    """
    Verifies the Python evaluation metrics against a live run of the C++ binary.
    """
    # 1. Define paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_binary = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Binary', 'Evaluate_MultipleGroundTruth'))

    # Input files for the evaluation
    gt_file1 = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', '001-1_fingering.txt'))
    gt_file2 = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', '001-2_fingering.txt'))

    # We need a reference estimated file. Let's generate one for 001-1.
    est_file = os.path.realpath(os.path.join(test_dir, 'ref_outputs', 'ref_001_est_for_eval.txt'))
    hmm_runner = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Binary', 'FingeringHMM2_Run'))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM2.txt'))
    subprocess.run([hmm_runner, param_file, gt_file1, est_file, "0.5", "0.5", "0", "-5"], check=True)

    gt_files = [gt_file1, gt_file2]
    num_gt = len(gt_files)

    # 2. Generate C++ reference metrics
    command = [cpp_binary, str(num_gt)] + gt_files + [est_file]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    cpp_metrics = parse_cpp_evaluation_output(process.stdout)

    # 3. Run Python evaluation
    py_metrics = evaluate.calculate_metrics(gt_files, est_file)

    # 4. Compare metrics
    assert cpp_metrics.keys() == py_metrics.keys(), "Metric keys mismatch"
    for key in cpp_metrics:
        assert np.isclose(py_metrics[key], cpp_metrics[key], atol=1e-6), \
            f"Metric '{key}' mismatch. Python: {py_metrics[key]}, C++: {cpp_metrics[key]}"

    # Clean up the generated estimation file
    os.remove(est_file)
