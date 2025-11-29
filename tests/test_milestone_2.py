import numpy as np
import psutil
import os
import sys

# Ensure the test can find the 'soft_position_hmm' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from soft_position_hmm.structural import ViterbiLattice, N_FINGERS, N_ANCHORS
from soft_position_hmm.core import ANCHORS

def test_dimension_integrity():
    """
    Test A: Validates the dimensions and initial values of the ViterbiLattice.
    """
    print("--- Running Test A: Dimension Integrity ---")
    n_obs = 100
    lattice = ViterbiLattice(n_obs)

    expected_log_probs_shape = (n_obs, N_FINGERS, N_FINGERS, N_ANCHORS)
    expected_backpointers_shape = (n_obs, N_FINGERS, N_FINGERS, N_ANCHORS, 3)

    print(f"log_probs shape: {lattice.log_probs.shape}, Expected: {expected_log_probs_shape}")
    print(f"backpointers shape: {lattice.backpointers.shape}, Expected: {expected_backpointers_shape}")

    assert lattice.log_probs.shape == expected_log_probs_shape
    assert lattice.backpointers.shape == expected_backpointers_shape
    assert lattice.log_probs[0, 0, 0, 0] == -np.inf
    assert lattice.backpointers[0, 0, 0, 0, 0] == -1

    print("Dimension integrity test passed.")
    print("-------------------------------------------\n")

def test_large_scale_memory_check():
    """
    Test B: Validates the memory consumption of a large ViterbiLattice.
    """
    print("--- Running Test B: Large Scale Memory Check ---")
    n_obs = 10000
    lattice = ViterbiLattice(n_obs)

    total_bytes = lattice.log_probs.nbytes + lattice.backpointers.nbytes
    total_mb = total_bytes / (1024 * 1024)

    print(f"Calculated memory for {n_obs} observations: {total_mb:.2f} MB")

    assert total_mb < 500, f"Memory usage ({total_mb:.2f} MB) exceeds 500 MB limit."

    print("Memory check passed.")
    print("-------------------------------------------\n")

def test_coordinate_mapping_sanity():
    """
    Test C: Verifies that the lattice is fully addressable up to its limits.
    """
    print("--- Running Test C: Coordinate Mapping Sanity ---")
    n_obs = 1
    lattice = ViterbiLattice(n_obs)

    print(f"N_ANCHORS from structural.py: {N_ANCHORS}")
    print(f"len(ANCHORS) from core.py: {len(ANCHORS)}")
    assert N_ANCHORS == len(ANCHORS)

    # Use broadcasting to set a value
    lattice.log_probs[0, :, :, :] = 1.0

    # Check the boundary index
    boundary_value = lattice.log_probs[0, N_FINGERS - 1, N_FINGERS - 1, N_ANCHORS - 1]
    print(f"Value at boundary (0, 4, 4, 8): {boundary_value}")
    assert boundary_value == 1.0

    print("Coordinate mapping sanity test passed.")
    print("-------------------------------------------\n")


if __name__ == "__main__":
    test_dimension_integrity()
    test_large_scale_memory_check()
    test_coordinate_mapping_sanity()
    print("All milestone 2 tests completed.")
