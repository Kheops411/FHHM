import numpy as np
import pytest
import numba as nb
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import utils

def test_lut_integrity():
    """Ensure LUT is C-contiguous and int16 for Numba efficiency."""
    assert utils.PITCH_TO_KEYPOS_LUT.dtype == np.int16
    assert utils.PITCH_TO_KEYPOS_LUT.flags['C_CONTIGUOUS']

def test_pitch_to_keypos_exact_match():
    """Compare Python LUT against C++ keypos_ref.cpp output."""
    # Load C++ Ground Truth
    ref_data = np.loadtxt("python/tests/ref_outputs/pitch_to_keypos_reference.txt", dtype=int)

    for row in ref_data:
        pitch, expected_x, expected_y = row
        py_x, py_y = utils.pitch_to_keypos(pitch)

        assert py_x == expected_x, f"X Mismatch at pitch {pitch}"
        assert py_y == expected_y, f"Y Mismatch at pitch {pitch}"

def test_subtract_keypos_exact_match():
    """Compare subtraction logic against C++ subtr_ref.cpp output."""
    ref_data = np.loadtxt("python/tests/ref_outputs/subtract_keypos_reference.txt", dtype=int)

    for row in ref_data:
        x1, y1, x2, y2, exp_dx, exp_dy = row
        dx, dy = utils.subtract_keypos((x1, y1), (x2, y2))
        assert (dx, dy) == (exp_dx, exp_dy), f"Mismatch at {x1},{y1} - {x2},{y2}"

def test_numba_compilation():
    """Ensure the hot-path functions compile without object-mode fallback."""
    # This will raise if Numba cannot compile in nopython mode
    lut = utils.PITCH_TO_KEYPOS_LUT

    @nb.njit
    def driver():
        utils.pitch_to_keypos_numba(60, lut)
        utils.subtract_keypos_numba(1, 0, 2, 1)
        utils.lattice_delta_to_index(5, 1)

    # The call itself will trigger compilation and raise on failure
    driver()

def test_lattice_delta_to_index_clamping():
    """Verify the clamping logic of the index helper."""
    width_x = 15
    # Test lower bound
    assert utils.lattice_delta_to_index(-width_x - 5, 0, width_x) == utils.lattice_delta_to_index(-width_x, 0, width_x)
    # Test upper bound
    assert utils.lattice_delta_to_index(width_x + 5, 1, width_x) == utils.lattice_delta_to_index(width_x, 1, width_x)
    # Test within bounds
    assert utils.lattice_delta_to_index(5, 0, width_x) == 3 * (5 + width_x) + 0 + 1
