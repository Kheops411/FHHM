import numpy as np
import sys
import os

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import generate_luts

def test_generate_luts():
    """
    Mandatory Test (`tests/test_utils.py`):
    - Call `generate_luts`.
    - Assert: `len(keypos_lut) == 128`.
    - Assert: `keypos_lut[61] - keypos_lut[60]` is approx `1.17` (C4 to C#4 distance).
    - Assert: `keypos_lut[77] - keypos_lut[65]` is **exactly** `16.5` (F4 to F5 distance).
    """
    keypos_lut, is_black_lut = generate_luts()

    # Assert: `len(keypos_lut) == 128`
    assert len(keypos_lut) == 128
    assert len(is_black_lut) == 128

    # Assert: `keypos_lut[61] - keypos_lut[60]` is approx `1.17`
    c_sharp_4_pos = keypos_lut[61]
    c_4_pos = keypos_lut[60]
    diff_c_c_sharp = c_sharp_4_pos - c_4_pos
    assert np.isclose(diff_c_c_sharp, 1.17857, atol=1e-5)

    # Assert: `keypos_lut[77] - keypos_lut[65]` is **exactly** `16.5`
    f5_pos = keypos_lut[77]
    f4_pos = keypos_lut[65]
    diff_f4_f5 = f5_pos - f4_pos
    assert np.isclose(diff_f4_f5, 16.5)

    print("All tests in test_generate_luts passed!")

if __name__ == "__main__":
    test_generate_luts()
