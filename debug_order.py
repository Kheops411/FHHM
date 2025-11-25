import numpy as np
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python import utils

# Load C++ Dump
ref = np.loadtxt("python/tests/ref_outputs/score_dump_ref.txt")

# Load Python
notes = utils.load_pig_file("scores/001-1_fingering.txt")
rh_notes = utils.filter_notes_by_hand(notes, 0)
py_ordered = utils.apply_time_dep_pitch_order(rh_notes)

# Compare
print(f"Length: Py={len(py_ordered)} C++={len(ref)}")
for i in range(min(len(py_ordered), len(ref))):
    py_p = py_ordered[i]['pitch']
    ref_p = ref[i, 2]
    if py_p != ref_p:
        print(f"MISMATCH at index {i}: Py Pitch={py_p}, C++ Pitch={ref_p}")
        break
