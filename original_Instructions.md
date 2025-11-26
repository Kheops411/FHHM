# **Project: High-Performance Python Port of Piano Fingering HMM**

## **Phase 1 & 2: Environment, Foundations, and Lattice Logic**

**Objective:** Establish the development environment, compile the reference C++ code, and implement the core mathematical foundation (KeyPos Lattice System) in Python. The Python implementation must be **bit-for-bit identical** to the C++ reference and optimized using **NumPy** and **Numba**.

### **Directory Structure**

Ensure your workspace is organized exactly as follows:

```text
.
├── cpp/                  # [EXISTING] Reference C++ implementation
│   ├── Code/             # Source files (.hpp, .cpp)
│   ├── Binary/           # Compiled executables (will be created)
│   ├── compile.sh        # Build script
│   └── ...
├── scores/               # [EXISTING] PIG Dataset (.txt files)
├── python/               # [NEW] Python package
│   ├── __init__.py
│   ├── utils.py          # Module to implement in Phase 2
│   └── tests/            # Unit tests
│       ├── ref_outputs/  # Storage for C++ reference dumps
│       └── ...
├── requirements.txt
└── README.md
```

---

### **Phase 1: Environment & Reference Verification**

**Goal:** Ensure the C++ code compiles and generates valid "Ground Truth" data.

1.  **Python Setup:**
    Create a virtual environment and install the required high-performance libraries.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy numba pytest
    ```

2.  **Compile C++ Reference:**
    You must use the provided `compile.sh` to build the original project.
    ```bash
    cd cpp
    chmod +x compile.sh
    ./compile.sh
    # Verify success:
    ls -l Binary/FingeringHMM2_Run
    cd ..
    ```

3.  **Generate Full-System Reference Outputs:**
    Run the C++ binary on real PIG scores to establish a baseline for the final integration tests (Phase N).
    ```bash
    mkdir -p python/tests/ref_outputs
    
    # Generate reference for a simple piece
    ./cpp/Binary/FingeringHMM2_Run scores/001-1_fingering.txt python/tests/ref_outputs/ref_001.txt
    
    # Generate reference for a complex/longer piece
    ./cpp/Binary/FingeringHMM2_Run scores/110-1_fingering.txt python/tests/ref_outputs/ref_110.txt
    ```
    *Check:* Ensure the output files `ref_*.txt` contain valid fingering data.

---

### **Phase 2: Core Lattice Logic (KeyPos) & Utils**

**Goal:** Implement `python/utils.py` containing the `KeyPos` logic.
**Constraint:** You must **not** guess the logic. You will compile specific C++ "Probes" to dump the logic's output, then write Python code that reproduces it exactly using a Lookup Table (LUT) for O(1) performance.

#### **Step 2.1: Create C++ Probes**

Create two C++ files in the root directory to dump the internal logic of `KeyPos_v161230.hpp`.

**File 1:** `keypos_ref.cpp` (Dumps Pitch -> Lattice mapping)
```cpp
#include "cpp/Code/KeyPos_v161230.hpp"
#include <iostream>

int main() {
    // Dump all MIDI pitches 0-127
    for (int p = 0; p < 128; ++p) {
        KeyPos kp = PitchToKeyPos(p);
        std::cout << p << " " << kp.x << " " << kp.y << "\n";
    }
    return 0;
}
```

**File 2:** `subtr_ref.cpp` (Dumps KeyPos Subtraction logic)
```cpp
#include "cpp/Code/KeyPos_v161230.hpp"
#include <iostream>

int main() {
    // Dump a range of subtractions to cover edge cases
    for (int x1=-40; x1<=40; x1+=5) {
        for (int y1=0; y1<=1; ++y1) {
            for (int x2=-40; x2<=40; x2+=5) {
                for (int y2=0; y2<=1; ++y2) {
                    KeyPos kp1; kp1.x=x1; kp1.y=y1;
                    KeyPos kp2; kp2.x=x2; kp2.y=y2;
                    KeyPos r = SubtrKeyPos(kp1,kp2);
                    std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << " " << r.x << " " << r.y << "\n";
                }
            }
        }
    }
    return 0;
}
```

**Compile and Run Probes:**
```bash
# Compile (adjust -I path if necessary)
g++ -O2 -std=c++17 -I cpp/Code keypos_ref.cpp -o keypos_ref
g++ -O2 -std=c++17 -I cpp/Code subtr_ref.cpp -o subtr_ref

# Generate Golden Data
./keypos_ref > python/tests/ref_outputs/pitch_to_keypos_reference.txt
./subtr_ref > python/tests/ref_outputs/subtract_keypos_reference.txt
```

#### **Step 2.2: Implement `python/utils.py`**

Implement the module using **NumPy** for data structures and **Numba** for performance.

**Requirements:**
1.  **Precomputed LUT:** `PITCH_TO_KEYPOS_LUT` must be a `(128, 2)` int16 array initialized at import time.
2.  **Logic Mirroring:** The logic in `_compute_pitch_to_keypos_lut` must conceptually mirror `KeyPos_v161230.hpp`.
3.  **JIT Compatibility:** Provide `numba.njit` compatible functions for the hot path.

**Skeleton Code:**
```python
import numpy as np
import numba as nb
from typing import Tuple

# Global LUT: (128 pitches, 2 coordinates [x, y])
PITCH_TO_KEYPOS_LUT = np.zeros((128, 2), dtype=np.int16)

def _compute_pitch_to_keypos_lut():
    """
    Port of C++ PitchToKeyPos from KeyPos_v161230.hpp
    Convention: C4=60=(0,0), D4=62=(1,0), Eb4=63=(1,1)
    """
    for pitch in range(128):
        pc = pitch % 12
        octave = (pitch // 12) - 1
        
        # Base X mapping (White key index 0-6)
        if pc in (0, 1): x = 0
        elif pc in (2, 3): x = 1
        elif pc == 4: x = 2
        elif pc in (5, 6): x = 3
        elif pc in (7, 8): x = 4
        elif pc in (9, 10): x = 5
        elif pc == 11: x = 6
        
        # Add octave offset (7 white keys per octave)
        # C++: keyPos.x+=7*(oct-4); NOTE: Check if python needs exact match on octave base
        x += 7 * (octave - 4) 
        
        # Y mapping (0=White/Natural, 1=Black/Accidental)
        # C++: if(pc==0||pc==2||pc==4||pc==5||pc==7||pc==9||pc==11){keyPos.y=0;}
        if pc in (0, 2, 4, 5, 7, 9, 11):
            y = 0
        else:
            y = 1
            
        PITCH_TO_KEYPOS_LUT[pitch, 0] = x
        PITCH_TO_KEYPOS_LUT[pitch, 1] = y

# Initialize on import
_compute_pitch_to_keypos_lut()

# --- Public API ---

def pitch_to_keypos(midi_pitch: int) -> Tuple[int, int]:
    """Python-friendly wrapper for tests/non-critical paths."""
    if not (0 <= midi_pitch < 128):
        raise ValueError(f"Pitch {midi_pitch} out of bounds")
    row = PITCH_TO_KEYPOS_LUT[midi_pitch]
    return int(row[0]), int(row[1])

def subtract_keypos(kp1: Tuple[int,int], kp2: Tuple[int,int]) -> Tuple[int,int]:
    """Python-friendly wrapper for tests."""
    return (kp1[0] - kp2[0], kp1[1] - kp2[1])

# --- Numba Optimized API (Hot Path) ---

@nb.njit(cache=True)
def pitch_to_keypos_numba(midi_pitch: int, lut: np.ndarray) -> np.ndarray:
    """
    Numba-optimized lookup. 
    Usage: pitch_to_keypos_numba(60, PITCH_TO_KEYPOS_LUT)
    Returns array([x, y])
    """
    # Numba implementation...
    return lut[midi_pitch]

@nb.njit(cache=True)
def subtract_keypos_numba(x1, y1, x2, y2):
    return x1 - x2, y1 - y2
```

#### **Step 2.3: Implement Tests (`python/tests/test_lattice.py`)**

You must verify your implementation against the C++ dumps.

```python
import numpy as np
import pytest
import numba as nb
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
        return utils.pitch_to_keypos_numba(60, lut)
        
    res = driver()
    assert res[0] == 0 and res[1] == 0
```

### **Execution Order**

1.  Run the **Phase 1** setup commands.
2.  Compile and Run the **C++ Probes** (Step 2.1).
3.  Write `python/utils.py` (Step 2.2).
4.  Write `python/tests/test_lattice.py` (Step 2.3).
5.  Run tests: `pytest -v python/tests/test_lattice.py`.

**Do not proceed to Phase 3 (HMM Model) until all Lattice tests pass.**

# Phase 3: Parameter Loading & Probability Indexing

**Goal:** Implement a module to parse `param_FHMM2.txt` and `param_FHMM3.txt` into NumPy arrays (Log-Space).
**Critical:** You must replicate the specific indexing formula used in C++ for the output probabilities.

### **3.1 Understand C++ Indexing Logic**
In `FingeringHMM_v180925.hpp`, `outProb` is indexed as follows:
```cpp
// 3 * (dx + widthX) + dy + 1
// where widthX = 15
```
This maps the 2D lattice difference `(dx, dy)` to a 1D array index.

### **3.2 Implement `python/model.py` (Part A: Parameters)**

Create a class `HMMParameters` that loads the file.

**Requirements:**
1.  **Log-Space:** Convert all probabilities to Log-Space immediately upon loading using `np.log()`. Handle strict zeros (if any) by assigning a very small number (e.g., `-1e30`) to avoid `-inf` if preferred, or handle `-inf` correctly.
2.  **Dense Arrays:** Store transitions in dense NumPy arrays.
    *   Order-2: `(2, 5, 5)` for Initial/Trans1, `(2, 5, 5, 5)` for Trans2.
    *   *(Note: The first dimension is usually Hand: 0=Right, 1=Left).*
3.  **Output Probabilities:** Store as `(2, 5, 5, N_OUT)` where `N_OUT` corresponds to the size in the text file.

**Code Snippet (Output Index Helper):**
Add this to `python/utils.py` (and test it!):
```python
@nb.njit(cache=True)
def lattice_delta_to_index(dx: int, dy: int, width_x: int = 15) -> int:
    # C++: 3*(keyInt.x+widthX)+keyInt.y+1
    # We must clamp dx exactly as C++ does:
    if dx < -width_x: dx = -width_x
    if dx > width_x:  dx = width_x
    
    return 3 * (dx + width_x) + dy + 1
```

### **3.3 Verification Strategy (No Mocks)**

You cannot verify this with a simple "it looks right". You must verify against C++ memory.

**Action: Create `param_dump.cpp` probe**
Create `cpp/param_dump.cpp` that includes `FingeringHMM_v180925.hpp`, loads `param_FHMM2.txt`, and dumps specific values to stdout.

```cpp
#include "Code/FingeringHMM_v180925.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if(argc < 2) return 1;
    FingeringHMM_2nd hmm;
    hmm.ReadParamFile(argv[1]); 
    
    std::cout << std::setprecision(10);
    // Dump a specific Transition probability (Right Hand, 1->2->3)
    // Note: Adjust indices based on C++ vector structure [hand][prev][curr]
    std::cout << "TR_R_1_2: " << hmm.trProb[0][0].P[1] << "\n";
    
    // Dump a specific Output probability (Right Hand, 1->2, Delta C4->E4)
    // C4->E4 is dx=1, dy=1. Index = 3*(1+15)+1+1 = 50
    std::cout << "OUT_R_1_2_C4E4: " << hmm.outProb[0][0][1].P[50] << "\n";
    
    return 0;
}
```

**Test `python/tests/test_params.py`:**
1.  Compile and run `param_dump` using `cpp/param_FHMM2.txt`.
2.  Load `cpp/param_FHMM2.txt` using your Python class.
3.  Assert `abs(python_val - cpp_val) < 1e-6`.
4.  **Crucial:** Test the `lattice_delta_to_index` helper in Python against manual calculations to ensure it matches the C++ formula logic.

---

# Phase 4: Data Parsing & Ordering (The Trap)

**Context:** The C++ code reorders notes. If `TimeDepPitchOrder` is not implemented **exactly**, the notes fed into Viterbi will be in a different order than C++, making the transition probabilities meaningless and the result wrong.

### **4.1 Understand `TimeDepPitchOrder`**
Read `PianoFingering_v170101_2.hpp` -> `TimeDepPitchOrder`.
*   Logic: It clusters notes that occur within `0.03s` of each other.
*   Within a cluster, it sorts them by **Pitch Descending** (High to Low).
*   *Note:* Standard MIDI processing often sorts Low to High. You must follow the C++ logic.

### **4.2 Implement `python/utils.py` (Part B)**

Implement `load_pig_file(filepath)` and `apply_time_dep_pitch_order(notes)`.

**Requirements:**
1.  Parse the PIG text format.
2.  Separate hands (Channel 0 vs 1, or Finger > 0 vs < 0).
3.  Implement the clustering and sorting logic.

### **4.3 Verification Strategy (The Dump Comparison)**

**Action: Create `score_dump.cpp` probe**
This probe loads a score file, applies the C++ ordering, and dumps the sequence of `(ontime, pitch)` that is fed into Viterbi.

```cpp
#include "Code/FingeringHMM_v180925.hpp"
#include <iostream>

int main(int argc, char** argv) {
    PianoFingering pf;
    pf.ReadFile(argv[1]);
    pf.SelectHandByFingerNum(0); // Test Right Hand (0)
    pf.TimeDepPitchOrder();      // THE CRITICAL STEP

    for(size_t i=0; i<pf.evts.size(); ++i) {
        std::cout << i << " " << pf.evts[i].ontime << " " << pf.evts[i].pitch << "\n";
    }
    return 0;
}
```

**Test `python/tests/test_parsing.py`:**
1.  Run `score_dump` on `scores/001-1_fingering.txt`.
2.  Parse `scores/001-1_fingering.txt` with Python.
3.  Apply your `apply_time_dep_pitch_order`.
4.  Assert the list of pitches matches the C++ dump exactly. **If this fails, do not proceed to Phase 5.**

---

# Phase 5: Viterbi Implementation (The Engine)

**Goal:** Implement the Viterbi algorithm using Numba.

### **5.1 The Implementation (`python/model.py`)**

You need a generic Viterbi function that accepts:
*   `n_obs`: int
*   `pitches`: array of int
*   `onsets`: array of float
*   `lut`: The KeyPos LUT
*   `init_log_prob`: array
*   `trans_log_prob`: array (2D or 3D depending on order)
*   `out_log_prob`: array
*   `weights`: (w1, w2...)

**Optimization Constraints:**
*   Use `@nb.njit(cache=True)`.
*   Pre-allocate your DP tables (`viterbi_mat`, `backpointer_mat`) as NumPy arrays.
*   Use the `pitch_to_keypos_numba` and `lattice_delta_to_index` helpers inside the loop.
*   **Replica:** You must replicate the specific cost additions found in `FingeringHMM_2nd::Viterbi`:
    *   `shortTimeCost` logic (applied when `abs(t[n] - t[n-1]) < 0.03`).
    *   The `delPitch` check for finger crossing penalties used in C++.

### **5.2 Testing Strategy (End-to-End)**

Now you combine everything.

**Test `python/tests/test_viterbi.py`:**
1.  **Setup:**
    *   Load `param_FHMM2.txt`.
    *   Load `scores/001-1_fingering.txt`.
    *   Separate RH.
    *   Order notes.
2.  **Execution:**
    *   Run Python Viterbi.
3.  **Assertion (The Gold Standard):**
    *   Load `python/tests/ref_outputs/ref_001.txt` (generated in Phase 1).
    *   Extract the fingerings from the reference file.
    *   Compare your Python output fingerings against the reference fingerings.
    *   **Requirement:** They must match **100%**.

### **5.3 Debugging Mismatches**

If the fingerings differ:
1.  Is the input sequence order identical? (Phase 4).
2.  Are the parameters loaded identically? (Phase 3).
3.  **Intermediate Dump:** Modify the Python code to print the max-log-prob at step `N=5`. Modify the C++ code to print `LP[k]` at step `N=5`. Compare.
    *   If they differ, check the `lattice_delta_to_index` logic or the `shortTimeCost` application.

---

# Summary Checklist for Developer

*   [ ] **Phase 3:** `HMMParameters` class written.
*   [ ] **Phase 3:** `lattice_delta_to_index` unit tested against manual math/C++ logic.
*   [ ] **Phase 3:** Parameters loaded in Python match `param_dump.cpp` output.
*   [ ] **Phase 4:** `score_dump.cpp` created and compiled.
*   [ ] **Phase 4:** Python parser + sorting matches `score_dump` output for `001-1`.
*   [ ] **Phase 5:** `viterbi_numba` implemented.
*   [ ] **Phase 5:** End-to-end test compares Python output vs `ref_001.txt`. **Passes with 0 differences.**
