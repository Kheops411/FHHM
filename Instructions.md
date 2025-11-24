
# Project: High-Performance Python Port of Piano Fingering HMM

## Context & Objective
You are tasked with porting an existing C++ implementation of a High-Order Hidden Markov Model (HMM) for piano fingering estimation into Python.
**Crucial Goal:** The Python version must be **extremely fast** (using NumPy/Numba) and must produce **mathematically identical results** to the C++ reference when using the same parameter files.

## Project Structure
Create the following file structure:
```
/
├── cpp/                  # Provided C++ source and parameter files
├── scores/               # Provided PIG dataset (.txt files)
├── python/
│   ├── utils.py          # Core math, Parsing, Data structures
│   ├── model.py          # HMM logic, Viterbi algorithm (Numba optimized)
│   ├── training.py       # (Placeholder for Phase 2)
│   ├── evaluate.py       # Comparison tools
│   └── tests/            # Unit tests
```

---


## Phase 0: Understanding the reference C++ code

**Goal:** Get a solid understanding of the reference C++ code and its design.

Instruction : read the `./cpp/README.txt` file


## Phase 1: Core Utilities & Lattice Representation

**Goal:** Replicate the specific geometric representation of the piano keyboard used in the C++ code.

### Step 1.1: Port `KeyPos` Logic (`utils.py`)
**Reference:** `./cpp/KeyPos_v161230.hpp`

1.  Create a `pitch_to_keypos(midi_pitch)` function.
    *   It must return a tuple `(x, y)` or a NumPy array.
    *   **Logic:** Implement the logic inside `PitchToKeyPos` exactly. Note that `x` is base-7 relative to octaves, and `y` distinguishes black/white keys within the cluster.
    *   **Lattice Width:** The C++ code defines `widthX=15`. Ensure your logic handles the coordinate shift centered around 0.
2.  Create a `subtract_keypos(kp1, kp2)` function.
    *   Replicate `SubtrKeyPos`.
3.  **Optimization:** Since MIDI pitches are integers 0-127, pre-calculate a **Look-Up Table (LUT)** (a NumPy array of shape `(128, 2)`) initialized at import time. Do not calculate `x, y` on the fly during Viterbi.

### Step 1.2: Test KeyPos
**Action:** Create `tests/test_utils.py`.
*   **Test Case:** Compare specific MIDI notes (e.g., C4=60, C#4=61, B4=71) against manual calculations based on the C++ code comments (`C4=(0,0), D4=(1,0), Eb4=(1,1)`).
*   **Verify:** Ensure `subtract_keypos` works correctly for large intervals (wrapping/clamping logic is handled in the HMM, but the raw subtraction must be correct).

---

## Phase 2: Data Parsing & Sorting

**Goal:** Parse PIG files and replicate the specific note ordering logic used in C++.

### Step 2.1: PIG Parser (`utils.py`)
**Reference:** `./cpp/PianoFingering_v170101_2.hpp` (Function: `ReadFile`)

1.  Write a function to read a `.txt` score file.
2.  Extract: ID, ontime, offtime, pitch, channel (hand), and fingerNum.
3.  **Important:** The parser must handle the header lines (`//Version...`) and comments (`#...`) gracefully.

### Step 2.2: Time-Dependent Sorting
**Reference:** `./cpp/PianoFingering_v170101_2.hpp` (Function: `TimeDepPitchOrder`)

The C++ code reorders notes based on simultaneity. This changes the sequence fed into the HMM. You **must** replicate this exactly.
1.  **Cluster:** Group notes where `abs(t_n - t_{n-1}) < 0.03` seconds.
2.  **Sort:** Inside a cluster, sort notes by **Pitch Ascending** (Low to High).
    *   *Note:* The C++ code uses `MorePair` with negative pitch values, which results in ascending order.
3.  **Data Structure:** Use a structured NumPy array or a clean dataclass list. Do NOT use `music21` objects (too slow).
    *   Required fields: `pitch`, `onset`, `channel`, `finger_int` (mapped -5 to 5), `original_index`.

### Step 2.3: Test Parsing
**Action:** Create `tests/test_parsing.py`.
1.  Compile and run the C++ binary `./Binary/FingeringHMM2_Run` (or similar) on a specific score (e.g., `scores/001_fingering.txt`) without arguments or in a debug mode that prints the processed event list.
2.  Run your Python parser on `scores/001_fingering.txt`.
3.  **Compare:** Ensure the sequence of pitches in your Python list matches the C++ processed sequence exactly. Any deviation here will cause the Viterbi path to diverge.

---

## Phase 3: Model & Parameters

**Goal:** Load the trained C++ parameters into efficient NumPy structures.

### Step 3.1: Parameter Loader (`model.py`)
**Reference:** `./cpp/FingeringHMM_v180925.hpp` (Function: `ReadParamFile`) and example file `./cpp/param_FHMM2.txt`.

1.  Create a `HMMParameters` class.
2.  Write a parser for the specific custom format of `param_FHMM*.txt`:
    *   It uses headers like `### Initial Prob Right`.
    *   It stores data in tab/space-separated values.
    *   **Transformation:** The C++ file stores **Probabilities**. Convert them to **Log-Probabilities** (`np.log()`) immediately upon loading.
        *   *Warning:* Handle `log(0)` by using a very small number (e.g., -100 or -inf) to avoid NaNs.
3.  **Matrices Dimensions (Order 3 example):**
    *   `initial_prob`: `(2, 5)`  (Hand x Finger)
    *   `trans_prob_1`: `(2, 5, 5)` (Hand x Prev x Curr)
    *   `trans_prob_2`: `(2, 5, 5, 5)` (Hand x Prev2 x Prev1 x Curr)
    *   `trans_prob_3`: `(2, 5, 5, 5, 5)` (Hand x Prev3 x Prev2 x Prev1 x Curr)
    *   `output_prob`: `(2, 5, 5, LatticeSize)` where `LatticeSize = 3 * (2*15 + 1)`.
    *   *Note:* Ensure you map the lattice index correctly: `3*(x + 15) + y + 1`.

### Step 3.2: Test Parameter Loading
**Action:** Create `tests/test_model_loading.py`.
1.  Load `./cpp/param_FHMM2.txt` in Python.
2.  Manually inspect the file and your array. Check specific values (e.g., the first value of `### Initial Prob Right` vs `model.initial_prob[0, 0]`).
3.  Assert that arrays are not empty and have the correct shapes.

---

## Phase 4: High-Performance Viterbi (`model.py`)

**Goal:** Implement the Viterbi decoding. This is the bottleneck.

### Step 4.1: Implementation Strategy
**Reference:** `./cpp/FingeringHMM_v180925.hpp` (Function: `Viterbi` inside `FingeringHMM_2nd` and `FingeringHMM_3rd`)

1.  **Use Numba:** You must use `@numba.jit(nopython=True)` for the core Viterbi loop. Pure Python loops will be 100x slower.
2.  **Logic:**
    *   Implement the HMM order 3 logic (which covers order 2 if coefficients are set to 0).
    *   **State Space:** For Order 3, the state at step `n` depends on `n-1` and `n-2`.
    *   **Indices:** Be very careful with 0-based indexing (Python) vs. the C++ logic. The C++ code maps fingers 1..5 to indices 0..4.
    *   **Hand Separation:** The algorithm processes Right Hand (Channel 0) and Left Hand (Channel 1) independently.

### Step 4.2: Symmetry & Lattice Mapping
Inside the Viterbi loop:
1.  Calculate `delta_pitch` and convert to `KeyPos` (Lattice x, y).
2.  Apply the lattice index formula: `idx = 3 * (keyInt.x + 15) + keyInt.y + 1`.
3.  **Short Time Cost:** Implement the `shortTimeCost` penalty for notes played very quickly (chords) if they violate finger ordering (e.g., lower finger playing higher pitch). See C++: `bool shortTime = abs(...) < 0.03`.

### Step 4.3: Test Viterbi
**Action:** Create `tests/test_viterbi.py`.
1.  **Gold Standard Generation:**
    *   Compile the C++ code: `./compile.sh` (ensure you have `g++`).
    *   Run `./Binary/FingeringHMM2_Run ./scores/001_fingering.txt ./cpp_output_001.txt`.
2.  **Python Execution:**
    *   Run your Python model on `scores/001_fingering.txt` using `param_FHMM2.txt`.
3.  **Comparison:**
    *   Compare the fingering column of your Python output against `./cpp_output_001.txt`.
    *   **Success Criterion:** 100% match on the fingering sequence.

---

## Summary of deliverables for the Developer

1.  **`utils.py`**:
    *   `PitchToKeyPos` (Lattice logic)
    *   `TimeDepPitchOrder` (Sorting logic)
    *   `PIGParser`
2.  **`model.py`**:
    *   `HMMParams` loader (Handles `param_FHMM*.txt`)
    *   `viterbi_core` (Numba optimized function)
3.  **Tests**:
    *   Unit tests for KeyPos.
    *   Unit tests for Sorting (vs C++ logic).
    *   Integration test comparing Python output vs C++ binary output on real files.

**Important Note:** Do not proceed to training or evaluation scripts until the inference (Viterbi) produces identical results to the C++ binary on the provided score files. Performance should be < 1s per song.