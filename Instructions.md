The `./cpp` folder contains the C++ source code for training and estimation algorithms for 1st-order, 2nd-order, 3rd-order HMMs, and chord HMM, along with an Evaluation script.

**Scope of Work:**
You will focus on porting the following components:
1.  2nd-order HMM
2.  Chord HMM
3.  Evaluation script

The `./scores` folder contains the input `.txt` files for training and estimation.

1. Start by reading and understanding the C++ source code in cpp/README.txt file.
2. Then read the below
3. Then install dependencies (NumPy, Numba, etc. more details below)

**Objective:**
Your goal is to convert the C++ code into Python modules located in the `./python` folder.
The translation must be **functionally equivalent** to the C++ code (producing the same outputs) but **architecturally modernized** to support future features.

**Required Module Structure:**
Please organize the code into a concise set of modules:
1.  `utils.py`: Helper objects (Input/Output, Piano modeling/coordinates, conversions).
2.  `training.py`: Functions for training 2nd-order HMM and Chord HMM.
3.  `inference.py`: Functions for estimation using the trained models.
4.  `evaluate.py`: The evaluation logic.

**Critical Instructions:**

You must balance two goals:
1.  **Functional Fidelity:** The logic (Viterbi algorithm) must behave exactly like the C++ original initially to ensure the port is correct.
2.  **Architectural Foundation:** You must **not** simply write "C++ style Python". You must immediately implement the high-performance data structures required for our future "Hybrid System" (Beam Search + Constraints).

**Detailed Requirements:**

#### 1. HMM Engine & Optimization (Prepare for Future Beam Search)
*   **Current Algorithm:** Implement the standard **Viterbi** algorithm (as found in the C++ code) to ensure baseline correctness.
*   **Future-Proofing:** Structure your decoder function so it is decoupled from the model logic. We will replace Viterbi with a **Beam Search** decoder in the next phase.
*   **Numba Readiness:** Even though you are implementing Viterbi now, write the core decoding loops in a way that is compatible with **Numba** (`@jit(nopython=True)`). Do not use slow Python loops or objects inside the core inference path.
*   **Pairwise Decomposition:** If the C++ code uses specific factorization, preserve it. If not, structure the probability calculation so we can easily inject the pairwise decomposition formula ($P(f_n | f_{n-1})^{\alpha} \cdot P(f_n | f_{n-2})^{\beta}$) later.

#### 2. Runtime Data Structures (High Performance)
Do **not** use `music21` objects or standard Python classes (like `Note` objects) inside the training/inference loops, as they create technical debt for the future system.
*   **Input Processing:** Parse files once, then immediately convert them to **Structured NumPy Arrays**.
*   **Schema:** The matrix passed to your functions should have shape $(N_{notes} \times K_{features})$.
    *   Required Features: `pitch` (int8), `onset` (float32), `duration` (float32), `lattice_x` (float32), `lattice_y` (int8).

#### 3. Lattice Coordinate System (Geometry)
To prepare for future biomechanical constraints, implement the spatial representation now:
*   **Implementation:** Use a static **Look-up Table (LUT)** as a constant array.
*   **Structure:** Array of size 128 (indices 0-127 mapping to MIDI pitches).
*   **Value:** `LUT[midi_pitch] = (x, y)`.
*   **Constraint:** Do not calculate coordinates algorithmically at runtime; use the LUT.

#### 4. Transition Matrix Storage
*   **Format:** Use **Dense Matrices** (NumPy arrays), not Sparse matrices.
*   **Shape:** $(5^N, 5)$. For Order $N=2$, this is $(25, 5)$.
*   **Content:** Store **Log-Probabilities**.


#### 5. Testing & Validation Requirements
We cannot accept code that "looks correct" but produces different results from the reference implementation. You must provide proofs of correctness.

**A. Unit Testing (sanity checks)**
*   Create a `tests` folder.
*   Write unit tests for your `utils` module, specifically verifying:
    *   The MIDI-to-Lattice LUT returns correct `(x,y)` values for known pitches (e.g., check C4, C#4, D4).
    *   The `Input` parser correctly converts a sample `.txt` file into the specified NumPy structured array format.

**B. Functional Equivalence Testing (Python vs C++)**
Since you have the original C++ code and input data:
1.  **Compile and run the C++ code** on a small subset of the data (e.g., the first 5 files in `scores/`).
2.  Save the C++ output (the estimated fingering sequences) as "Ground Truth" reference files.
3.  Run your new **Python Viterbi implementation** on the exact same files.
4.  **Automated Comparison:** Write a script `verify_port.py` that:
    *   Loads the C++ output.
    *   Loads your Python output.
    *   Asserts that the generated fingerings are **identical** (Match Rate = 100%).
    *   *Note:* Small floating-point differences in log-probabilities are acceptable ($1e-6$ tolerance), but the final integer fingering sequence must be identical.

**C. Deliverable Checklist**
Before submitting, verify:
*   [ ] `python/inference.py` runs without errors.
*   [ ] `verify_port.py` confirms 99% match between C++ and Python outputs on a sample dataset.
*   [ ] The code uses NumPy arrays and is compatible with Numba (no Python objects in loops).

# Workflow

1. start with `utils` modules then test it : 

    *About the Column Parsing* :
    *   1st column (index 0) is ID
    *   2nd column (index 1) is Onset
    *   3rd column (index 2) is Offset (not Duration). Calculate `duration = col[2] - col[1]`.
    *   4th column (index 3) is a Note Name (e.g., "Eb4"), not a MIDI integer. Ensure `pitch_name_to_midi` robustly handles sharps (`#`) and flats (`b`).
    *   5th column (index 4) is Note MIDI Pitch. 
    *   6th column (index 5) is Velocity
    *   7th column (index 6) is Hand: Store this as an integer (0 for RH, 1 for LH).
    *   8th column (index 7) is Finger:
        *   Handle substitutions by splitting the string on `_` (e.g., "4_1" becomes 4).
        *   Negative values (e.g., "-5") are for LH only. Handle negatives values by taking the absolute value for the finger number, relying on the 'Hand' column for side distinction.

    *About testing the Parsing* 
    * Discard the synthetic test. Set `TestUtils` to glob and parse every real `.txt` file located in the `./scores` directory. The parser must process the entire actual dataset without raising `ValueError` or `IndexError` to be considered valid.

2. Then, once you are sure your `utils` module is solid, you can move on to the remaining modules.
    *About comparing C++ and Python outputs:*
    * Add well‑placed debugging instructions to both the C++ code and the Python code in order to compare their standard outputs and identify exactly where the divergence occurs, and whether it is caused by floating‑point errors.


