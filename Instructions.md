# Project: High-Performance Python Port of Piano Fingering HMM (Reboot)

## **Philosophy & Golden Rules**
1.  **NO MOCK DATA.** Never verify your logic with handwritten dummy arrays. Always use the real `.txt` score files and the real `param_FHMM*.txt` files provided.
2.  **C++ IS THE TRUTH.** Your goal is not to write "good Python code", but to produce **mathematically identical outputs** to the C++ binary. If the C++ code has a quirk, you reproduce the quirk.
3.  **VALIDATE STEP-BY-STEP.** Do not write the whole Viterbi algorithm before verifying that your data parser produces the exact same array as the C++ parser.

## **Prerequisites**
*   **Data:**
    *   Scores: `./scores/` (e.g., `001_fingering.txt`)
    *   Parameters: `./cpp/param_FHMM2.txt` (and others)
    *   C++ Source: `./cpp/`
*   **Environment:**
    *   Python 3.9+
    *   `numpy`, `numba` (Mandatory for performance)
    *   `g++` (To compile the reference C++ binary)

---

## **Phase 1: Environment & Reference Generation**

**Objective:** Before writing any Python, generate the "Ground Truth" using the C++ code.

**Instructions:**
1.  Compile the C++ code:
    ```bash
    cd cpp
    sh compile.sh
    # Verify that ./Binary/FingeringHMM2_Run exists
    ```
2.  Generate Reference Outputs for **at least 3 different scores** (simple, complex, short):
    *   Run: `./cpp/Binary/FingeringHMM2_Run ./scores/001-1_fingering.txt ./tests/ref_001.txt`
    *   Run: `./cpp/Binary/FingeringHMM2_Run ./scores/002-1_fingering.txt ./tests/ref_002.txt`
    *   Run: `./cpp/Binary/FingeringHMM2_Run ./scores/003-1_fingering.txt ./tests/ref_003.txt`
    *   *Note:* These `ref_*.txt` files are your absolute source of truth for Phase 4.

---

## **Phase 2: Core Math & Lattice Logic**

**Objective:** Replicate the coordinate system logic.

**Instructions:**
1.  Create `python/utils.py`.
2.  **Port `KeyPos` Logic:**
    *   Read `./cpp/KeyPos_v161230.hpp` carefully.
    *   Implement `pitch_to_keypos(midi_pitch)` -> `(x, y)`.
    *   Implement `subtract_keypos(kp1, kp2)` -> `(dx, dy)`.
    *   **CRITICAL:** Pre-calculate a Lookup Table (LUT) `PITCH_TO_KEYPOS_LUT` of shape `(128, 2)` for MIDI pitches 0-127. Using a LUT is faster and safer than calculating on the fly.
3.  **Verification (Unit Test):**
    *   Create `python/tests/test_lattice.py`.
    *   Test specific notes mentioned in C++ comments: `C4 (60) -> (0,0)`, `D4 (62) -> (1,0)`, `Eb4 (63) -> (1,1)`.
    *   Test octave shifts: `C5 (72) -> (7,0)` (Since `7*(oct-4)` logic).

---

## **Phase 3: Data Parsing & Sequence Ordering**

**Objective:** Parse the text files and replicate the specific re-ordering of notes used by the C++ HMM.

**Instructions:**
1.  **Parse PIG Files:**
    *   In `python/utils.py`, implement `parse_pig_file(filepath)`.
    *   It must return a structured NumPy array with fields: `('original_idx', 'ontime', 'offtime', 'pitch', 'channel', 'finger_str')`.
    *   Handle lines starting with `//` or `#` (ignore them).
    *   Convert `sitch` (e.g., "C#4") to MIDI pitch integers.
2.  **Implement Time-Dependent Sorting:**
    *   Read `./cpp/PianoFingering_v170101_2.hpp`, function `TimeDepPitchOrder`.
    *   Implement this logic in Python:
        *   Cluster events where `abs(t_n - t_{n-1}) < 0.03`.
        *   **Inside each cluster, sort by Pitch Ascending.**
    *   **CRITICAL:** Your function must return the sorted events array. The HMM processes this sorted sequence, NOT the original file sequence.
3.  **Verification:**
    *   Create `python/tests/test_parsing.py`.
    *   Parse `scores/001-1_fingering.txt`.
    *   Apply your sorting.
    *   **Debug Trick:** Modify the C++ code (`FingeringHMM_v180925.hpp` inside `Viterbi`) to print the pitch sequence it processes: `cout << testData[i].evts[pos[n]].pitch << endl;`. Recompile. Run on 001. Capture output.
    *   Compare your Python sorted pitch sequence against this C++ debug print. They must match exactly.

---

## **Phase 4: Parameter Loading**

**Objective:** Load the trained HMM matrices into memory.

**Instructions:**
1.  Create `python/model.py`.
2.  Create class `HMMParameters`.
3.  Implement a parser for `param_FHMM2.txt` (Order 2) and `param_FHMM3.txt` (Order 3).
    *   The file structure relies on headers (e.g., `### Transition Prob Right`).
    *   **Log-Space Conversion:** Convert all probabilities to `np.log()` immediately. Use `np.log(val + 1e-300)` or handle `0.0` explicitly to avoid `-inf` if possible (though `-inf` is mathematically correct, it needs care in Viterbi).
    *   **Output Probabilities:** These are sparse/indexed. Store them in a dense array if memory allows, or a Numba-friendly dictionary.
        *   Target structure: A dictionary mapping `(hand_idx, prev_finger, curr_finger)` to a vector of size 93 (`3 * (2*15 + 1)`).
        *   **Assertion:** Verify that every loaded probability vector has length exactly 93.
4.  **Verification:**
    *   Load `cpp/param_FHMM2.txt`.
    *   Check: `initial_prob` shape is `(2, 5)`.
    *   Check: `trans_prob_1` shape is `(2, 5, 5)`.
    *   Check: `trans_prob_2` shape is `(2, 5, 5, 5)`.

---

## **Phase 5: The Numba Viterbi Core**

**Objective:** Implement the decoding algorithm.

**Instructions:**
1.  In `python/model.py`, implement `viterbi_core` decorated with `@numba.jit(nopython=True)`.
2.  **Arguments:** Pass Numpy arrays (events), matrices (params), and scalar weights (`w1`, `w2`, etc.).
3.  **Logic:**
    *   Iterate through the **sorted** events.
    *   Calculate Lattice distances `dx, dy` between notes.
    *   Calculate the lattice index: `idx = 3 * (dx + 15) + dy + 1`. **Hardcode 15** (widthX) for now as it matches the C++ header file, but verify `idx` is within [0, 92].
    *   Implement the **Short Time Cost**: If `abs(t_n - t_{n-1}) < 0.03`, add penalty if finger ordering violates pitch ordering.
    *   **Recursion:**
        *   `Score[n, prev_finger, curr_finger] = max over prev_prev_finger ( ... )`
4.  **Backtracking:**
    *   Store `argmax` pointers in a `(N, 5, 5)` array of `int8`.
    *   Reconstruct the path backwards.

---

## **Phase 6: Integration & Validation**

**Objective:** Prove the port is perfect.

**Instructions:**
1.  Create `python/tests/test_integration.py`.
2.  **The Ultimate Test:**
    *   Loop over the 3 reference files generated in Phase 1 (`ref_001.txt`, etc.).
    *   For each file:
        *   Run your Python `viterbi_core` on the corresponding source score.
        *   Extract the resulting fingerings.
        *   Parse the C++ `ref_*.txt` file to get the C++ fingerings.
        *   **Assert Equality:** `assert python_fingers == cpp_fingers`.
3.  **Performance Check:**
    *   Measure execution time. Processing `001-1_fingering.txt` should take < 0.5 seconds.

---

## **Developer Checklist (Self-Correction)**
*   [ ] Did I create the `ref_*.txt` files using the binary first?
*   [ ] Did I use `numba.jit`?
*   [ ] Did I verify the Lattice Index formula `3*(dx+15)+dy+1`?
*   [ ] Did I handle the `0.03s` clustering exactly like the C++ code?
*   [ ] Did I convert probabilities to Log-Space?
*   [ ] Do my unit tests use the real `./scores/` files?