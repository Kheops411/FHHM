
# **PROJECT SPECIFICATION: High-Performance Piano Fingering Engine**

**Role:** Lead Backend Developer
**Goal:** Port a legacy Python fingering engine to a high-performance **NumPy/Numba** architecture.
**Paradigm:** Structure of Arrays (SoA). Strict Physical Modeling. No "Patching".
**Language:** Python 3.9+

---

## **0. PROJECT STRUCTURE**

You must respect this exact file structure.

```text
/project_root
│
├── legacy/                  # [READ-ONLY] Reference implementation provided to you
│   ├── engine.py
│   ├── utils.py
│   └── xml_parser.py
│
├── resources/               # [PROVIDED] Test fixtures (MusicXML files)
│   ├── scale_c_major.xml
│   ├── chord_c_major.xml
│   ├── silence_gap.xml
│   ├── ties_no_gap.xml
│   ├── chromatic_black_keys.xml
│   └── # ... and so on ...
├── src/                     # [WRITE] Your code goes here
│   ├── __init__.py
│   ├── utils.py             # Physics constants & LUT generation
│   ├── xml_parser.py        # Validated parser + Normalization logic
│   ├── engine.py            # Preprocessing + Numba Viterbi Core
│   └── eval.py              # Comparison metrics
│
└── tests/                   # [WRITE] Unit tests
    ├── test_utils.py
    ├── test_parser.py
    ├── test_segmentation.py
    └── test_viterbi.py
```

---

## **1. CONTEXT & PHYSICS (Crucial)**

You are modeling a **human hand** on a **physical piano**. Logic must follow physics.

1.  **The Keyboard:**
    *   Standard width of a white key: **~2.35 cm**.
    *   Octave span (12 semitones): **16.5 cm**.
    *   Black keys are narrower and raised.
2.  **The Hand:**
    *   **5 Fingers:** Indexed internally as `0` (Thumb) to `4` (Pinky).
    *   **Constraints:**
        *   **Max Span:** A hand cannot span 50cm. Max is approx an octave + 2 tones.
        *   **Polyphony:** You cannot use the same finger for two simultaneous notes.
        *   **Crossings:** Only the Thumb (0) acts as a pivot. Finger 2 cannot cross over Finger 3.

---

## **2. IMPLEMENTATION PHASES**

**Rule:** Do not move to Phase N+1 until Phase N tests pass.

### **Phase 1: Foundations (`src/utils.py`)**

**Objective:** Establish physical ground truth using Lookup Tables (LUTs).

1.  **Define Constants:**
    *   `KEYBOARD_SIZE_CM = 16.5`
    *   `K_STEP = KEYBOARD_SIZE_CM / 7.0`
    *   `EPSILON_CHORD = 0.05` (seconds).
    *   `GAP_THRESHOLD = 0.5` (seconds).
2.  **Implement `generate_luts()`:**
    *   Port the exact logic from `legacy/utils.py` (do not reinvent the geometry).
    *   **Return:** Two Numpy arrays.
        *   `keypos_lut`: `np.float64[128]`. X position (cm) for every MIDI pitch.
        *   `is_black_lut`: `np.int8[128]`. `1` if black key, `0` if white.

**Mandatory Test (`tests/test_utils.py`):**
*   Call `generate_luts`.
*   **Assert:** `len(keypos_lut) == 128`.
*   **Assert:** `keypos_lut[61] - keypos_lut[60]` is approx `1.17` (C4 to C#4 distance).
*   **Assert:** `keypos_lut[77] - keypos_lut[65]` is **exactly** `16.5` (F4 to F5 distance).

---

### **Phase 2: Parser & Normalization (`src/xml_parser.py`)**

**Objective:** Ensure clean input data.

1.  Copy the validated `xml_parser.py` (provided separately) into `src/`.
2.  **Do not modify the parser logic.**
3.  **Implement `get_valid_mask(score)`:**
    *   Return a boolean array: `mask = score.pitch > 0`.
    *   **Reasoning:** We do not delete silences or change pitch. We filter them out using this mask.

**Mandatory Test (`tests/test_parser.py`):**
*   Load `resources/silence_gap.xml`.
*   **Assert:** `score.pitch` contains `0` values (silences).
*   **Assert:** `get_valid_mask(score)` returns `False` at indices where pitch is `0`.

---

### **Phase 3: Preprocessing & Segmentation (`src/engine.py` - Part A)**

**Objective:** Prepare vectors for the Numba core.

**Implement `preprocess_data(score)`:**
1.  **Filter:** Apply `valid_mask` to `pitch`, `onset`, `duration`, `hand`.
2.  **Compute `x_pos`:**
    *   Map filtered pitches using `keypos_lut`.
    *   **Crucial:** If `hand == LEFT`, apply `x_pos = -x_pos`.
3.  **Compute `chord_id`:**
    *   Group notes where `onset[i] - onset[i-1] <= EPSILON_CHORD`.
    *   Use `np.int32` for IDs.
4.  **Segmentation (Silence as Reset):**
    *   Calculate `offset = onset + duration`.
    *   Identify gaps: `gap = onset[i] - offset[i-1]`. (Note: use **offset** of previous note, not onset).
    *   Split data into segments where `gap > GAP_THRESHOLD`.

**Mandatory Test (`tests/test_segmentation.py`):**
1.  **Case Tie:** Load `resources/ties_no_gap.xml`.
    *   **Assert:** Returns **1 segment**. (The tied note bridges the gap).
2.  **Case Silence:** Load `resources/silence_gap.xml`.
    *   **Assert:** Returns **2 segments**. (The rest causes a reset).
3.  **Case Chord:** Load `resources/chord_c_major.xml`.
    *   **Assert:** All notes have the same `chord_id`.

---

### **Phase 4: The Core Engine (`src/engine.py` - Part B)**

**Objective:** High-performance calculation using Numba.

**Constraint:** You must use **Combinatorial States** for chords. Do not process chords sequentially.

**Implement `core_viterbi`:**
*   **Decorator:** `@numba.njit(cache=True)`
*   **Signature:**
    ```python
    def core_viterbi(
        x_pos: np.float64[:],
        duration: np.float64[:],
        chord_id: np.int32[:],
        is_black: np.int8[:],
        hand_factor: float
    ) -> np.int8[:]
    ```
*   **Logic:**
    1.  Iterate through unique `chord_id`s (Time Steps).
    2.  **State Generation:**
        *   If Single Note: 5 states (Fingers 0-4).
        *   If Chord (N notes): Generate valid **permutations** of N fingers.
        *   *Validation:* Eliminate states with crossing fingers inside the chord.
    3.  **Transition Cost:**
        *   Compute physical distance moved by each finger.
        *   Apply penalties (Black key usage by thumb, large stretches).
        *   **Impossible Transitions:** Cost = Infinity (e.g., using a finger still held down by a previous long note).
    4.  **Backtracking:** Return the optimal path (0-4).

**Mandatory Test (`tests/test_viterbi.py`) - Using provided resources:**

1.  **Scale Test:** Load `resources/scale_c_major.xml`.
    *   **Assert:** The output array is valid (values 0-4).
    *   **Assert:** A thumb cross occurs. (Check indices where `x_pos` increases but `finger` index decreases, e.g., 2 -> 0).
    *   *Do not assert strict 1-2-3-1-2-3-4-5 sequence, just physical validity.*

2.  **Chord Test:** Load `resources/chord_c_major.xml`.
    *   **Assert:** The output corresponding to the chord contains **unique** values.
    *   Example: `[0, 2, 4]` is valid. `[0, 0, 4]` is **FAIL**.

3.  **Black Key Test:** Load `resources/chromatic_black_keys.xml`.
    *   **Assert:** Finger `0` (Thumb) is **not** used for these black keys.

---

### **Phase 5: Evaluation (`src/eval.py`)**

**Objective:** Compare results with Ground Truth.

1.  Loop through segments.
2.  Run `core_viterbi`.
3.  Reconstruct the full `finger_out` array using `valid_mask`.
4.  **Output:** Convert internal `0..4` to external `1..5`.
5.  Compare `finger_out` vs `score.finger_gt` (accuracy).

---

## **3. CRITICAL RULES (DO NOT IGNORE)**

1.  **No Patching:** Do not hardcode results to make tests pass. If `test_viterbi.py` fails, your cost function or state generation is mathematically wrong. Fix the math.
2.  **Strict Types:** Use `np.float64` for time/position and `np.int8`/`np.int32` for discrete values. Numba will crash if types are mixed.
3.  **Debug Prints:** Numba is hard to debug. You may use `objmode` or return the cost matrix to inspect values during development, but clean up for production.
4.  **Legacy Code:** Do not import `legacy.engine`. You are rewriting it, not wrapping it. Use `legacy` only for reading logic.

