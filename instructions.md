# Introduction

The program in `soft_position_hmm` implements a probabilistic model (2nd‑order HMM) to predict or learn piano fingering from sequences of MIDI notes.

**Summary of roles:**

* **core.py**: defines the basic biomechanical model.  
  * Numba functions to compute:  
    * an emission score (finger ↔ pitch interval compatibility),  
    * an inertia cost (difficulty of hand movement).  
  * Model parameters (RBF for each finger, temporal inertia).

* **structural.py**: defines the structure of the Viterbi lattice (log‑probabilities, backpointers) for optimal path search in the HMM.

* **utils.py**: utility functions.  
  * Conversion MIDI pitch → keyboard position.  
  * Strict parsing of annotated files (PIG).  
  * Temporal sorting of notes and filtering by hand.

* **inference.py**: core of the HMM algorithm.  
  * Forward pass (constrained or free) that fills the lattice by combining emission, inertia, transitions, and smoothing.  
  * Backtracking to extract the optimal sequence of fingers and anchors.

* **training.py**: EM loop for parameter learning.  
  * E‑step: optimal anchoring constrained by true fingers.  
  * M‑step: update of RBF parameters and agility matrix (transition probabilities between fingers).

* **interface.py**: high‑level user API.  
  * Takes a sequence of notes and produces predictive fingering by calling forward pass + backtracking.

**Overview of operation:**

1. Notes are read, cleaned, ordered, and filtered by hand.  
2. The model evaluates for each note the likelihood of using a given finger and hand position.  
3. The HMM explores all possible combinations and selects the most probable trajectory (Viterbi).  
4. In training mode, this trajectory is used to re‑estimate parameters (EM).  
5. In prediction mode, the optimal finger sequence is retrieved.  

---  


# Technical Specification — Soft-Position HMM

## 1. Model and Lattice Structure

The model is a 2nd-order HMM for fingers and a 1st-order HMM for anchors.
For each time step ($t$), the lattice contains all combinations of:

*   current finger
    $$ f_t \in \{0,1,2,3,4\}, $$
*   previous finger
    $$ f_{t-1} \in \{0,1,2,3,4\}, $$
*   current anchor
    $$ k_t \in \{0,\dots,24\}. $$

The lattice dimensions are:
$$ T \times 5 \times 5 \times 25. $$

Transitions also require ($f_{t-2}$) and the previous anchor ($k_{t-1}$).
The exact lattice structure is defined in `structural.py` (log-probability matrices, backpointers, storage of two successive fingers, etc.).

---

## 2. Emission Model

**Module:** `core.compute_emission_score`

For a time step ($t$), emission depends on the finger ($f_t$) and the anchor ($k_t$), applied to the MIDI pitch note ($p_t$).

### 2.1 Definition of Delta ($\delta_t$)

The code defines:
$$ \delta_t = -\mathrm{ANCHORS}[k_t], $$
where `ANCHORS` is a constant array of 25 integers covering the range $[-12,+12]$ semitones.

### 2.2 Biomechanical Parameters (per finger)

Each finger ($f$) possesses:

*   a mean ($\mu_f$),
*   a standard deviation ($\sigma_f$).

These parameters are updated during training.

### 2.3 Standard Deviation Clamp (Inference)

In inference, the code forces:
$$ \hat\sigma_f = \max(\sigma_f, 1.0). $$

### 2.4 Exact Formula

The emission score is:
$$ \mathrm{emit}(t,f_t,k_t) = -\log(\hat\sigma_{f_t}) - \tfrac12\log(2\pi) - \frac{(\delta_t - \mu_{f_t})^2}{2\hat\sigma_{f_t}^2}. $$

---

## 3. Transition Model

The transition score between $(f_{t-2},f_{t-1},k_{t-1})$ and $(f_{t-1},f_t,k_t)$ is:

$$ \mathrm{trans} = \mathrm{agility} - \mathrm{inertia} - \mathrm{smoothing}. $$

All components below correspond exactly to the provided files.

---

### 3.1 Agility: Finger Trigram

The model contains a tensor:
$$ A \in \mathbb{R}^{5\times 5\times 5}, $$
storing the **log-probabilities**:

$$ \mathrm{agility} = A[f_{t-2}, f_{t-1}, f_t]. $$

If untrained, `A` = 0 for all entries (uniform prior in log-space).

---

### 3.2 Inertia: Hand Center Displacement Cost

**Modules:** `core.compute_inertia_cost` and usage in `inference.py`

The formula used in inference is **strictly scalar in semitones**; no conversion to 2D keyboard coordinates is performed.

#### 3.2.1 Definition of Hand Center

For a MIDI pitch note ($p_t$) and an anchor ($k_t$), the hand center is:
$$ \mathrm{center}_t = p_t + \mathrm{ANCHORS}[k_t]. $$
Both terms are integers (semitones).

#### 3.2.2 Physical Distance Used

The distance used in inertia is:
$$ d_{\text{phys}} = |\mathrm{center}_t - \mathrm{center}_{t-1}|. $$

It is a **1D distance in semitones**, in accordance with the code:

```python
hand_pos_prev = notes_pitch[t-1] + ANCHORS[k_prev]
hand_pos_curr = notes_pitch[t]   + ANCHORS[k_curr]
dist = np.abs(hand_pos_curr - hand_pos_prev)
```

#### 3.2.3 Simultaneity Constraint (dt < 1e-4)

If notes are quasi-simultaneous:

*   If ($d_{\text{phys}} > 10^{-4}$):
    $$ \mathrm{inertia} = +\infty. $$
*   Otherwise:
    $$ \mathrm{inertia} = 0. $$

#### 3.2.3 Standard Case (dt ≥ 1e-4)

The code applies a sigmoidal stiffness factor:

$$ \lambda(dt) = \frac{1}{1 + \exp(\mathrm{slope}\cdot(dt - \mathrm{center}))}. $$

Final cost:
$$ \mathrm{inertia} = \lambda(dt) \cdot d_{\text{phys}} \cdot w_{\text{inertia}}. $$

---

### 3.3 Smoothing: Anchor Change

Smoothing penalizes anchor variation between two notes:

$$ \mathrm{smoothing} = |\mathrm{ANCHORS}[k_t] - \mathrm{ANCHORS}[k_{t-1}]|\cdot w_{\mathrm{smooth}}. $$

---

## 4. Training (Hard EM — Viterbi Training)

**Module:** `training.py`

Learning is performed entirely in log-probability space, with updates to biomechanical parameters and the agility tensor.

---

### 4.1 E-Step: Constrained Inference (Anchors Only)

The actual fingering provided in the data is imposed:

*   Fingers ($f_t$) are *fixed* to the ground truth.
*   Only anchors ($k_t$) are optimized.
*   The modified forward pass is implemented in: `run_constrained_forward_pass`.

---

### 4.2 M-Step: Parameter Updates

#### A. Emission Parameters (RBF per finger)

From the observed $\delta_t$ (defined by $-\mathrm{ANCHORS}[k_t]$):

$$ \mu_f \leftarrow \text{empirical mean}, \qquad \sigma_f \leftarrow \max(\text{empirical std dev}, 0.5). $$

The **0.5** threshold is specific to training (different from the 1.0 threshold used in inference).

#### B. Agility Tensor (Trigram)

1.  Counting transitions:
    $$ C(f_{t-2},f_{t-1},f_t). $$

2.  Normalization for each pair $(f_{t-2},f_{t-1})$:
    $$ P = C / \sum_{f_t} C. $$

3.  Empty rows:
    if $\sum_{f_t} C = 0$, then
    $$ P(f_t) = 1/5. $$

4.  Conversion to log-space:
    $$ A = \log(P + 10^{-12}). $$

---

## 5. Input Data Pre-processing

**Module:** `utils.py`

### 5.1 PIG Parsing

Files contain exactly 8 columns.
Fingers can be written as strings such as `"4_1"`; the code correctly extracts the main integer.

### 5.2 Hand Separation

*   Right Hand: finger > 0
*   Left Hand: finger < 0

Lines where finger is 0 or invalid are rejected.

### 5.3 Time Sorting and Simultaneity Handling

1.  Primary sort by onset.
2.  Simultaneity cluster detection:
    two notes belong to the same cluster if the onset difference is < 0.03 s.
3.  Secondary sort within each cluster by ascending pitch.

The internal cluster order directly influences the correct application of the simultaneity inertia constraint.

---

## 6. Model Result

Inference (module `interface.py`):

1.  constructs the full lattice ($T \times 5 \times 5 \times 25$),
2.  fills scores via emission + transition accumulation,
3.  performs backtracking (defined in `structural.py`),
4.  returns the optimal sequence of fingers and anchors (or only fingers if requested).



# Instructions

**SUBJECT: STRICT PROTOCOL FOR REFACTORING AND DEBUGGING PIANO FINGERING HMM**

You are tasked with refactoring a Python-based Hidden Markov Model (HMM) system. The current codebase contains critical mathematical and logical flaws.

**STRICT OPERATIONAL RULES:**
1.  **Do NOT rely on your intuition.** You do not know the domain (music/piano). Follow the mathematical instructions exactly.
2.  **NO "Patching".** Do not hardcode values to satisfy a test case (e.g., `if value == 5: return expected_result`). You must fix the underlying logic/equation.
3.  **NO "Magic Numbers" inside functions.** Define constants at the top of the module or in the class `__init__`.
4.  **NO `if/else` spaghetti for math.** If I ask for a matrix initialization, do not write a loop with conditionals. Use Numpy vectorization.
5.  **MANDATORY LOGGING:** Before *any* action (editing a file, running a script, even fixing a typo), you must append an entry to a file named `CHANGELOG_DEV.md` at the project root.
    *   Format: `[TIMESTAMP] [FILE] [ACTION] [OBSERVATION]`
    *   Example: `[10:05] core.py - Added clip() to inertia - Result: Integers are now bounded.`
6.  **EXECUTION:** For every task, you must create a dedicated test script (`test_task_N.py`), run it, and paste the output into `CHANGELOG_DEV.md`. If the script fails, you do not proceed.

---

### TASK 0: SETUP
1.  Create `CHANGELOG_DEV.md` at the root.
2.  Log the start of the session.

---

### TASK 1: FIX AGILITY MATRIX INITIALIZATION
**Context:** Currently, `agility_matrix` is initialized to zeros. In log-space, `log(0) = -inf`, which breaks the HMM (all probabilities become zero).
**Target File:** `training.py`

**Instructions:**
1.  Locate `SoftPositionTrainer.__init__`.
2.  Replace `np.zeros(...)` with a uniform distribution: `np.full((5, 5, 5), 1.0 / 125, dtype=np.float64)`.
3.  In `_update_agility_parameters`:
    *   Before calculating probabilities, apply Laplace Smoothing: add `1e-3` to `counts` to ensure no zero-counts exist.
    *   Calculate `log_probs`. Ensure you use `np.log(probs + 1e-12)` (epsilon) as a safety net.

**Validation Script (`test_task_1.py`):**
```python
import numpy as np
from soft_position_hmm.training import SoftPositionTrainer

trainer = SoftPositionTrainer()
# Check initialization
print(f"Init Mean: {np.mean(trainer.agility_matrix)}")
assert np.all(trainer.agility_matrix > 0), "Matrix must be positive before log"

# Check Update Logic
counts = np.zeros((5,5,5)) # Empty counts
trainer._update_agility_parameters(counts)
print(f"Log Agility Max: {np.max(trainer.agility_matrix)}")
print(f"Log Agility Min: {np.min(trainer.agility_matrix)}")

# FAILURE CONDITION: If min is -inf or nan.
if not np.isfinite(trainer.agility_matrix).all():
    raise ValueError("Agility matrix contains Inf or NaN!")
print("TASK 1 SUCCESS")
```

---

### TASK 2: FIX CHORD HANDLING AND TIMING
**Context:** The condition `dt < 1e-4` is too strict for real-world data. It causes "teleportation" errors where infinite cost is applied to chords.
**Target File:** `core.py` (logic) and `inference.py` (implementation)

**Instructions:**
1.  In `core.py`, modify `compute_inertia_cost`:
    *   Change the logic to: **If `dt` is less than `0.03` (30ms), the inertia cost is `0.0`**, regardless of physical distance.
    *   Do **NOT** simply change the `1e-4` threshold. You must add the explicit check: `if dt < 0.03: return 0.0`.
    *   Remove the `return np.inf` logic for small time steps. Inertia should simply be zero for chords.

**Validation Script (`test_task_2.py`):**
```python
from soft_position_hmm.core import compute_inertia_cost

# Scenario: Two notes in a chord (dt=0.001), distance is 5 semitones
cost_chord = compute_inertia_cost(physical_distance=5.0, dt=0.001, slope=10.0, center=0.2, weight=1.0)
print(f"Cost Chord: {cost_chord}")

# Scenario: Fast scale (dt=0.1), distance 5
cost_scale = compute_inertia_cost(physical_distance=5.0, dt=0.1, slope=10.0, center=0.2, weight=1.0)
print(f"Cost Scale: {cost_scale}")

if cost_chord != 0.0:
    raise ValueError("Chords (dt < 0.03) must have 0 inertia cost.")
if cost_scale == 0.0:
    raise ValueError("Scales must have non-zero inertia.")
print("TASK 2 SUCCESS")
```

---

### TASK 3: PARSING ROBUSTNESS
**Context:** Currently, if a finger annotation is invalid/unknown, the parser returns `0` or drops the note. This corrupts the rhythm.
**Target File:** `utils.py` and `inference.py`

**Instructions:**
1.  In `utils.py`:
    *   Define a global constant: `FINGER_UNKNOWN = -999`.
    *   Modify `clean_finger_str`: If parsing fails or input is invalid, return `FINGER_UNKNOWN` instead of `0`.
    *   Modify `filter_notes_by_hand`: **Remove this filtering logic entirely.** Do not filter notes based on finger values. We need ALL notes to maintain correct `dt`.
2.  In `inference.py`, inside `run_constrained_forward_pass`:
    *   Locate where `f_curr` and `f_prev` are retrieved from `true_fingers`.
    *   Add a condition: If `true_fingers[t] == -999`, do **NOT** force the path. Allow the loop to explore all `N_FINGERS` (treat it as an unconstrained hidden state for that step).
    *   *Implementation Hint:* You likely need an `if/else` block: `if true_fingers[t] != -999: # enforce constraint ... else: # standard forward pass logic ...`

**Validation Script (`test_task_3.py`):**
```python
import numpy as np
from soft_position_hmm.utils import clean_finger_str, FINGER_UNKNOWN, filter_notes_by_hand
from soft_position_hmm.structural import NOTE_DTYPE

# 1. Test Parser
res = clean_finger_str("invalid")
if res != FINGER_UNKNOWN:
    raise ValueError(f"Expected {FINGER_UNKNOWN}, got {res}")

# 2. Test Filter
# Create a dummy note array with an unknown finger
dummy_notes = np.zeros(1, dtype=NOTE_DTYPE)
dummy_notes[0]['finger'] = FINGER_UNKNOWN
filtered = filter_notes_by_hand(dummy_notes, 0)

if len(filtered) == 0:
    raise ValueError("filter_notes_by_hand deleted the unknown note! It must persist.")
print("TASK 3 SUCCESS")
```

---

### TASK 4: GEOMETRY UPDATE (EUCLIDEAN DISTANCE)
**Context:** The system uses `delta_pitch` (1D semitones) for inertia. It needs 2D physical distance (keys).
**Target File:** `inference.py`

**Instructions:**
1.  Import `PITCH_TO_KEYPOS_LUT` from `.utils`.
2.  In `run_forward_pass` and `run_constrained_forward_pass`:
    *   Locate the inertia calculation loop.
    *   Currently, it calculates `dist = abs(hand_pos_curr - hand_pos_prev)`. **This is wrong.**
    *   You must calculate the **2D Euclidean distance**:
        *   Get coordinates for `hand_pos_prev` (which is a pitch value) using the LUT.
        *   Get coordinates for `hand_pos_curr` using the LUT.
        *   `dx = x2 - x1`, `dy = y2 - y1`
        *   `dist = sqrt(dx*dx + dy*dy)`
    *   Pass *this* `dist` to `compute_inertia_cost`.
    *   *Note:* Ensure you handle array bounds if `hand_pos` falls outside 0-127 (clamp it if necessary before looking up in LUT).

**Validation Script (`test_task_4.py`):**
*No script provided. Run the existing `inference.py` logic. If it crashes with "IndexError", you forgot to clamp the pitch before LUT lookup. Log the fix.*

---

### TASK 5: INERTIA CLIPPING
**Context:** Large jumps cause infinite costs which break the math (NaNs).
**Target File:** `core.py`

**Instructions:**
1.  In `compute_inertia_cost`:
    *   After calculating `cost`, apply a clamp/min function.
    *   `cost = min(cost, 8.0)`
    *   This ensures that even impossible jumps have a finite probability (exp(-8) is small but not zero).

**Validation Script (`test_task_5.py`):**
```python
from soft_position_hmm.core import compute_inertia_cost
# Scenario: Impossible jump (distance 100), short time
cost = compute_inertia_cost(100.0, 0.1, 10.0, 0.2, 1.0)
print(f"Capped Cost: {cost}")
if cost > 8.0001:
    raise ValueError("Inertia cost was not capped!")
print("TASK 5 SUCCESS")
```

---

### TASK 6: EM STABILITY
**Context:** Training fails if `sigma` becomes 0 or if parameters oscillate.
**Target File:** `training.py`

**Instructions:**
1.  In `_update_emission_parameters`:
    *   Replace `self.model.rbf_sigma[i] = max(0.5, new_sigma)` with:
        `self.model.rbf_sigma[i] = max(0.3, new_sigma)` (Allow slightly tighter variance).
    *   Implement **Momentum** for `mu` updates. Do not simply overwrite `self.model.rbf_mu`.
        *   Formula: `new_mu = 0.9 * self.model.rbf_mu[i] + 0.1 * computed_mean`
        *   Then apply: `self.model.rbf_mu[i] = np.clip(new_mu, -12, 12)`

**Validation Script (`test_task_6.py`):**
```python
import numpy as np
from soft_position_hmm.training import SoftPositionTrainer

trainer = SoftPositionTrainer()
trainer.model.rbf_mu[0] = 5.0
# Simulate data that would pull mu to -5.0
deltas = [[-5.0]*10, [], [], [], []]

trainer._update_emission_parameters(deltas)
new_mu = trainer.model.rbf_mu[0]
print(f"Old Mu: 5.0, Target: -5.0, New Mu (Momentum): {new_mu}")

# Expected: 0.9*5 + 0.1*(-5) = 4.5 - 0.5 = 4.0
if not (3.5 < new_mu < 4.5):
    raise ValueError("Momentum logic is incorrect or missing.")
print("TASK 6 SUCCESS")
```

---

### FINAL TASK: LEFT HAND SUPPORT (INTERFACE ONLY)
**Context:** The model is agnostic, but the user needs positive (RH) and negative (LH) outputs.
**Target File:** `interface.py`

**Instructions:**
1.  Modify `predict_fingering` signature to accept a new argument: `hand_sign: int = 1` (default to 1).
2.  At the very end of the function, before returning `fingers`, multiply the result array:
    `fingers = fingers * hand_sign`
3.  Do NOT touch the internal logic of Viterbi.

**Validation Script (`test_task_7.py`):**
```python
import numpy as np
from soft_position_hmm.interface import predict_fingering
# Mock objects not needed, just check if the function accepts the arg
# and if we can visually confirm logic by reading the code or 
# running a dummy prediction if possible.
print("TASK 7: Manual Verify - check if 'fingers * hand_sign' is at end of predict_fingering.")
```


DO NOT FORGET TO APPEND AN ENTRY TO `CHANGELOG_DEV.md` FOR EACH ACTION (editing a file, running a script, even fixing a typo)