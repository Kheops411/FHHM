# Project Context: Soft-Position HMM for Piano Fingering

You are entering a project designed to solve a complex biomechanical problem: **predicting optimal piano fingering** (which finger plays which note) given a musical score.

### The Core Problem
Standard Hidden Markov Models (HMMs) fail at piano fingering because they only track the finger (1-5). They ignore the **hand's position** on the keyboard.
*   *Example:* Playing a generic "C" with the thumb (1) is easy. Playing a high "C" with the thumb while the hand is anchored two octaves lower is physically impossible.

### The Solution: "Soft-Position" Architecture
This project implements a hybrid HMM that tracks a **dual hidden state**:
1.  **The Finger ($f$):** 1 (Thumb) to 5 (Pinky).
2.  **The Anchor ($k$):** The center position of the hand relative to the note.

**How it works:**
*   **Emission (Geometry):** Instead of a hard matrix, we use Gaussian distributions (`core.py`). We calculate the probability of a finger reaching a note based on where the hand is anchored.
*   **Transition (Physics):** We model the "cost" of moving the hand (Inertia) vs. stretching the fingers (Elasticity).
*   **Inference:** We use the Viterbi algorithm (`inference.py`) optimized with **Numba** for high performance.

### Your Mission
The codebase (`soft_position_hmm/`) has been written based on theoretical specifications but **has never been executed**. It is currently a "black box" of unverified math.

**Why strict testing is required:**
Because this is a probabilistic system (EM Algorithm), **bugs do not always cause crashes**. They often result in "silent failures" where the model converges to mathematical garbage or produces physically impossible fingerings (e.g., jumping the hand 10 times a second).

**You cannot "guess" your way through debugging this.** You must verify the mathematical integrity of each component before the system can be trained.

# Technical Specification: Soft-Position HMM (Implementation v1.0)

**Architecture Overview:**
Hybrid HMM combining a **Trigram (2nd-Order) chain** for fingers and a **1st-Order chain** for hand position.
*   **Geometric Engine:** Parametric Gaussian Emission ($\mu, \sigma$).
*   **Training Method:** Viterbi Training (Hard EM) with Analytical MLE updates.
*   **Topology:** Sequential processing with Soft Constraints for chords.

#### 1. State Space Structure

The hidden state $S_t$ at time $t$ describes the current finger, the previous finger, and the hand position:

$$ S_t = (f_{t-1}, f_t, k_t) $$

**Definitions:**
*   **$f_{t-1}, f_t \in \{0..4\}$**: The finger indices (mapped to 1..5). We store $f_{t-1}$ to enable trigram transitions ($f_{t-2} \to f_{t-1} \to f_t$).
*   **$k_t \in \{0..24\}$**: A discrete index representing the "center of the hand" (Anchor).
    *   **Anchor Grid ($c_k$):** Integer range `[-12, -11, ..., 0, ..., +11, +12]` (semitones relative to the pitch).
    *   **Resolution:** 1 semitone.

**Viterbi Trellis Dimensions:**
*   State size per time step: $5 \times 5 \times 25 = 625$ logical states.
*   Transition complexity: Each state calculation considers $(f_{t-2}, k_{t-1})$ from the previous timestep.

#### 2. Emission Score (Geometric Compatibility)

Quantifies the probability of a finger $f_t$ playing a pitch $p_t$ given a hand position $c_{k_t}$, assuming a Gaussian distribution of finger reach.

**Calculation:**
For an observation $p_t$ and a candidate state $(f_t, k_t)$:

1.  **Calculate Delta:** $\delta = p_t - (p_t + c_{k_t}) = -c_{k_t}$ (Relative offset from hand center).
    *   *Note in code:* Effectively $\delta_{pitch} = -ANCHORS[k]$.
2.  **Gaussian Density:**
    $$ z = \exp\left( -\frac{(\delta - \mu_{f_t})^2}{2\sigma_{f_t}^2} \right) $$
    *   $\mu_{f_t}$: Learned mean position for finger $f_t$.
    *   $\sigma_{f_t}$: Learned standard deviation (reach width) for finger $f_t$ (clamped $\ge 1.0$).
3.  **Log-Probability:**
    $$ E_{emit} = \ln(z + \epsilon) $$

#### 3. Unified Transition Model

| Symbole | Variable Code | Définition |
| :--- | :--- | :--- |
| $p_t$ | `notes_pitch[t]` | Hauteur MIDI de la note à l'instant $t$. |
| $c_{k_t}$ | `ANCHORS[k_curr]` | Valeur de l'ancre (décalage relatif) choisie à l'instant $t$. |
| $H_t$ | `hand_pos_curr` | Position absolue de la main ($p_t + c_{k_t}$). |

The transition score aggregates digital agility, dynamic inertia (time-dependent), and static smoothing.

$$ T(S_{t-1} \to S_t) = T_{agile} - T_{inertia} - T_{smooth} $$

**3.1 Digital Agility ($T_{agile}$)**
Look-up in a trigram tensor $A$ of shape $5 \times 5 \times 5$:
$$ T_{agile} = A[f_{t-2}, f_{t-1}, f_t] $$

**3.2 Dynamic Inertia ($T_{inertia}$)**
Models the biomechanical cost of moving the hand base, modulated by the inter-onset interval ($\Delta t$).

$$ T_{inertia} = \lambda(\Delta t) \cdot \Delta d_{phys} \cdot w_{inertia} $$

*   Let $H_t = p_t + c_{k_t}$ be the absolute position of the hand center (MIDI pitch + Anchor offset).
*   **$\Delta d_{phys} = |H_t - H_{t-1}| = |(p_t + c_{k_t}) - (p_{t-1} + c_{k_{t-1}})|$**: The absolute physical distance traveled by the hand base in semitones.
*   **Stiffness Function $\lambda(\Delta t)$:**
    $$ \lambda(\Delta t) = \frac{1}{1 + e^{\text{slope} \cdot (\Delta t - \text{center})}} $$
    *   Legato ($\Delta t \to 0 \implies \lambda \to 1$): Movement is costly.
    *   Rest ($\Delta t \gg 0 \implies \lambda \to 0$): Movement is cheap.

**3.3 Static Smoothing ($T_{smooth}$)**
A time-independent regularization term to prevent hand jitter.

$$ T_{smooth} = |c_{k_t} - c_{k_{t-1}}| \cdot w_{smooth} $$
**Definitions:**
*   $|c_{k_t} - c_{k_{t-1}}|$: The **relative anchor change** (change in hand posture/offset), independent of the pitch played. This term penalizes shifting the "center of the hand" relative to the fingers, encouraging stability in the internal hand configuration.


#### 4. Topology & Chord Handling

**4.1 Sequential ordering**
Input notes are strictly ordered by time ($t_{on}$), then by pitch ($p$) ascending.

**4.2 Soft Constraints for Chords**
Notes with $\Delta t \approx 0$ are treated sequentially.
*   **Mechanism:** The Inertia function generates a maximal stiffness $\lambda \approx 1.0$.
*   **Effect:** Any change in hand position ($k_t \neq k_{t-1}$) incurs a maximal penalty ($1.0 \times \text{dist} \times w_{inertia}$).
*   **Result:** The Viterbi path is mathematically forced to maintain the same anchor $k$ for all notes in a chord, unless the emission gain of moving is astronomically high (physically impossible intervals).

#### 5. Training Strategy (Viterbi Training / Hard EM)

The model parameters ($\mu, \sigma$) are learned via Iterative Viterbi Training.

**5.1 Biomechanical Initialization**
Parameters are initialized with fixed scalar values representing a relaxed right hand:
*   $\mu_{init}$: `[-4.0, -2.0, 0.0, 2.0, 5.0]` (Thumb left, Pinky right).
*   $\sigma_{init}$: `[4.0, 1.5, 1.5, 1.5, 2.5]` (Thumb/Pinky more flexible).

**5.2 Learning Loop**
1.  **E-Step (Alignment):** Run **Constrained Forward Pass** on training data.
    *   $f_t$ is fixed to the ground truth annotation.
    *   $k_t$ is inferred to find the optimal hand position sequence.
2.  **M-Step (Analytical Update):**
    *   Collect all observed relative distances $\delta$ for each finger $f$.
    *   Update $\mu_f = \text{Mean}(\delta_f)$.
    *   Update $\sigma_f = \text{Std}(\delta_f)$ (with floor at 1.0).

**Required Data Structures:**

*   **Global Constants:**
    *   `ANCHORS`: `int32` array `[-12..+12]`.
*   **Model Parameters (`SoftPositionModel`):**
    *   `rbf_mu`: `float64` array `[5]`.
    *   `rbf_sigma`: `float64` array `[5]`.
    *   `inertia_weight`: `float`.
    *   `time_slope`, `time_center`: `float` (Sigmoid params).
*   **Hyperparameters:**
    *   `agility_matrix`: `float64` tensor `[5, 5, 5]` (Log-probabilities).
    *   `smoothing_weight`: `float`.
*   **Runtime (`ViterbiLattice`):**
    *   `log_probs`: `[Time, 5, 5, 25]`.
    *   `backpointers`: `[Time, 5, 5, 25, 3]`.


# Technical Validation Protocol (Strict Implementation)

**Context:** You are initializing the `soft_position_hmm` module.
**Developer Profile Constraints:**
1.  **No improvisation:** Use the exact code snippets provided below for data generation.
2.  **No assumptions:** Verify every component in isolation before running integration tests.
3.  **Mandatory Debugging:** If a test fails, you must log the shapes (`.shape`) and types (`.dtype`) of arrays involved.

**Execution Order:** Phase 0 -> Phase 1 -> Phase 2. **Do not skip steps.**

---

## Phase 0: Low-Level Component Validation (Unit Tests)

**File:** `tests_validation/test_00_components.py`
**Goal:** Validate data parsing regex and Numba JIT compilation before any model logic is touched.

### 0.1 Parser Robustness
**Instruction:** Copy this test function exactly. It verifies that `load_pig_file` correctly handles comments and garbage data, preventing silent failures later.

```python
def test_parser_logic():
    from soft_position_hmm.utils import clean_finger_str, sitch_to_pitch
    
    # 1. Test Finger Cleaning
    assert clean_finger_str("3") == 3, "Failed simple finger"
    assert clean_finger_str("4_1") == 4, "Failed substitution handling"
    assert clean_finger_str("-2") == -2, "Failed left hand negative"
    assert clean_finger_str("x") == 0, "Failed garbage input"

    # 2. Test Pitch Parsing
    assert sitch_to_pitch("C4") == 60, "C4 must be 60"
    assert sitch_to_pitch("A#0") == 22, "A#0 calculation wrong"
    try:
        sitch_to_pitch("H5")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid pitch string did not raise ValueError")
    
    print("[PASS] Parser Logic")

if __name__ == "__main__":
    test_parser_logic()
```

### 0.2 Numba JIT Math
**Instruction:** Validate that `core.py` functions compile and return floats, not NaNs.

```python
def test_numba_math():
    import numpy as np
    from soft_position_hmm.core import compute_emission_score, compute_inertia_cost
    
    # Dummy data
    mu = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    
    # 1. Test Emission
    # Finger 0, Delta 0 -> Should be high probability (close to 0 log-prob)
    score = compute_emission_score(0, 0, mu, sigma)
    assert isinstance(score, float), "Emission must return float"
    assert score < 0, "Log prob must be negative"
    assert score > -1.0, "Perfect match should have high score"

    # 2. Test Inertia
    cost = compute_inertia_cost(physical_distance=12.0, dt=0.05, slope=10.0, center=0.2, weight=1.0)
    assert cost > 0, "Inertia cost must be positive"
    
    print("[PASS] Numba Math")

if __name__ == "__main__":
    test_numba_math()
```

---

## Phase 1: Integration & Logic (Functional Tests)

**File:** `tests_validation/test_01_integration.py`

### 1.1 Cold Start Sanity Check (With Strict Data Generation)
**Instruction:** Do not create your own data. Use this `generate_c_major` function to prevent array shape errors.

```python
import numpy as np
from soft_position_hmm.utils import NOTE_DTYPE
from soft_position_hmm.core import SoftPositionModel
from soft_position_hmm.interface import predict_fingering

def generate_c_major():
    # 8 notes
    notes = np.zeros(8, dtype=NOTE_DTYPE)
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    onsets = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    for i in range(8):
        notes[i]['pitch'] = pitches[i]
        notes[i]['ontime'] = onsets[i]
        notes[i]['original_idx'] = i
    return notes

def test_cold_start():
    notes = generate_c_major()
    model = SoftPositionModel()
    
    # Extract arrays expected by inference
    p = np.array([n['pitch'] for n in notes], dtype=np.int32)
    t = np.array([n['ontime'] for n in notes], dtype=np.float64)
    
    fingers, anchors = predict_fingering(p, t, model)
    
    print(f"Fingers: {fingers}")
    
    assert len(fingers) == 8, "Output length mismatch"
    assert np.all(fingers > 0), "Found invalid finger 0"
    assert np.all(fingers <= 5), "Found invalid finger > 5"
    
    # Heuristic check: C-Major typically uses thumb (1) and avoids repeated fingers
    assert fingers[0] in [1, 2], "Scale should start with thumb or index"
    
    print("[PASS] Cold Start")

if __name__ == "__main__":
    test_cold_start()
```

### 1.2 Chord Constraints Check
**Instruction:** Verify specifically that simultaneous notes force the same Anchor index.

```python
def test_chord_constraint():
    model = SoftPositionModel()
    # C-Major Chord (C, E, G) at exact same time
    p = np.array([60, 64, 67], dtype=np.int32)
    t = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    
    fingers, anchors = predict_fingering(p, t, model)
    
    print(f"Chord Anchors: {anchors}")
    
    assert anchors[0] == anchors[1] == anchors[2], \
        f"Anchors must be identical for chords. Got {anchors}"
        
    print("[PASS] Chord Constraint")
```

---

## Phase 2: Training Mechanics

**File:** `tests_validation/test_02_training.py`

### 2.1 Convergence Monotonicity
**Instruction:** Do not use the full dataset. Point to `scores/001-1_fingering.txt` only.
**Requirement:** You must print the delta between iterations.

```python
def test_convergence():
    from soft_position_hmm.training import SoftPositionTrainer
    import numpy as np

    trainer = SoftPositionTrainer()
    # Run 5 iterations on a single file
    history = trainer.train(['scores/001-1_fingering.txt'], n_iterations=5)
    
    print("Likelihood History:", history)
    
    # Strict Monotonicity Check
    history_arr = np.array(history)
    diffs = np.diff(history_arr)
    
    if np.any(diffs < -1e-5): # Allow tiny float errors
        print("FAIL: Log-Likelihood decreased!")
        # Debugging info
        for i, diff in enumerate(diffs):
            if diff < 0:
                print(f"Iter {i}->{i+1} dropped by {diff}")
        raise ValueError("EM Algorithm failed monotonicity check.")
        
    print("[PASS] Convergence")

if __name__ == "__main__":
    test_convergence()
```

---

**Summary of Deliverables:**
You are required to implement and run these 3 files:
1.  `tests_validation/test_00_components.py`
2.  `tests_validation/test_01_integration.py`
3.  `tests_validation/test_02_training.py`

**Exit Criteria:**
Do NOT proceed to full model training until all snippets print `[PASS]`.
If an error occurs, do not comment out the assertion. Fix the code in `soft_position_hmm`.