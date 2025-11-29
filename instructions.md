
# Introduction

### Project Overview: The "Soft-Position" HMM

**The Goal**
We are refactoring an existing piano fingering algorithm (you'll find in the `./python/` folder). The current version treats the hand as a teleporting object: it calculates fingering based only on the interval between the previous note and the current note. This fails in real music because it doesn't understand that moving the whole hand sideways takes effort.

We are building a **"Soft-Position" Model**. This model giveBachs the hand a virtual "Center of Gravity" (an anchor point) that persists over time. It forces the algorithm to decide: *"Should I stretch my fingers to hit this note, or should I move my whole hand?"*

Here is how the logic works from a computer science perspective.

#### 1. The Core Concept: The "Anchor" ($k$)
In the old code, the state was just `[PreviousFinger, CurrentFinger]`.
In the new code, the state is **3-Dimensional**: `[PreviousFinger, CurrentFinger, HandAnchor]`.

*   **What is an Anchor?**
    Imagine a discrete grid relative to the current note. We define a set of "Anchors" (e.g., 9 positions) representing where the center of the hand is relative to the key being played.
    *   *Example:* If the Anchor is 0, the hand is centered right over the note. If the Anchor is -6, the hand is centered 6 semitones to the left.
*   **The Consequence:**
    The Viterbi algorithm now has to track the hand position. If the algorithm wants to keep the hand static (low cost), it must choose an Anchor index that stays the same from note $t$ to note $t+1$.

#### 2. The "Geometry" Logic (Emission Score)
Instead of a simple lookup table for note intervals, we implement a **Learnable Reach Function**.

*   **Input:** A specific Finger (e.g., Index finger) and a Distance (how far the note is from the Hand Anchor).
*   **Logic:**
    We use a "Radial Basis Function" (RBF). Think of it as a heatmap or a bell curve.
    *   Does the Index finger "like" playing a note 2 semitones to the right of the hand center? **Yes** (High Score).
    *   Does the Thumb "like" playing a note 5 semitones to the *right* of the hand center (crossing over)? **No** (Low Score).
*   **Why this matters:** We don't hardcode these rules. The model will *learn* the shape of these curves (the "shape" of the hand) during training.

#### 3. The "Physics" Logic (Transition Cost)
This is the most critical part for fixing the musical errors. The cost of moving from one state to another is now the sum of two distinct costs:

1.  **Finger Agility (Standard):** Is 2-then-1 a good sequence? (Same as before).
2.  **Hand Inertia (New):** This calculates the physical distance between the previous Anchor and the current Anchor.

**The "Time-Aware" Constraint:**
The cost of moving the Anchor is dynamic, based on the time difference ($\Delta t$) between notes:
*   **Fast playing (Legato):** The Hand Inertia cost becomes **extreme**. The algorithm is forced to keep the Anchor constant and stretch the fingers.
*   **Slow playing / Pauses:** The Hand Inertia cost drops to **zero**. The algorithm is allowed to reset the hand position freely.

#### 4. The Training Logic: "Guess and Refine" (Hard EM)
This is the trickiest part. Our dataset contains notes and fingers, but **it does not contain hand positions (Anchors)**. We have to infer them.

We use an iterative approach (Expectation-Maximization):
1.  **Initialize:** Create a "dummy" hand shape (e.g., thumb is on the left, pinky on the right).
2.  **E-Step (The Guess):** Run the Viterbi algorithm on the training data. Since we know the correct fingers, we force the path to use those fingers, but we let the algorithm find the *optimal sequence of Hand Anchors* that fits those fingers.
3.  **M-Step (The Update):** Now that we have a guessed sequence of Anchors, we calculate the distances and update our "Geometry" (RBF) weights to make those distances more probable.
4.  **Repeat:** We loop this until the hand shape stops changing.

### Summary of Changes for the Developer

*   **Data Structures:** The Viterbi grid expands from `5x5` to `5x5x9`.
*   **Math:** Probability lookups are replaced by function evaluations (RBF curves).
*   **Logic:** A dynamic "Time vs. Movement" check is added to every transition.
*   **Training:** Simple counting is replaced by an iterative loop.

### Work Order: Soft-Position HMM Implementation

We are deprecating the previous interval-based statistical model. We are moving to a new architecture called **"Soft-Position HMM"**.

**⚠️ READ CAREFULLY BEFORE CODING:**

1.  **Do Not Reuse Logic blindly:** While `utils.py` remains mostly valid, `model.py` and `training.py` require a complete rewrite. Do not attempt to "patch" the old probability tables.
2.  **Mandatory Testing:** You must implement a unit test for the **Emission Score (RBF)** function in isolation before integrating it into Viterbi. I want to see the output values for specific inputs to ensure the math is correct.
3.  **Debug Tracing:** You are required to implement a `debug_mode` boolean flag in the Viterbi function. When `True`, it must print:
    *   The calculated $\Delta t$ and resulting $\lambda$ (inertia factor).
    *   The raw RBF activation values for the winning state.
    *   **Do not guess** if the path looks wrong; inspect these values.

---

### Technical Specification: Soft-Position HMM

**Architecture Overview:**
Hybrid HMM combining a 3rd-Order chain for fingers and a 1st-Order chain for hand position.
*   **Geometric Engine:** 1D RBF Lattice with Normalization and Softplus.
*   **Training Method:** Viterbi Training (Hard EM) with Biomechanical Initialization.

#### 1. State Space Structure

To capture both digital agility and hand placement, the hidden state $S_t$ at time $t$ is defined as a triplet:

$$ S_t = (f_{t-1}, f_t, k_t) $$

**Definitions:**
*   **$f_{t-1}, f_t \in \{1..5\}$**: The finger used at the previous step and the current step. We store $f_{t-1}$ to allow for 3rd-order finger transitions ($f_{t-2} \to f_{t-1} \to f_t$).
*   **$k_t \in \{0..N_{anchors}-1\}$**: A discrete index representing the "center of the hand" on a relative grid (Lattice).
    *   **Required Anchors ($c_k$):** `[-12, -9, -6, -3, 0, +3, +6, +9, +12]` (relative semitones).

**Viterbi Trellis Dimensions:**
*   State size per time step: $5 \times 5 \times 9 = 225$ states.
*   Transition complexity: Each state depends on $(f_{t-2}, k_{t-1})$.

#### 2. Emission Score (Geometric Compatibility)

This replaces the old discrete output tables. It quantifies the anatomical feasibility of a finger $f$ playing a pitch $p$ given a hand position $c_k$.

**2.1 Calculation Pipeline (Normalized RBF)**
For an observation $(p_t)$ and a candidate state $(f_t, k_t)$:

1.  **Calculate Delta:** $\delta = p_t - c_{k_t}$ (Distance from note to hand center).
2.  **RBF Activation Vector ($\Phi$):**
    Compute the activation of the 9 anchors for the value $\delta$ (using a Triangle kernel or fixed Gaussian).
3.  **L2 Field Normalization:**
    $$ \tilde\Phi(\delta) = \frac{\Phi(\delta)}{\|\Phi(\delta)\|_2 + \epsilon} $$
4.  **Morphological Projection & Centering:**
    $$ z = \frac{W_{f_t} \cdot \tilde\Phi(\delta) - \mu_{f_t}}{\sigma_{f_t} + \epsilon} $$
    *   $W_{f_t}$: Learnable weight vector for finger $f_t$ (Size 9).
    *   $\mu_{f_t}, \sigma_{f_t}$: Running mean/std of activations (for numerical stability).
5.  **Positive Transformation (Softplus):**
    $$ S_{emit}(f_t, \delta) = \ln(1 + e^z) $$

**2.2 Final Energy Term**
$$ E_{emit} = -\frac{1}{\tau} \log(S_{emit}(f_t, \delta)) $$
*   $\tau$: Learnable temperature scalar.

#### 3. Unified Transition Model (Agility & Inertia)

The transition score combines finger agility with the cost of moving the arm, modulated by available time.

$$ T(S_{t-1} \to S_t) = T_{agile}(f_{t-2}, f_{t-1} \to f_t) + T_{inertia}(k_{t-1} \to k_t, \Delta t) $$

**3.1 Digital Agility ($T_{agile}$)**
Standard 3rd-order transition tensor (shape $5 \times 5 \times 5$). Learned via counting on the dataset (logic similar to previous model).

**3.2 Dynamic Inertia ($T_{inertia}$)**
This models the physics of arm movement.

$$ T_{inertia} = \lambda(\Delta t) \cdot |c_{k_t} - c_{k_{t-1}}| \cdot w_{shift} $$

*   $|c_{k_t} - c_{k_{t-1}}|$: Physical distance in semitones between hand positions.
*   $w_{shift}$: Global weight for movement cost.

**Stiffness Function $\lambda(\Delta t)$:**
An inverted sigmoid function:
$$ \lambda(\Delta t) = \frac{1}{1 + e^{\alpha(\Delta t - t_0)}} $$

*   **Legato ($\Delta t \approx 0$):** $\lambda \to 1$. Hand movement is penalized. The model prefers keeping $k$ stable.
*   **Staccato/Rest ($\Delta t \gg 0$):** $\lambda \to 0$. Hand movement is "free". The model allows resetting the hand position ($k_t \neq k_{t-1}$).

#### 4. Topology Constraints (Chords)

For note clusters where $\Delta t < 30ms$ (Chords), replace probabilistic rules with binary masks:
1.  **Spatial Uniqueness:** All notes in the chord must share the exact same hand index $k_t$.
2.  **Physical Order:** If $p_A < p_B$, then $f_A$ must be $< f_B$ (with configurable exceptions for thumb/index crossover).
    *   *Implementation:* Assign a log-probability of $-\infty$ to any state violating these rules.

#### 5. Training Strategy (Viterbi Training / Hard EM)

Since $k_t$ is latent (not in dataset), use **Hard EM**.

**5.1 Biomechanical Initialization (Warm Start)**
The matrix $W$ (5x9) **must not be random**. Initialize it using a Gaussian formula:
$$ W_{f,i} = \exp\left( - \frac{(c_i - \mu_{init}^f)^2}{2(\sigma_{init}^f)^2} \right) $$

**Initialization Parameters (Right Hand example):**
*   **Finger 1 (Thumb):** $\mu \approx -4.0, \sigma \approx 4.0$. (Asymmetric: strong left/center).
*   **Fingers 2, 3, 4:** Centered at -2, 0, +2 respectively, $\sigma \approx 1.5$.
*   **Finger 5:** $\mu \approx +5.0, \sigma \approx 2.5$.

**5.2 Learning Loop**
1.  **E-Step (Alignment):** Run Viterbi on training data with **fixed observed fingers** ($f_t = f_{annotated}$) to find the optimal sequence of hand positions $k_t$.
2.  **M-Step (Optimization):** Optimize weights $W$ and $\tau$ to maximize the likelihood of the collected $\delta$ values (Gradient Descent). Update running stats $\mu_f, \sigma_f$.

---

**Required Data Structures for Implementation:**

*   `RBF_ANCHORS`: Constant Vector `[9]`.
*   `morphology_weights` ($W$): Learnable Matrix `[5, 9]`.
*   `emission_scaling` ($\tau$): Learnable Scalar.
*   `agility_matrix`: Fixed Tensor `[5, 5, 5]` (Log-probs).
*   `inertia_params`: Scalars `alpha`, `t0` for sigmoid.
*   `Viterbi_Lattice`: `[Time, 5, 5, 9]` (dims: $f_{t-1}, f_t, k_t$).


---

#### Usuful to know

- The project involves refactoring a piano fingering algorithm into a 'Soft-Position' Hidden Markov Model (HMM). This model introduces a hand 'anchor' (k) to represent the hand's center of gravity and penalize large movements. The HMM state is (f_{t-2}, f_{t-1}, k_{t-1}).
	
- Python dependencies (numpy, numba, matplotlib, pytest, psutil) can be installed with pip install numpy numba matplotlib pytest psutil.

- To update the local repository with the latest changes from the remote, run git checkout main followed by git pull origin main.
	
- The ViterbiLattice class in soft_position_hmm/structural.py is designed as a strict data container. Algorithmic logic, such as setting initial probabilities for t=0, should be handled outside of this class.

- The ViterbiLattice backpointer tensor intentionally includes a redundant coordinate for prev_finger_1. This design simplifies the backtracking algorithm in Numba by making it stateless with respect to loop indices.
	
- Generated test artifacts, such as plots (.png files), should be saved in the tests/ directory.
	
- New Python packages must include an empty __init__.py file to be importable.
	
- The source code for the new 'Soft-Position' HMM is located in the soft_position_hmm/ directory.

---


# Developer Instructions: Milestone 1 - The Mathematical Core

## Project Context & Restrictions
We are building the **Soft-Position HMM** from scratch.
*   **Do not** modify the existing files in `./python/`.
*   **Do not** attempt to write the full Viterbi algorithm yet.
*   **Do not** guess parameter values randomly when things fail. You will use the provided test script to inspect values.

## 1. Environment & Setup

1.  **Install Dependencies:**
    ```bash
    pip install numpy numba matplotlib
    ```
2.  **Create the Directory Structure:**
    Create a new folder named `soft_position_hmm` at the root of the project.
    Create a folder named `tests` at the root.
    
    Structure must look like this:
    ```text
    project_root/
    ├── python/ (Old code - ignore)
    ├── scores/ (Data - ignore for now)
    ├── soft_position_hmm/      <-- NEW WORKSPACE
    │   ├── __init__.py         (Empty)
    │   ├── utils.py            (Copy this from ./python/utils.py)
    │   └── core.py             (YOU WILL CREATE THIS)
    ├──tests/
    │   └── test_milestone_1.py (YOU WILL CREATE THIS)
    └──other files and folders (IGNORE and do not modify!!)
    ```

3.  **Action:** Copy `./python/utils.py` into `./soft_position_hmm/utils.py`.

---

## 2. Implementation: `core.py`

You must create `soft_position_hmm/core.py`. This file defines the physics and geometry of the hand.

### Step 2.1: The `SoftPositionModel` Class
Create a class that holds the parameters. It must *not* contain logic, only data.

```python
import numpy as np
import numba as nb

# ANCHORS: The relative hand positions.
# Range: -12 (hand to the left) to +12 (hand to the right) step 3
ANCHORS = np.array([-12, -9, -6, -3, 0, 3, 6, 9, 12], dtype=np.int32)

class SoftPositionModel:
    def __init__(self):
        # 1. Geometry Parameters (RBF)
        # 5 fingers, 9 anchors. 
        # For now, initialize with random noise for testing, 
        # but structured enough to pass sanity checks.
        self.rbf_weights = np.random.uniform(0, 1, (5, 9)).astype(np.float64)
        self.rbf_mu      = np.zeros(5, dtype=np.float64) # Centers for normalization
        self.rbf_sigma   = np.ones(5, dtype=np.float64)  # Widths for normalization
        
        # 2. Inertia Parameters (Movement Cost)
        self.inertia_weight = 1.0
        self.time_slope     = 10.0  # Controls how fast the hand "stiffens"
        self.time_center    = 0.2   # Pivot point (seconds)
```

### Step 2.2: The Emission Function (Geometry)
In `core.py`, write a JIT-compiled function `compute_emission_score`.

**Math Logic:**
For a specific finger $f$ and a distance $\delta$ (note position - anchor position):
1.  Calculate RBF activation: We will use a simplified Gaussian for Milestone 1.
    $$ z = \exp\left( - \frac{(\delta - \text{ideal\_offset}_f)^2}{2 \cdot \text{width}_f^2} \right) $$
    *(Note: In later milestones we will use the learnable weights, but for M1, hardcode `ideal_offset` to verify the logic).*
    *   Thumb (0): ideal_offset = -4 (Likes playing to the left of hand center)
    *   Middle (2): ideal_offset = 0
    *   Pinky (4): ideal_offset = +4
2.  Return log probability: $\ln(z + \epsilon)$

**Implementation Requirement:**
*   Use `@nb.njit`
*   Inputs: `delta_pitch` (int), `finger_idx` (int 0-4).
*   Returns: `float` (Log Probability).

### Step 2.3: The Inertia Function (Physics)
In `core.py`, write a JIT-compiled function `compute_inertia_cost`.

**Math Logic:**
1.  Calculate Stiffness $\lambda(t)$:
    $$ \lambda = \frac{1}{1 + e^{\text{slope} \cdot (\Delta t - \text{center})}} $$
    *   If $\Delta t$ is small (0.05s), $\lambda \to 1$ (High Stiffness).
    *   If $\Delta t$ is large (1.0s), $\lambda \to 0$ (Low Stiffness).
2.  Calculate Distance Cost:
    $$ Cost = \lambda \cdot |Anchor_{current} - Anchor_{prev}| \cdot \text{weight} $$

**Implementation Requirement:**
*   Use `@nb.njit`
*   Inputs: `k_prev` (int), `k_curr` (int), `dt` (float), `slope` (float), `center` (float), `weight` (float).
*   Returns: `float` (Cost, usually negative in log-space, or positive cost to be subtracted).

---

## 3. Validation: `test_milestone_1.py`

**Crucial:** You are not allowed to say "it works". You must prove it by generating a plot and a data table.
Create `tests/test_milestone_1.py`.

### Test A: The Hand Shape Visualizer
Write code that:
1.  Initializes the Model.
2.  Loops through `delta` from -15 to +15.
3.  Calculates `compute_emission_score` for Finger 1 (Thumb) and Finger 5 (Pinky).
4.  Uses `matplotlib` to plot these two curves.
    *   **Requirement:** The Thumb peak must be to the LEFT (negative delta). The Pinky peak must be to the RIGHT (positive delta).
    *   **Debug Print:** Print the max value coordinate for both fingers.

### Test B: The Time-Inertia Check
Write code that:
1.  Defines a movement of anchors: from Anchor 0 to Anchor 3 (Distance = 3).
2.  Calculates `compute_inertia_cost` for three time intervals:
    *   `dt = 0.05` (Fast/Legato)
    *   `dt = 0.2` (Medium)
    *   `dt = 1.5` (Pause)
3.  **Debug Print:** Print a formatted table like this:

```text
DT (sec) | Stiffness (0-1) | Total Cost
---------------------------------------
0.05     | ......          | ......
0.20     | ......          | ......
1.50     | ......          | ......
```

**Pass Condition:**
*   Cost at 0.05 must be HIGH.
*   Cost at 1.50 must be NEAR ZERO.

## 4. Debugging Guidelines

If the test fails, do **not** change the code randomly.
1.  **Add `print()` inside your python test loop** (not the numba function) to see the raw inputs you are sending.
2.  Check the sign of `delta`. Remember: `delta = pitch - anchor`.
3.  Check the sigmoid logic. If `dt` is small, the exponent `slope * (dt - center)` should be negative (assuming slope > 0).

**Deliverables for Milestone 1:**
1.  `soft_position_hmm/core.py`
2.  `tests/test_milestone_1.py`
3.  The generated plot image (`hand_shape_test.png`).
4.  The text output of the Inertia table.

# Developer Instructions: Milestone 2 - The Data Structures (State Space)

## Project Context
In Milestone 1, we validated the physics engine.
In Milestone 2, we define the **State Space** for the HMM.
Unlike the old model which used 2D matrices (`5x5`), the Soft-Position HMM uses a **3D State**:
$$ S_t = (\text{Finger}_{t-1}, \text{Finger}_{t}, \text{Anchor}_{t}) $$

We need to create the structural code that allocates memory for this 3D lattice and manages the dimensions.

## 1. Environment & Setup

1.  **Install Dependencies:**
    You need `psutil` to verify memory consumption in your tests.
    ```bash
    pip install psutil
    ```

2.  **File Structure:**
    Create a new file `soft_position_hmm/structural.py`.

## 2. Implementation: `structural.py`

This file handles the allocation of the Viterbi Trellis (the grid of probabilities).

### Step 2.1: Constants & Dimensions
Import `ANCHORS` from `core.py` to ensure consistency.

**Requirements:**
*   Define `N_FINGERS = 5`.
*   Define `N_ANCHORS = len(ANCHORS)`.
*   Define `N_STATES = N_FINGERS * N_FINGERS * N_ANCHORS` (Should be $5 \times 5 \times 9 = 225$).

### Step 2.2: The `ViterbiLattice` Class
Create a class `ViterbiLattice`.
The `__init__` method must take `n_obs` (number of notes) as an input and allocate **Numpy arrays** filled with zeros (or `-inf` for probabilities).

**Required Tensors (Attributes):**

1.  **`log_probs`**:
    *   **Shape:** `(n_obs, N_FINGERS, N_FINGERS, N_ANCHORS)`
    *   **Dtype:** `np.float64`
    *   **Initialization:** Fill with `-np.inf` (Log-space zero).
    *   *Explanation:* Stores the max probability of reaching state $(f_{t-1}, f_t, k_t)$ at time $t$.

2.  **`backpointers`**:
    *   **Shape:** `(n_obs, N_FINGERS, N_FINGERS, N_ANCHORS, 3)`
    *   **Dtype:** `np.int8` (to save memory, fingers are 0-4, anchors 0-8).
    *   **Initialization:** Fill with `-1`.
    *   *Explanation:* Stores the coordinates of the *previous* state that led to the current max probability.
        *   Index 0: `prev_finger_2` ($f_{t-2}$)
        *   Index 1: `prev_finger_1` ($f_{t-1}$)
        *   Index 2: `prev_anchor` ($k_{t-1}$)

**Debug Requirement:**
Inside `__init__`, add a print statement (commented out by default, but you must write it) that calculates the size of these arrays in Megabytes and prints it.
`# print(f"Allocated Lattice: {size_in_mb:.2f} MB")`

## 3. Validation: `test_milestone_2.py`

Create `tests/test_milestone_2.py`. You must verify that the dimensions are exactly as expected and that the memory layout is correct.

### Test A: Dimension Integrity
1.  Initialize `ViterbiLattice` with `n_obs = 100`.
2.  Assert that `log_probs.shape` is exactly `(100, 5, 5, 9)`.
3.  Assert that `backpointers.shape` is exactly `(100, 5, 5, 9, 3)`.
4.  Assert that `log_probs[0, 0, 0, 0]` is `-inf`.
5.  **Debug Print:** Print the shapes to the console.

### Test B: Large Scale Memory Check
1.  Initialize `ViterbiLattice` with `n_obs = 10,000` (A very long concerto).
2.  Use `psutil` or `nbytes` to calculate the memory usage.
3.  **Constraint:** The total memory must be under **500 MB**.
    *   *Math check:* $10000 \times 225 \times 8$ bytes (float64) $\approx 18$ MB. $10000 \times 225 \times 3$ bytes (int8) $\approx 6.75$ MB. Total should be $\approx 25$ MB.
    *   If your test reports huge numbers (GBs), you have an error in your dimension logic.
4.  **Debug Print:** Print the calculated size in MB.

### Test C: Coordinate Mapping Sanity
1.  Verify that `N_ANCHORS` imported from `structural` matches `len(ANCHORS)` from `core`.
2.  Write a loop that iterates through every dimension of the `log_probs` array for `t=0` and sets a value (e.g., `1.0`).
3.  Verify that no `IndexError` is raised.

## 4. Debugging Guidelines for Milestone 2

*   **If you get an `IndexError`:** Check the order of dimensions. Is it `(t, f_prev, f_curr, k)` or `(t, k, f_prev, f_curr)`? Stick strictly to **`(Time, F_prev, F_curr, Anchor)`**.
*   **If memory usage is high:** Check your Dtypes. `backpointers` do not need `int64` or `float64`. Use `int8`.
*   **If `backpointers` shape is wrong:** Remember the last dimension is `3` because we need to point back to a triplet $(f_{t-2}, f_{t-1}, k_{t-1})$. Even though $f_{t-1}$ is redundant (it is part of the current state), we store it for explicit clarity during the backtracking phase.

**Deliverables for Milestone 2:**
1.  `soft_position_hmm/structural.py`
2.  `tests/test_milestone_2.py`
3.  Console output showing the exact shapes and memory usage in MB.

---

# Developer Instructions: Milestone 3 - The Forward Pass (Inference)

## Project Context
We now have the Physics Engine (`core.py`) and the Memory Structure (`structural.py`).
It is time to implement the core algorithm: **The Viterbi Forward Pass**.

This algorithm fills the lattice we created in Milestone 2. It finds the most probable path through the 3D state space.

## 1. Environment & Setup

1.  **File Structure:**
    Create a new file `soft_position_hmm/inference.py`.

2.  **Dependencies:**
    You will need `numba` for speed. Python loops are too slow for this 5-level nested operation.

## 2. Implementation: `inference.py`

You need to write a JIT-compiled function that takes the notes and the lattice, and computes the path.

### Step 2.1: The Function Signature
Define the function as follows:

```python
import numpy as np
import numba as nb
from .core import compute_emission_score, compute_inertia_cost, ANCHORS

@nb.njit(cache=True)
def run_forward_pass(
    n_obs: int,
    notes_pitch: np.ndarray, # Int32 array of pitches
    notes_ontime: np.ndarray, # Float64 array of onsets
    lattice_log_probs: np.ndarray,
    lattice_backpointers: np.ndarray,
    agility_matrix: np.ndarray, # Shape (5, 5, 5) -> log probs
    inertia_param_slope: float,
    inertia_param_center: float,
    inertia_weight: float
):
    """
    Fills the lattice_log_probs and lattice_backpointers in-place.
    """
    # ... implementation ...
```

### Step 2.2: Algorithm Logic (The Nested Loops)

This is the most critical part. Read carefully. The state at time $t$ is $(f_{prev}, f_{curr}, k_{curr})$.

**A. Initialization (t = 0)**
At the first note, we have no history. We must define the probabilities for all possible starting states $(f_0, k_0)$.
*   Loop over all `f_curr` (0-4) and `k_curr` (0-8).
*   Compute `emit = compute_emission_score(...)`.
*   Assign `lattice_log_probs[0, :, f_curr, k_curr] = emit`.
*   *Note:* Since there is no previous finger, we broadcast the same value to all `f_prev` indices at $t=0$, or just loop `f_prev` and assign the same value.

**B. Recursion (t = 1 to n_obs - 1)**
You must construct a loop structure that iterates through time. Inside, you calculate the max probability coming from the previous time step.

**Loop Hierarchy:**
1.  `for t` in `1` to `n_obs-1`:
    2.  Calculate `dt = notes_ontime[t] - notes_ontime[t-1]`.
    3.  `for f_curr` (Target Finger):
        4.  `for k_curr` (Target Anchor):
            5.  `emit = compute_emission_score(...)` (Depends on `notes_pitch[t]`, `f_curr`, `k_curr`).
            6.  `for f_prev` (The "Connector"):
                *   *Crucial:* `f_prev` is the **Previous Finger**. It is part of the *Target State* index, but it was the *Current Finger* in the previous timestep.
                7.  **Find the Best Previous Context:**
                    Initialize `max_prob = -inf`
                    Initialize `best_k_prev = -1`, `best_f_prev2 = -1`
                    
                    8.  `for k_prev` (Previous Anchor):
                        *   `inertia = compute_inertia_cost(...)` (Using `dt`).
                        
                        9.  `for f_prev2` (Finger at t-2):
                            *   `prev_prob = lattice_log_probs[t-1, f_prev2, f_prev, k_prev]`
                            *   `agility = agility_matrix[f_prev2, f_prev, f_curr]`
                            *   `candidate = prev_prob + agility + inertia + emit`
                            
                            *   **Update Max:** If `candidate > max_prob`, update `max_prob` and record `k_prev`, `f_prev2`.
                    
                    10. **Store Result:**
                        `lattice_log_probs[t, f_prev, f_curr, k_curr] = max_prob`
                        `lattice_backpointers[t, f_prev, f_curr, k_curr, 0] = best_f_prev2`
                        `lattice_backpointers[t, f_prev, f_curr, k_curr, 1] = f_prev`
                        `lattice_backpointers[t, f_prev, f_curr, k_curr, 2] = best_k_prev`

## 3. Validation: `test_milestone_3.py`

Create `tests/test_milestone_3.py`.
Since we cannot perform the full training yet, we will validate using a **synthetic micro-scenario**.

### The Scenario: "The Impossible Stretch"
We will create a sequence of 3 notes:
1.  **Note 0:** Pitch 60 (C4).
2.  **Note 1:** Pitch 72 (C5) - 1 octave up.
3.  **Note 2:** Pitch 60 (C4) - Back down.

**Conditions:**
*   `dt` is very small (0.05s) -> High Inertia Cost (Hand shouldn't move).
*   But the interval is 12 semitones. No single hand position can reach both C4 and C5 comfortably without moving.

**Test Setup:**
1.  Create dummy `notes_pitch = [60, 72, 60]`.
2.  Create dummy `notes_ontime = [0.0, 0.05, 0.10]`.
3.  Create a dummy `agility_matrix` filled with `0.0` (ignore finger agility for this test).
4.  Initialize `ViterbiLattice(3)`.
5.  Run `run_forward_pass`.

**Verification Logic (The "Probe"):**
After running the pass, inspect `lattice_log_probs` at `t=1` (The middle note, C5).
*   **Case A (Static Hand):** Look at probabilities where `k_curr` is same as `t=0`. They should be very low (bad emission score for reaching C5 from a C4 anchor).
*   **Case B (Moving Hand):** Look at probabilities where `k_curr` has shifted to the right. They should be higher (good emission), *even though* inertia cost was applied.
*   **Action:** Find the indices `(f_prev, f_curr, k_curr)` that have the **maximum** value in the entire lattice at `t=1` and `t=2`.
*   Print: "Best Anchor at t=1: ...", "Best Anchor at t=2: ..."

**Success Criteria:**
The test passes if the code runs without error and produces a chosen path where the probabilities are not all `-inf`.

## 4. Debugging Guidelines

1.  **Initialization Error:** If your results are all `-inf`, check your `t=0` loop. Did you add the emission score?
2.  **Index Out of Bounds:** Ensure `f_prev` loop goes 0-4 and `k_prev` goes 0-8.
3.  **Variable Confusion:** Be very careful with `f_prev` vs `f_prev2`.
    *   `f_prev` is the *row* of the lattice at time `t`.
    *   `f_prev` is the *column* of the lattice at time `t-1`.
    *   `f_prev2` is the *row* of the lattice at time `t-1`.

**Deliverables for Milestone 3:**
1.  `soft_position_hmm/inference.py`
2.  `tests/test_milestone_3.py`
3.  Console output showing the "Best Anchor" selected by the algorithm for the test sequence.