
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

---

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