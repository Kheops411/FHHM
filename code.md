### core.py

```py
import numpy as np
import numba as nb

# Defines the discrete space of hand positions (lateral offset in pseudo-mm)
ANCHORS = np.arange(-150, 151, 15, dtype=np.int32)

# Average finger position relative to hand center (anchor) in pseudo-mm
FINGER_BASE_POS = np.array([-40.0, -20.0, 0.0, 20.0, 40.0], dtype=np.float64)

# Precomputed constant for Gaussian Normalization: 0.5 * log(2 * pi)
HALF_LOG_2PI = 0.9189385332046727

class SoftPositionModel:
    def __init__(self):
        # 1. Geometry Parameters (RBF)
        # Biomechanical Initialization
        self.rbf_mu = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.rbf_sigma = np.array([20.0, 15.0, 15.0, 15.0, 25.0], dtype=np.float64)

        # 2. Inertia Parameters (Movement Cost)
        self.inertia_weight = 1.0
        self.time_slope     = 10.0  # Controls how fast the hand "stiffens"
        self.time_center    = 0.2   # Pivot point (seconds)

@nb.njit(cache=True)
def compute_emission_score(
    note_coord_x: float,
    anchor_pos_x: float,
    finger_idx: int,
    rbf_mu: np.ndarray,
    rbf_sigma: np.ndarray
) -> float:
    """
    Computes the Log-PDF of the observed key position given the finger and anchor.
    Formula: ln(P(x)) = -ln(sigma) - 0.5*ln(2pi) - 0.5*((x-mu)/sigma)^2
    """
    sigma_min = 1.0

    # 1. Calculate the target x-position of the finger
    finger_target_pos = anchor_pos_x + FINGER_BASE_POS[finger_idx]

    # 2. Calculate the spatial delta
    delta = note_coord_x - finger_target_pos

    # 3. Get finger-specific RBF parameters
    mu = rbf_mu[finger_idx]
    sigma = rbf_sigma[finger_idx]
    if sigma < sigma_min:
        sigma = sigma_min

    # 4. Log-Likelihood calculation
    variance = sigma * sigma
    log_norm = -np.log(sigma) - HALF_LOG_2PI
    quadratic = -((delta - mu) ** 2) / (2.0 * variance)

    return log_norm + quadratic

@nb.njit(cache=True)
def compute_inertia_cost(physical_distance: float, dt: float, slope: float, center: float, weight: float) -> float:
    """
    Computes the inertia cost for hand movement.
    Now includes a hard constraint for simultaneous notes (chords).
    """
    # Chord Handling: If notes are close enough in time, inertia is zero.
    if dt < 0.03:
        return 0.0

    # Speed limit is determined by time available (dt), with a floor of 30ms
    speed_limit = max(dt, 0.03)

    # Inertia is the cost of moving a certain distance within the allowed time
    cost = physical_distance / speed_limit

    # Clip the cost to prevent extreme values from dominating the HMM
    return min(cost, 8.0) * weight # Keep weight for tuning

```

### inference.py

```py
import numpy as np
import numba as nb
from .core import compute_emission_score, compute_inertia_cost, ANCHORS
from .structural import N_FINGERS, N_ANCHORS

@nb.njit(cache=True)
def _clip(val, min_val, max_val):
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val

@nb.njit(cache=True)
def run_forward_pass(
    n_obs: int,
    notes_coord_x: np.ndarray,
    notes_ontime: np.ndarray,
    lattice_log_probs: np.ndarray,
    lattice_backpointers: np.ndarray,
    agility_matrix: np.ndarray,
    inertia_param_slope: float,
    inertia_param_center: float,
    inertia_weight: float,
    rbf_mu: np.ndarray,
    rbf_sigma: np.ndarray,
    smoothing_weight: float
):
    """
    Remplit le treillis Viterbi (log_probs et backpointers) en place.
    """

    # A. Initialisation (t = 0)
    current_note_coord_x = notes_coord_x[0]
    for f_curr in range(N_FINGERS):
        for k_curr in range(N_ANCHORS):
            hand_pos_center_x = current_note_coord_x + ANCHORS[k_curr]
            emit = compute_emission_score(
                current_note_coord_x,
                hand_pos_center_x,
                f_curr,
                rbf_mu,
                rbf_sigma
            )
            for f_prev in range(N_FINGERS):
                lattice_log_probs[0, f_prev, f_curr, k_curr] = emit

    # B. Récursion (t = 1 à n_obs - 1)
    for t in range(1, n_obs):
        dt = notes_ontime[t] - notes_ontime[t-1]
        current_note_coord_x = notes_coord_x[t]
        prev_note_coord_x = notes_coord_x[t-1]
        
        for f_curr in range(N_FINGERS):
            for k_curr in range(N_ANCHORS):
                hand_pos_center_x = current_note_coord_x + ANCHORS[k_curr]
                emit = compute_emission_score(
                    current_note_coord_x,
                    hand_pos_center_x,
                    f_curr,
                    rbf_mu,
                    rbf_sigma
                )

                for f_prev in range(N_FINGERS):
                    max_prob = -np.inf
                    best_k_prev = -1
                    best_f_prev2 = -1

                    for k_prev in range(N_ANCHORS):
                        hand_center_x_prev = prev_note_coord_x + ANCHORS[k_prev]
                        hand_center_x_curr = current_note_coord_x + ANCHORS[k_curr]
                        dist = np.abs(hand_center_x_curr - hand_center_x_prev)
                        inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)

                        for f_prev2 in range(N_FINGERS):
                            prev_prob = lattice_log_probs[t-1, f_prev2, f_prev, k_prev]
                            agility = agility_matrix[f_prev2, f_prev, f_curr]
                            smoothing = np.abs(ANCHORS[k_curr] - ANCHORS[k_prev]) * smoothing_weight
                            candidate = prev_prob + agility - inertia - smoothing + emit

                            if candidate > max_prob:
                                max_prob = candidate
                                best_k_prev = k_prev
                                best_f_prev2 = f_prev2

                    lattice_log_probs[t, f_prev, f_curr, k_curr] = max_prob
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 0] = best_f_prev2
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 1] = f_prev
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 2] = best_k_prev

@nb.njit(cache=True)
def run_constrained_forward_pass(
    n_obs: int,
    notes_coord_x: np.ndarray,
    notes_ontime: np.ndarray,
    true_fingers: np.ndarray,
    lattice_log_probs: np.ndarray,
    lattice_backpointers: np.ndarray,
    agility_matrix: np.ndarray,
    inertia_param_slope: float,
    inertia_param_center: float,
    inertia_weight: float,
    rbf_mu: np.ndarray,
    rbf_sigma: np.ndarray,
    smoothing_weight: float
):
    # A. Initialisation (t = 0)
    current_note_coord_x = notes_coord_x[0]
    f_init_range = range(N_FINGERS)
    if true_fingers[0] != -999: # FINGER_UNKNOWN
        f_init_range = range(true_fingers[0] - 1, true_fingers[0])

    for f_curr in f_init_range:
        for k_curr in range(N_ANCHORS):
            hand_pos_center_x = current_note_coord_x + ANCHORS[k_curr]
            emit = compute_emission_score(
                current_note_coord_x,
                hand_pos_center_x,
                f_curr,
                rbf_mu,
                rbf_sigma
            )
            for f_prev in range(N_FINGERS):
                lattice_log_probs[0, f_prev, f_curr, k_curr] = emit

    # B. Récursion (t = 1 à n_obs - 1)
    for t in range(1, n_obs):
        dt = notes_ontime[t] - notes_ontime[t-1]
        current_note_coord_x = notes_coord_x[t]
        prev_note_coord_x = notes_coord_x[t-1]
        
        f_curr_range = range(N_FINGERS)
        if true_fingers[t] != -999:
            f_curr_range = range(true_fingers[t] - 1, true_fingers[t])
            
        f_prev_range = range(N_FINGERS)
        if true_fingers[t-1] != -999:
            f_prev_range = range(true_fingers[t-1] - 1, true_fingers[t-1])

        for f_curr in f_curr_range:
            for k_curr in range(N_ANCHORS):
                hand_pos_center_x = current_note_coord_x + ANCHORS[k_curr]
                emit = compute_emission_score(
                    current_note_coord_x,
                    hand_pos_center_x,
                    f_curr,
                    rbf_mu,
                    rbf_sigma
                )

                for f_prev in f_prev_range:
                    max_prob = -np.inf
                    best_k_prev = -1
                    best_f_prev2 = -1

                    for k_prev in range(N_ANCHORS):
                        hand_center_x_prev = prev_note_coord_x + ANCHORS[k_prev]
                        hand_center_x_curr = current_note_coord_x + ANCHORS[k_curr]
                        dist = np.abs(hand_center_x_curr - hand_center_x_prev)
                        inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)

                        for f_prev2 in range(N_FINGERS):
                            prev_prob = lattice_log_probs[t-1, f_prev2, f_prev, k_prev]
                            agility = agility_matrix[f_prev2, f_prev, f_curr]
                            smoothing = np.abs(ANCHORS[k_curr] - ANCHORS[k_prev]) * smoothing_weight
                            
                            candidate = prev_prob + agility - inertia - smoothing + emit

                            if candidate > max_prob:
                                max_prob = candidate
                                best_k_prev = k_prev
                                best_f_prev2 = f_prev2

                    lattice_log_probs[t, f_prev, f_curr, k_curr] = max_prob
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 0] = best_f_prev2
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 1] = f_prev
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 2] = best_k_prev

@nb.njit(cache=True)
def backtracking(
    n_obs: int,
    lattice_log_probs: np.ndarray,
    lattice_backpointers: np.ndarray
):
    opt_fingers = np.zeros(n_obs, dtype=np.int32)
    opt_anchors = np.zeros(n_obs, dtype=np.int32)

    # 1. Trouver le meilleur état final
    best_prob = -np.inf
    f_prev, f_curr, k_curr = -1, -1, -1

    reshaped_probs = lattice_log_probs[-1].ravel()
    best_flat_idx = np.argmax(reshaped_probs)

    k_curr = best_flat_idx % N_ANCHORS
    f_curr = (best_flat_idx // N_ANCHORS) % N_FINGERS
    f_prev = best_flat_idx // (N_FINGERS * N_ANCHORS)

    opt_fingers[-1] = f_curr + 1
    opt_anchors[-1] = k_curr

    # 2. Itérer à rebours
    for t in range(n_obs - 1, 0, -1):
        bp = lattice_backpointers[t, f_prev, f_curr, k_curr]
        f_prev2, _, k_prev = bp[0], bp[1], bp[2]

        opt_fingers[t-1] = f_prev + 1
        opt_anchors[t-1] = k_prev

        f_curr = f_prev
        f_prev = f_prev2
        k_curr = k_prev

    return opt_fingers, opt_anchors
```

### interface.py

```py
import numpy as np
from .core import SoftPositionModel
from .structural import ViterbiLattice
from .inference import run_forward_pass, backtracking
from .utils import PITCH_TO_KEYPOS_LUT

def predict_fingering(
    notes_pitch: np.ndarray,
    notes_ontime: np.ndarray,
    model: SoftPositionModel,
    agility_matrix: np.ndarray = None,
    smoothing_weight: float = 0.0,
    hand_sign: int = 1 # 1 pour main droite (RH), -1 pour main gauche (LH)
):
    """
    API de haut niveau pour prédire le doigté d'une séquence de notes.
    """
    n_obs = len(notes_pitch)
    if n_obs == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    lattice = ViterbiLattice(n_obs)
    if agility_matrix is None:
        agility_matrix = np.zeros((5, 5, 5), dtype=np.float64)

    notes_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch, 0].copy()
    if hand_sign == -1:
        notes_coord_x *= -1

    run_forward_pass(
        n_obs=n_obs,
        notes_coord_x=notes_coord_x,
        notes_ontime=notes_ontime,
        lattice_log_probs=lattice.log_probs,
        lattice_backpointers=lattice.backpointers,
        agility_matrix=agility_matrix,
        inertia_param_slope=model.time_slope,
        inertia_param_center=model.time_center,
        inertia_weight=model.inertia_weight,
        rbf_mu=model.rbf_mu,
        rbf_sigma=model.rbf_sigma,
        smoothing_weight=smoothing_weight
    )

    fingers, anchors = backtracking(
        n_obs,
        lattice.log_probs,
        lattice.backpointers
    )

    return fingers * hand_sign, anchors
```

### structural.py

```py
import numpy as np
from .core import ANCHORS

# --- Constants & Dimensions ---

N_FINGERS = 5
N_ANCHORS = len(ANCHORS)
N_STATES = N_FINGERS * N_FINGERS * N_ANCHORS

class ViterbiLattice:
    """
    A data structure to hold the Viterbi trellis (log probabilities) and
    backpointers for the Soft-Position HMM.
    """
    def __init__(self, n_obs: int):
        """
        Initializes the Viterbi lattice with the given number of observations.

        Args:
            n_obs (int): The number of notes (time steps) in the sequence.
        """
        # 1. log_probs: Stores the max log probability of reaching each state.
        self.log_probs = np.full(
            (n_obs, N_FINGERS, N_FINGERS, N_ANCHORS),
            -np.inf,
            dtype=np.float64
        )

        # 2. backpointers: Stores the coordinates of the previous state.
        self.backpointers = np.full(
            (n_obs, N_FINGERS, N_FINGERS, N_ANCHORS, 3),
            -1,
            dtype=np.int8
        )

        # Debug: Print memory allocation
        log_probs_mb = self.log_probs.nbytes / (1024 * 1024)
        backpointers_mb = self.backpointers.nbytes / (1024 * 1024)
        # print(f"Allocated Lattice: {log_probs_mb + backpointers_mb:.2f} MB")

```

### training.py

```py
import numpy as np
from .core import SoftPositionModel, ANCHORS, FINGER_BASE_POS
from .structural import ViterbiLattice, N_FINGERS
from .inference import run_constrained_forward_pass, backtracking
from .utils import load_pig_file, apply_time_dep_pitch_order, FINGER_UNKNOWN, PITCH_TO_KEYPOS_LUT

class SoftPositionTrainer:
    def __init__(self):
        self.model = SoftPositionModel()
        self.agility_matrix = np.log(np.full((5, 5, 5), 1.0 / 125, dtype=np.float64))

    def train(self, file_paths: list, n_iterations: int = 5, smoothing_weight: float = 0.0) -> list:
        log_likelihood_history = []

        for it in range(n_iterations):
            print(f"--- Iteration {it + 1}/{n_iterations} ---")

            total_log_likelihood = 0.0
            finger_deltas = [[] for _ in range(N_FINGERS)]
            transition_counts = np.zeros((N_FINGERS, N_FINGERS, N_FINGERS), dtype=np.float64)

            for fpath in file_paths:
                try:
                    all_notes = load_pig_file(fpath)
                except ValueError as e:
                    print(f"Skipping file {fpath}: {e}")
                    continue

                rh_notes = all_notes[all_notes['channel'] == 0]
                lh_notes = all_notes[all_notes['channel'] == 1]
                sequences_to_process = [rh_notes, lh_notes]

                # MODIFIÉ: Utiliser enumerate pour identifier la main (0=RH, 1=LH)
                for hand_idx, hand_notes in enumerate(sequences_to_process):
                    if len(hand_notes) < 3:
                        continue

                    notes_sorted = apply_time_dep_pitch_order(hand_notes)
                    n_obs = len(notes_sorted)
                    notes_pitch = np.array([n['pitch'] for n in notes_sorted], dtype=np.int32)
                    notes_ontime = np.array([n['ontime'] for n in notes_sorted], dtype=np.float64)

                    # MODIFIÉ: Préparer les coordonnées et les inverser pour la main gauche
                    notes_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch, 0].copy()
                    if hand_idx == 1:  # Si c'est la main gauche
                        notes_coord_x *= -1

                    true_fingers = np.array(
                        [abs(n['finger']) if n['finger'] != FINGER_UNKNOWN else FINGER_UNKNOWN
                         for n in notes_sorted],
                        dtype=np.int32
                    )

                    lattice = ViterbiLattice(n_obs)
                    run_constrained_forward_pass(
                        n_obs=n_obs,
                        notes_coord_x=notes_coord_x, # MODIFIÉ: Passer les coordonnées
                        notes_ontime=notes_ontime,
                        true_fingers=true_fingers,
                        lattice_log_probs=lattice.log_probs,
                        lattice_backpointers=lattice.backpointers,
                        agility_matrix=self.agility_matrix,
                        inertia_param_slope=self.model.time_slope,
                        inertia_param_center=self.model.time_center,
                        inertia_weight=self.model.inertia_weight,
                        rbf_mu=self.model.rbf_mu,
                        rbf_sigma=self.model.rbf_sigma,
                        smoothing_weight=smoothing_weight
                    )

                    final_log_prob = np.max(lattice.log_probs[-1])
                    if np.isfinite(final_log_prob):
                        total_log_likelihood += final_log_prob

                    _, opt_anchors = backtracking(n_obs, lattice.log_probs, lattice.backpointers)

                    for i in range(n_obs):
                        if true_fingers[i] != FINGER_UNKNOWN:
                            finger_idx = true_fingers[i] - 1
                            anchor_idx = opt_anchors[i]
                            
                            delta = -ANCHORS[anchor_idx] - FINGER_BASE_POS[finger_idx]
                            finger_deltas[finger_idx].append(delta)

                    for i in range(2, n_obs):
                        f_prev2 = true_fingers[i - 2] - 1
                        f_prev1 = true_fingers[i - 1] - 1
                        f_curr = true_fingers[i] - 1

                        if 0 <= f_prev2 < 5 and 0 <= f_prev1 < 5 and 0 <= f_curr < 5:
                            transition_counts[f_prev2, f_prev1, f_curr] += 1.0

            self._update_emission_parameters(finger_deltas)
            self._update_agility_parameters(transition_counts)
            log_likelihood_history.append(total_log_likelihood)
            print(f"Total Log Likelihood: {total_log_likelihood}")

        return log_likelihood_history

    def _update_emission_parameters(self, finger_deltas):
        print("Updating RBF parameters (mu and sigma)...")
        for i in range(N_FINGERS):
            if len(finger_deltas[i]) > 1:
                self.model.rbf_mu[i] = np.mean(finger_deltas[i])
                new_sigma = np.std(finger_deltas[i])
                self.model.rbf_sigma[i] = max(5.0, new_sigma) # Plancher pour éviter l'effondrement
        print("Model RBF parameters updated.")
        print(f"  New Mu: {np.round(self.model.rbf_mu, 2)}")
        print(f"  New Sigma: {np.round(self.model.rbf_sigma, 2)}")

    def _update_agility_parameters(self, counts):
        print("Updating Agility Matrix...")
        smoothed_counts = counts + 1e-3
        sums = smoothed_counts.sum(axis=2, keepdims=True)
        # Éviter la division par zéro pour les états non observés
        sums[sums == 0] = 1.0
        probs = smoothed_counts / sums
        self.agility_matrix = np.log(probs + 1e-12)
        non_zeros = np.count_nonzero(counts)
        print(f"Agility Matrix updated. {non_zeros} active transitions learned.")
```

### utils.py

```py
import numpy as np
import numba as nb
from typing import Tuple
import re

# Global Constant for unknown/invalid fingers
FINGER_UNKNOWN = -999

# Global LUT: (128 pitches, 2 coordinates [x, y])
PITCH_TO_KEYPOS_LUT = np.zeros((128, 2), dtype=np.float32)

def _compute_pitch_to_keypos_lut():
    """
    Computes a LUT mapping MIDI pitch to a (x, y) coordinate system.
    - X is the lateral position in pseudo-mm.
    - Y is the vertical position (0 for white keys, 1 for black keys).
    Based on standard piano key dimensions.
    """
    # White key properties
    WHITE_KEY_WIDTH = 23.5  # pseudo-mm
    # Pitch class to white key index mapping (0=C, 1=D, etc.)
    PC_TO_WHITE_KEY = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}

    for pitch in range(128):
        octave = pitch // 12
        pc = pitch % 12  # Pitch Class (0-11 for C, C#, ..., B)

        if pc in PC_TO_WHITE_KEY:
            # This is a white key
            white_key_index = PC_TO_WHITE_KEY[pc]
            x_pos = octave * 7 * WHITE_KEY_WIDTH + white_key_index * WHITE_KEY_WIDTH
            y_pos = 0
        else:
            # This is a black key. Position it relative to the previous white key.
            prev_white_key_pc = pc - 1
            white_key_index = PC_TO_WHITE_KEY[prev_white_key_pc]
            # Black keys are positioned halfway between white keys
            x_pos = octave * 7 * WHITE_KEY_WIDTH + white_key_index * WHITE_KEY_WIDTH + (WHITE_KEY_WIDTH / 2.0)
            y_pos = 1

        PITCH_TO_KEYPOS_LUT[pitch, 0] = x_pos
        PITCH_TO_KEYPOS_LUT[pitch, 1] = y_pos

_compute_pitch_to_keypos_lut()

def pitch_to_keypos(midi_pitch: int) -> Tuple[int, int]:
    if not (0 <= midi_pitch < 128):
        raise ValueError(f"Pitch {midi_pitch} out of bounds")
    row = PITCH_TO_KEYPOS_LUT[midi_pitch]
    return int(row[0]), int(row[1])

def subtract_keypos(kp1: Tuple[int,int], kp2: Tuple[int,int]) -> Tuple[int,int]:
    return (kp1[0] - kp2[0], kp1[1] - kp2[1])

@nb.njit(cache=True)
def pitch_to_keypos_numba(midi_pitch: int, lut: np.ndarray) -> np.ndarray:
    return lut[midi_pitch]

@nb.njit(cache=True)
def subtract_keypos_numba(x1, y1, x2, y2):
    return x1 - x2, y1 - y2

@nb.njit(cache=True)
def lattice_delta_to_index(dx: int, dy: int, width_x: int = 15) -> int:
    if dx < -width_x: dx = -width_x
    if dx > width_x:  dx = width_x
    return 3 * (dx + width_x) + dy + 1

# --- Data Parsing & Ordering ---

# Updated DTYPE to match usage. 
# Note: 'channel' stores the Hand Index (0=RH, 1=LH) from column 6.
NOTE_DTYPE = np.dtype([
    ('original_idx', np.int32), 
    ('ontime', np.float64), 
    ('offtime', np.float64),
    ('pitch_str', 'U10'), 
    ('pitch', np.int32), 
    ('velocity', np.int32),
    ('channel', np.int32), 
    ('finger_str', 'U20'), 
    ('finger', np.int32)
])

SITCH_REGEX = re.compile(r'([A-G])([#b+-]*)([0-9])')

def sitch_to_pitch(sitch: str) -> int:
    if sitch in ("R", "rest"): return -1
    match = SITCH_REGEX.match(sitch)
    if not match: raise ValueError(f"Invalid pitch string: {sitch}")
    note_name, accidentals, octave_str = match.groups()
    p_rel = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}[note_name]
    octave = int(octave_str)
    pitch = p_rel + (octave - 4) * 12
    acc_val = 0
    for char in accidentals:
        if char in ('#', '+'): acc_val += 1
        elif char in ('b', '-'): acc_val -= 1
    return pitch + acc_val

def clean_finger_str(finger_str: str) -> int:
    """
    Parses a finger string (e.g., "4_1", "-3", "-5_-1") into a single integer.
    Handles substitution by taking the starting finger (first part).
    """
    try:
        # Take the part before any substitution marking (underscore)
        # "4_1" -> "4", "-5_-1" -> "-5"
        cleaned_str = finger_str.split('_')[0]
        finger_val = int(cleaned_str)
        
        # Clamp/Check validity: [-5, 5] excluding 0
        if 0 < finger_val <= 5:
            return finger_val
        if -5 <= finger_val < 0:
            return finger_val
    except (ValueError, IndexError):
        pass
    return FINGER_UNKNOWN

def load_pig_file(filepath: str) -> np.ndarray:
    """
    Robust line-by-line parser for PIG files.
    Enforces 8 columns per line to prevent data shifting.
    
    Columns:
    0: ID
    1: Onset
    2: Offset
    3: Note Name
    4: Onset Velocity
    5: Offset Velocity (Ignored)
    6: Hand Index (0=RH, 1=LH) -> stored in 'channel'
    7: Finger Index (can include substitution e.g. "1_2")
    """
    raw_notes = []
    
    with open(filepath, 'r') as f:
        line_num = 0
        for line in f:
            line_num += 1
            # 1. Strip comments and whitespace
            content = line.partition('//')[0].strip()
            
            if not content:
                continue
                
            tokens = content.split()
            
            # 2. Strict Structure Validation
            if len(tokens) != 8:
                raise ValueError(f"Line {line_num} malformed: expected 8 columns, got {len(tokens)}. Content: '{content}'")
            
            try:
                # 3. Parse fields
                original_idx = int(tokens[0])
                ontime       = float(tokens[1])
                offtime      = float(tokens[2])
                pitch_str    = tokens[3]
                pitch        = sitch_to_pitch(pitch_str)
                velocity     = int(tokens[4])
                # token[5] is offset velocity, ignored.
                hand_idx     = int(tokens[6]) 
                finger_str   = tokens[7]
                finger       = clean_finger_str(finger_str)
                
                raw_notes.append((
                    original_idx, ontime, offtime, pitch_str, pitch, 
                    velocity, hand_idx, finger_str, finger
                ))
                
            except Exception as e:
                raise ValueError(f"Parsing error on line {line_num}: {e}")

    # Convert to structured numpy array
    if not raw_notes:
        return np.zeros(0, dtype=NOTE_DTYPE)
        
    return np.array(raw_notes, dtype=NOTE_DTYPE)


def apply_time_dep_pitch_order(notes: np.ndarray, time_threshold: float = 0.03) -> np.ndarray:
    """
    Groups notes by onset (within 0.03s tolerance) and sorts them by Pitch ASCENDING.
    """
    if len(notes) == 0:
        return notes

    # Global sort by time
    notes = np.sort(notes, order=['ontime'], kind='stable')

    reordered_notes = []

    i = 0
    while i < len(notes):
        cluster_indices = [i]
        j = i + 1
        # Cluster simultaneous notes
        while j < len(notes) and abs(notes[j]['ontime'] - notes[j-1]['ontime']) < time_threshold:
            cluster_indices.append(j)
            j += 1

        cluster_notes = notes[cluster_indices]

        # Sort by pitch ASCENDING (Low -> High)
        sorted_indices = np.argsort(cluster_notes['pitch'], kind='stable')

        reordered_notes.extend(cluster_notes[sorted_indices])
        i = j

    return np.array(reordered_notes, dtype=notes.dtype)
```

