import numpy as np
import numba as nb

# Range -12 to +12 with step 1
# Defines the discrete space of hand positions relative to a note
ANCHORS = np.arange(-12, 13, 1, dtype=np.int32)

# Precomputed constant for Gaussian Normalization: 0.5 * log(2 * pi)
HALF_LOG_2PI = 0.9189385332046727

class SoftPositionModel:
    def __init__(self):
        # 1. Geometry Parameters (RBF)
        # Biomechanical Initialization
        self.rbf_mu = np.array([-4.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float64)
        self.rbf_sigma = np.array([4.0, 1.5, 1.5, 1.5, 2.5], dtype=np.float64)

        # 2. Inertia Parameters (Movement Cost)
        self.inertia_weight = 1.0
        self.time_slope     = 10.0  # Controls how fast the hand "stiffens"
        self.time_center    = 0.2   # Pivot point (seconds)

@nb.njit(cache=True)
def compute_emission_score(delta_pitch: int, finger_idx: int, rbf_mu: np.ndarray, rbf_sigma: np.ndarray) -> float:
    """
    Computes the Log-PDF of the observed pitch delta given the finger and anchor.
    Corrects the previous implementation by including the normalization term.
    
    Formula: ln(P(x)) = -ln(sigma) - 0.5*ln(2pi) - 0.5*((x-mu)/sigma)^2
    """
    sigma_min = 1.0
    
    mu = rbf_mu[finger_idx]
    sigma = rbf_sigma[finger_idx]

    if sigma < sigma_min:
        sigma = sigma_min

    # Standardized distance (z-score component)
    diff = delta_pitch - mu
    squared_diff = diff * diff
    variance = sigma * sigma

    # Log-Likelihood calculation
    # Term 1: Normalization (-ln(sigma * sqrt(2pi)))
    log_norm = -np.log(sigma) - HALF_LOG_2PI
    
    # Term 2: Quadratic penalty
    quadratic = -(squared_diff) / (2.0 * variance)

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

    # 1. Calculate Stiffness lambda(t) using Sigmoid
    # As dt -> 0, exp(...) -> large, stiffness -> 0 (wait, logic check below)
    # Original logic: 1 / (1 + exp(slope * (dt - center)))
    # If dt is small (e.g. 0.01) and center is 0.2: exp(10 * -0.19) ~ exp(-1.9) ~ 0.15 -> stiff ~ 0.86 (High cost)
    # If dt is large (e.g. 1.0): exp(10 * 0.8) ~ huge -> stiff ~ 0 (Low cost)
    stiffness = 1.0 / (1.0 + np.exp(slope * (dt - center)))

    # 2. Calculate Distance Cost
    cost = stiffness * physical_distance * weight

    return min(cost, 8.0)