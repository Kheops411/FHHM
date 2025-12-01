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
