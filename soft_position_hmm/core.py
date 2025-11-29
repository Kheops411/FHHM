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

@nb.njit
def compute_emission_score(delta_pitch: int, finger_idx: int) -> float:
    """
    Computes the emission score using a simplified Gaussian for Milestone 1.
    """
    # Hardcoded parameters for Milestone 1
    ideal_offsets = np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float64)
    widths        = np.array([4.0, 3.0, 3.0, 2.5, 3.5], dtype=np.float64)
    epsilon = 1e-9 # To avoid log(0)

    ideal_offset = ideal_offsets[finger_idx]
    width = widths[finger_idx]

    # Simplified Gaussian RBF
    z = np.exp(-((delta_pitch - ideal_offset)**2) / (2 * width**2))

    # Return log probability
    return np.log(z + epsilon)

@nb.njit
def compute_inertia_cost(k_prev: int, k_curr: int, dt: float, slope: float, center: float, weight: float) -> float:
    """
    Computes the inertia cost for hand movement.
    """
    # 1. Calculate Stiffness lambda(t)
    stiffness = 1.0 / (1.0 + np.exp(slope * (dt - center)))

    # 2. Calculate Distance Cost
    # Look up anchor values from indices
    anchor_prev_val = ANCHORS[k_prev]
    anchor_curr_val = ANCHORS[k_curr]

    distance = np.abs(anchor_curr_val - anchor_prev_val)

    cost = stiffness * distance * weight

    return cost
