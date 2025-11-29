import numpy as np
import numba as nb

# ANCHORS: The relative hand positions.
# Range: -12 (hand to the left) to +12 (hand to the right) step 3
ANCHORS = np.array([-12, -9, -6, -3, 0, 3, 6, 9, 12], dtype=np.int32)

class SoftPositionModel:
    def __init__(self):
        # 1. Geometry Parameters (RBF)
        # 5 fingers, 9 anchors.
        self.rbf_weights = np.random.uniform(0, 1, (5, 9)).astype(np.float64)

        # Biomechanical Initialization to prevent getting stuck at mu=0
        self.rbf_mu = np.array([-4.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float64)
        self.rbf_sigma = np.array([4.0, 1.5, 1.5, 1.5, 2.5], dtype=np.float64)

        # 2. Inertia Parameters (Movement Cost)
        self.inertia_weight = 1.0
        self.time_slope     = 10.0  # Controls how fast the hand "stiffens"
        self.time_center    = 0.2   # Pivot point (seconds)

@nb.njit
def compute_emission_score(delta_pitch: int, finger_idx: int, rbf_mu: np.ndarray, rbf_sigma: np.ndarray) -> float:
    """
    Computes the emission score using a simplified Gaussian.
    """
    epsilon = 1e-9 # To avoid log(0) or division by zero

    ideal_offset = rbf_mu[finger_idx]
    width = rbf_sigma[finger_idx]

    # Ensure width is not zero
    if width < epsilon:
        width = epsilon

    # Simplified Gaussian RBF
    z = np.exp(-((delta_pitch - ideal_offset)**2) / (2 * width**2))

    # Return log probability
    return np.log(z + epsilon)

@nb.njit
def compute_inertia_cost(physical_distance: float, dt: float, slope: float, center: float, weight: float) -> float:
    """
    Computes the inertia cost for hand movement.
    """
    # 1. Calculate Stiffness lambda(t)
    stiffness = 1.0 / (1.0 + np.exp(slope * (dt - center)))

    # 2. Calculate Distance Cost
    cost = stiffness * physical_distance * weight

    return cost
