import numpy as np
from .core import SoftPositionModel, RH_FINGER_BASE_POS, LH_FINGER_BASE_POS
from .structural import ViterbiLattice
from .inference import run_forward_pass, backtracking
from .utils import PITCH_TO_KEYPOS_LUT

def predict_fingering(
    notes_pitch: np.ndarray,
    notes_ontime: np.ndarray,
    model: SoftPositionModel,
    agility_matrix: np.ndarray = None,
    smoothing_weight: float = 0.0,
    hand_sign: int = 1
):
    n_obs = len(notes_pitch)
    if n_obs == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    lattice = ViterbiLattice(n_obs)
    if agility_matrix is None:
        agility_matrix = np.zeros((5, 5, 5), dtype=np.float64)

    notes_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch, 0].copy()

    if hand_sign == -1:
        notes_coord_x *= -1
        finger_base_pos = LH_FINGER_BASE_POS
    else:
        finger_base_pos = RH_FINGER_BASE_POS

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
        smoothing_weight=smoothing_weight,
        finger_base_pos=finger_base_pos
    )

    fingers, anchors = backtracking(
        n_obs,
        lattice.log_probs,
        lattice.backpointers
    )

    return fingers * hand_sign, anchors
