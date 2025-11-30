import numpy as np
from .core import SoftPositionModel
from .structural import ViterbiLattice
from .inference import run_forward_pass, backtracking

def predict_fingering(
    notes_pitch: np.ndarray,
    notes_ontime: np.ndarray,
    model: SoftPositionModel,
    agility_matrix: np.ndarray = None,
    smoothing_weight: float = 0.0
):
    """
    High-level API to predict fingerings for a sequence of notes.
    """
    n_obs = len(notes_pitch)
    if n_obs == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # 1. Setup Data Structures
    lattice = ViterbiLattice(n_obs)

    # Default agility if None (Zero log-prob = Uniform)
    if agility_matrix is None:
        agility_matrix = np.zeros((5, 5, 5), dtype=np.float64)

    # 2. Run Forward Pass
    run_forward_pass(
        n_obs=n_obs,
        notes_pitch=notes_pitch,
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

    # 3. Backtrack
    fingers, anchors = backtracking(
        n_obs,
        lattice.log_probs,
        lattice.backpointers
    )

    return fingers, anchors
