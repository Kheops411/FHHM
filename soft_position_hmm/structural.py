import numpy as np
from .core import ANCHORS

# --- Constants & Dimensions ---

N_FINGERS = 5
N_ANCHORS = len(ANCHORS)
N_STATES = N_FINGERS * N_FINGERS * N_ANCHORS # 5 * 5 * 9 = 225

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
