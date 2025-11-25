import numpy as np

class HMMParameters:
    """
    A class to load, store, and manage the parameters for the 2nd-order Fingering HMM.

    This class parses the `param_FHMM2.txt` file, which contains the initial, transition (1st and 2nd order),
    and output (1st and 2nd order) probabilities for both right and left hands.

    All probabilities are immediately converted to log-space upon loading. The values in the file
    are assumed to be pre-normalized. A small epsilon is added before taking the log to prevent `log(0)`.
    """
    def __init__(self, filepath: str, log_eps: float = 1e-30):
        self.filepath = filepath
        self.log_eps = log_eps

        # Initialize arrays to store log-probabilities
        self.log_initial_prob = np.zeros((2, 5), dtype=np.float64)
        self.log_transition1_prob = np.zeros((2, 5, 5), dtype=np.float64)
        self.log_transition2_prob = np.zeros((2, 5, 5, 5), dtype=np.float64)
        self.n_out = 3 * (15 * 2 + 1)
        self.log_output1_prob = np.zeros((2, 5, 5, self.n_out), dtype=np.float64)
        self.log_output2_prob = np.zeros((2, 5, 5, self.n_out), dtype=np.float64)

        self._parse_file()

    def _log(self, arr):
        return np.log(arr + self.log_eps)

    def _parse_file(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        line_idx = 0

        # --- Parse Initial Probabilities ---
        for h in range(2):
            line_idx += 1 # Skip header
            parts = np.array(list(map(float, lines[line_idx].strip().split())))
            self.log_initial_prob[h, :] = self._log(parts)
            line_idx += 1

        # --- Parse 1st Order Transition Probabilities ---
        for h in range(2):
            line_idx += 1 # Skip header
            for prev_f in range(5):
                parts = np.array(list(map(float, lines[line_idx].strip().split())))
                self.log_transition1_prob[h, prev_f, :] = self._log(parts)
                line_idx += 1

        # --- Parse 2nd Order Transition Probabilities ---
        for h in range(2):
            line_idx += 1 # Skip header
            for f_n_minus_2 in range(5):
                for f_n_minus_1 in range(5):
                    parts = np.array(list(map(float, lines[line_idx].strip().split())))
                    self.log_transition2_prob[h, f_n_minus_2, f_n_minus_1, :] = self._log(parts)
                    line_idx += 1

        # --- Parse 1st Order Output Probabilities ---
        for h in range(2):
            line_idx += 1 # Skip header
            for prev_f in range(5):
                for curr_f in range(5):
                    parts = np.array(list(map(float, lines[line_idx].strip().split()[2:])))
                    self.log_output1_prob[h, prev_f, curr_f, :] = self._log(parts)
                    line_idx += 1

        # --- Parse 2nd Order Output Probabilities ---
        for h in range(2):
            line_idx += 1 # Skip header
            for prev_f in range(5): # This corresponds to f_n-2
                for curr_f in range(5): # This corresponds to f_n
                    parts = np.array(list(map(float, lines[line_idx].strip().split()[2:])))
                    self.log_output2_prob[h, prev_f, curr_f, :] = self._log(parts)
                    line_idx += 1
