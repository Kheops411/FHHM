import numpy as np

class HMMParameters:
    """
    Loads, stores, and manages parameters for Fingering HMMs of order 2 or 3.
    """
    def __init__(self, filepath: str, log_eps: float = 1e-30):
        self.filepath = filepath
        self.log_eps = log_eps
        self.order = 2 # Default order

        with open(self.filepath, 'r') as f:
            content = f.read()
            if 'Transition Prob 3rd' in content:
                self.order = 3

        self.log_initial_prob = np.zeros((2, 5), dtype=np.float64)
        self.log_transition1_prob = np.zeros((2, 5, 5), dtype=np.float64)
        self.log_transition2_prob = np.zeros((2, 5, 5, 5), dtype=np.float64)
        self.n_out = 3 * (15 * 2 + 1)
        self.log_output1_prob = np.zeros((2, 5, 5, self.n_out), dtype=np.float64)
        self.log_output2_prob = np.zeros((2, 5, 5, self.n_out), dtype=np.float64)

        if self.order == 3:
            self.log_transition3_prob = np.zeros((2, 5, 5, 5, 5), dtype=np.float64)
            self.log_output3_prob = np.zeros((2, 5, 5, self.n_out), dtype=np.float64)
            # lam values from C++ constructor for 3rd order HMM
            self.lam1 = 0.0
            self.lam2 = 0.9

        self._parse_file()

    def _log(self, arr):
        return np.log(arr + self.log_eps)

    def _normalize_and_log(self, arr):
        """Normalizes a probability vector and converts it to log space."""
        s = np.sum(arr)
        if s > 0:
            arr /= s
        return self._log(arr)

    def _parse_file(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        def find_section(marker, start_line=0):
            for i in range(start_line, len(lines)):
                if marker in lines[i]:
                    return i
            return -1

        # --- Parse Initial Probabilities ---
        line_idx = find_section('Initial Prob Right') + 1
        self.log_initial_prob[0, :] = self._log(np.array(list(map(float, lines[line_idx].strip().split()))))
        line_idx = find_section('Initial Prob Left', line_idx) + 1
        self.log_initial_prob[1, :] = self._log(np.array(list(map(float, lines[line_idx].strip().split()))))

        # --- Parse Transitions ---
        def parse_block(marker, shape):
            arr = np.zeros(shape)
            line_idx = find_section(marker) + 1
            for h in range(shape[0]):
                it = np.nditer(arr[h, ..., 0], flags=['multi_index'])
                while not it.finished:
                    parts = np.array(list(map(float, lines[line_idx].strip().split())))
                    arr[h][it.multi_index] = parts
                    line_idx += 1
                    it.iternext()
                line_idx = find_section(marker.replace("Right", "Left"), line_idx) + 1
            return arr

        raw_tr1 = parse_block('Transition Prob Right', (2, 5, 5))
        raw_tr2 = parse_block('Transition Prob 2nd Right', (2, 5, 5, 5))

        if self.order == 3:
            raw_tr3 = parse_block('Transition Prob 3rd Right', (2, 5, 5, 5, 5))
            # Apply interpolation
            for h in range(2):
                for ippp in range(5):
                    for ipp in range(5):
                        for ip in range(5):
                            raw_tr3[h, ippp, ipp, ip, :] = (1 - self.lam2 - self.lam1) * raw_tr3[h, ippp, ipp, ip, :] + \
                                                             self.lam2 * raw_tr2[h, ipp, ip, :] + \
                                                             self.lam1 * raw_tr1[h, ip, :]
            # Normalize and log
            for h in range(2):
                for ippp in range(5):
                    for ipp in range(5):
                        for ip in range(5):
                           self.log_transition3_prob[h, ippp, ipp, ip, :] = self._normalize_and_log(raw_tr3[h, ippp, ipp, ip, :])

        # Normalize and log for order 2 and 1
        for h in range(2):
            for ip in range(5):
                self.log_transition1_prob[h, ip, :] = self._normalize_and_log(raw_tr1[h, ip, :])
                for ipp in range(5):
                    self.log_transition2_prob[h, ipp, ip, :] = self._normalize_and_log(raw_tr2[h, ipp, ip, :])

        # --- Parse Outputs ---
        def parse_output_block(marker, shape):
            arr = np.zeros(shape)
            line_idx = find_section(marker) + 1
            for h in range(shape[0]):
                for f1 in range(shape[1]):
                    for f2 in range(shape[2]):
                        parts = np.array(list(map(float, lines[line_idx].strip().split()[2:])))
                        arr[h, f1, f2, :] = parts
                        line_idx += 1
                line_idx = find_section(marker.replace("Right", "Left"), line_idx) + 1
            return arr

        raw_out1 = parse_output_block('Output Prob Right', (2, 5, 5, self.n_out))
        raw_out2 = parse_output_block('Output Prob 2nd Right', (2, 5, 5, self.n_out))
        if self.order == 3:
            raw_out3 = parse_output_block('Output Prob 3rd Right', (2, 5, 5, self.n_out))
            for h in range(2):
                for f1 in range(5):
                    for f2 in range(5):
                        self.log_output3_prob[h, f1, f2, :] = self._normalize_and_log(raw_out3[h, f1, f2, :])

        for h in range(2):
            for f1 in range(5):
                for f2 in range(5):
                    self.log_output1_prob[h, f1, f2, :] = self._normalize_and_log(raw_out1[h, f1, f2, :])
                    self.log_output2_prob[h, f1, f2, :] = self._normalize_and_log(raw_out2[h, f1, f2, :])
