import numpy as np
import numba as nb
from typing import Tuple
from . import utils

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
            self.lam1 = 0.0
            self.lam2 = 0.9
        else: # Order 2
            self.lam1 = 0.0

        self._parse_file()

    def _log(self, arr):
        # Ignore the "divide by zero" warning because it is intentional
        with np.errstate(divide='ignore'):
            return np.log(arr)

    def _normalize_and_log(self, arr):
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

        line_idx = find_section('Initial Prob Right') + 1
        self.log_initial_prob[0, :] = self._log(np.array(list(map(float, lines[line_idx].strip().split()))))
        line_idx = find_section('Initial Prob Left', line_idx) + 1
        self.log_initial_prob[1, :] = self._log(np.array(list(map(float, lines[line_idx].strip().split()))))

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

        if self.order == 2:
             for h in range(2):
                for ipp in range(5):
                    for ip in range(5):
                        raw_tr2[h, ipp, ip, :] = (1 - self.lam1) * raw_tr2[h, ipp, ip, :] + self.lam1 * raw_tr1[h, ip, :]

        if self.order == 3:
            raw_tr3 = parse_block('Transition Prob 3rd Right', (2, 5, 5, 5, 5))
            for h in range(2):
                for ippp in range(5):
                    for ipp in range(5):
                        for ip in range(5):
                            raw_tr3[h, ippp, ipp, ip, :] = (1 - self.lam2 - self.lam1) * raw_tr3[h, ippp, ipp, ip, :] + \
                                                             self.lam2 * raw_tr2[h, ipp, ip, :] + \
                                                             self.lam1 * raw_tr1[h, ip, :]
            for h in range(2):
                for ippp in range(5):
                    for ipp in range(5):
                        for ip in range(5):
                           self.log_transition3_prob[h, ippp, ipp, ip, :] = self._normalize_and_log(raw_tr3[h, ippp, ipp, ip, :])

        for h in range(2):
            for ip in range(5):
                self.log_transition1_prob[h, ip, :] = self._normalize_and_log(raw_tr1[h, ip, :])
                for ipp in range(5):
                    self.log_transition2_prob[h, ipp, ip, :] = self._normalize_and_log(raw_tr2[h, ipp, ip, :])

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

@nb.njit(cache=True)
def viterbi_2nd_order_numba(
    notes: np.ndarray,
    log_initial_prob: np.ndarray,
    log_transition1_prob: np.ndarray,
    log_transition2_prob: np.ndarray,
    log_output1_prob: np.ndarray,
    log_output2_prob: np.ndarray,
    pitch_to_keypos_lut: np.ndarray,
    hand: int,
    w1: float,
    w2: float,
    short_time_cost: float
) -> np.ndarray:
    """
    Optimized Viterbi for 2nd order HMM. Returns ONLY the optimal path (int32 array).
    """
    n_obs = len(notes)
    if n_obs < 3:
        # Return empty array if not enough notes
        return np.zeros(n_obs, dtype=np.int32)

    lp = np.full((5, 5), -np.inf, dtype=np.float64)
    amax = np.zeros((n_obs, 5, 5), dtype=np.int32)
    
    for n in range(1, n_obs):
        if n == 1:
            kp1 = utils.pitch_to_keypos_numba(notes[1]['pitch'], pitch_to_keypos_lut)
            kp0 = utils.pitch_to_keypos_numba(notes[0]['pitch'], pitch_to_keypos_lut)
            key_int_1_0 = utils.subtract_keypos_numba(kp1[0], kp1[1], kp0[0], kp0[1])
            idx1 = utils.lattice_delta_to_index(key_int_1_0[0], key_int_1_0[1])
            for kp in range(5):
                for k in range(5):
                    lp[kp, k] = (log_initial_prob[hand, kp] +
                                 log_transition1_prob[hand, kp, k] +
                                 log_output1_prob[hand, kp, k, idx1])
        else:
            pre_lp = lp.copy()

            kpn = utils.pitch_to_keypos_numba(notes[n]['pitch'], pitch_to_keypos_lut)
            kpn1 = utils.pitch_to_keypos_numba(notes[n-1]['pitch'], pitch_to_keypos_lut)
            key_int_n_n1 = utils.subtract_keypos_numba(kpn[0], kpn[1], kpn1[0], kpn1[1])
            idx1 = utils.lattice_delta_to_index(key_int_n_n1[0], key_int_n_n1[1])

            kpn2 = utils.pitch_to_keypos_numba(notes[n-2]['pitch'], pitch_to_keypos_lut)
            key_int_n_n2 = utils.subtract_keypos_numba(kpn[0], kpn[1], kpn2[0], kpn2[1])
            idx2 = utils.lattice_delta_to_index(key_int_n_n2[0], key_int_n_n2[1])

            short_time = abs(notes[n]['ontime'] - notes[n-1]['ontime']) < 0.03
            del_pitch = notes[n]['pitch'] - notes[n-1]['pitch']
            short_time2 = abs(notes[n]['ontime'] - notes[n-2]['ontime']) < 0.03
            del_pitch2 = notes[n]['pitch'] - notes[n-2]['pitch']

            for kp in range(5):
                for k in range(5):
                    max_log_prob = -np.inf
                    best_kpp = 0
                    for kpp in range(5):
                        st_cost = 0.0
                        if short_time and ((hand == 0 and (k - kp) * del_pitch < 0) or (hand == 1 and (k - kp) * del_pitch > 0)):
                            st_cost += short_time_cost
                        if short_time2 and ((hand == 0 and (k - kpp) * del_pitch2 < 0) or (hand == 1 and (k - kpp) * del_pitch2 > 0)):
                            st_cost += short_time_cost

                        log_prob = (pre_lp[kpp, kp] +
                                    log_transition2_prob[hand, kpp, kp, k] +
                                    w1 * log_output1_prob[hand, kp, k, idx1] +
                                    w2 * log_output2_prob[hand, kpp, k, idx2] +
                                    st_cost)

                        if log_prob > max_log_prob:
                            max_log_prob = log_prob
                            best_kpp = kpp

                    lp[kp, k] = max_log_prob
                    amax[n, kp, k] = best_kpp

    opt_path = np.zeros(n_obs, dtype=np.int32)
    max_lp = -np.inf
    kp_end, k_end = 0, 0
    for kp in range(5):
        for k in range(5):
            if lp[kp, k] > max_lp:
                max_lp = lp[kp, k]
                kp_end, k_end = kp, k

    opt_path[n_obs - 1] = k_end
    opt_path[n_obs - 2] = kp_end

    for n in range(n_obs - 3, -1, -1):
        opt_path[n] = amax[n + 2, opt_path[n + 1], opt_path[n + 2]]

    return (opt_path + 1).astype(np.int32)


@nb.njit(cache=True)
def viterbi_3rd_order_numba(
    notes: np.ndarray, log_initial_prob: np.ndarray, log_transition1_prob: np.ndarray, log_transition2_prob: np.ndarray, log_transition3_prob: np.ndarray, log_output1_prob: np.ndarray, log_output2_prob: np.ndarray, log_output3_prob: np.ndarray, pitch_to_keypos_lut: np.ndarray, hand: int, w1: float, w2: float, w3: float, short_time_cost: float) -> np.ndarray:
    n_obs = len(notes)
    if n_obs < 4:
        return np.zeros(n_obs, dtype=np.int32)
    lp = np.full((5, 5, 5), -np.inf, dtype=np.float64)
    amax = np.zeros((n_obs, 5, 5, 5), dtype=np.int32)
    kp0 = utils.pitch_to_keypos_numba(notes[0]['pitch'], pitch_to_keypos_lut)
    kp1 = utils.pitch_to_keypos_numba(notes[1]['pitch'], pitch_to_keypos_lut)
    kp2 = utils.pitch_to_keypos_numba(notes[2]['pitch'], pitch_to_keypos_lut)
    d_kp1_kp0 = utils.subtract_keypos_numba(kp1[0], kp1[1], kp0[0], kp0[1])
    idx_1_0 = utils.lattice_delta_to_index(d_kp1_kp0[0], d_kp1_kp0[1])
    d_kp2_kp1 = utils.subtract_keypos_numba(kp2[0], kp2[1], kp1[0], kp1[1])
    idx_2_1 = utils.lattice_delta_to_index(d_kp2_kp1[0], d_kp2_kp1[1])
    for kpp in range(5):
        for kp in range(5):
            for k in range(5):
                val = (log_initial_prob[hand, kpp] + log_transition1_prob[hand, kpp, kp] + log_transition1_prob[hand, kp, k] + log_output1_prob[hand, kpp, kp, idx_1_0] + log_output1_prob[hand, kp, k, idx_2_1])
                lp[kpp, kp, k] = val
    for n in range(3, n_obs):
        pre_lp = lp.copy()
        kpn = utils.pitch_to_keypos_numba(notes[n]['pitch'], pitch_to_keypos_lut)
        kpn1 = utils.pitch_to_keypos_numba(notes[n-1]['pitch'], pitch_to_keypos_lut)
        kpn2 = utils.pitch_to_keypos_numba(notes[n-2]['pitch'], pitch_to_keypos_lut)
        kpn3 = utils.pitch_to_keypos_numba(notes[n-3]['pitch'], pitch_to_keypos_lut)
        d_n_n1 = utils.subtract_keypos_numba(kpn[0], kpn[1], kpn1[0], kpn1[1])
        idx1 = utils.lattice_delta_to_index(d_n_n1[0], d_n_n1[1])
        d_n_n2 = utils.subtract_keypos_numba(kpn[0], kpn[1], kpn2[0], kpn2[1])
        idx2 = utils.lattice_delta_to_index(d_n_n2[0], d_n_n2[1])
        d_n_n3 = utils.subtract_keypos_numba(kpn[0], kpn[1], kpn3[0], kpn3[1])
        idx3 = utils.lattice_delta_to_index(d_n_n3[0], d_n_n3[1])
        short_time = abs(notes[n]['ontime'] - notes[n-1]['ontime']) < 0.03
        del_pitch = notes[n]['pitch'] - notes[n-1]['pitch']
        short_time2 = abs(notes[n]['ontime'] - notes[n-2]['ontime']) < 0.03
        del_pitch2 = notes[n]['pitch'] - notes[n-2]['pitch']
        short_time3 = abs(notes[n]['ontime'] - notes[n-3]['ontime']) < 0.03
        del_pitch3 = notes[n]['pitch'] - notes[n-3]['pitch']
        for kpp in range(5):
            for kp in range(5):
                for k in range(5):
                    max_log_prob = -np.inf
                    best_kppp = 0
                    for kppp in range(5):
                        st_cost = 0.0
                        if short_time:
                            if (hand == 0 and (k - kp) * del_pitch < 0) or  (hand == 1 and (k - kp) * del_pitch > 0):
                                st_cost += short_time_cost
                        if short_time2:
                            if (hand == 0 and (k - kpp) * del_pitch2 < 0) or  (hand == 1 and (k - kpp) * del_pitch2 > 0):
                                st_cost += short_time_cost
                        if short_time3:
                            if (hand == 0 and (k - kppp) * del_pitch3 < 0) or  (hand == 1 and (k - kppp) * del_pitch3 > 0):
                                st_cost += short_time_cost
                        log_prob = (pre_lp[kppp, kpp, kp] + log_transition3_prob[hand, kppp, kpp, kp, k] + w1 * log_output1_prob[hand, kp, k, idx1] + w2 * log_output2_prob[hand, kpp, k, idx2] + w3 * log_output3_prob[hand, kppp, k, idx3] + st_cost)
                        if log_prob > max_log_prob:
                            max_log_prob = log_prob
                            best_kppp = kppp
                    lp[kpp, kp, k] = max_log_prob
                    amax[n, kpp, kp, k] = best_kppp
    opt_path = np.zeros(n_obs, dtype=np.int32)
    max_lp = -np.inf
    best_end_state = (0, 0, 0)
    for kpp in range(5):
        for kp in range(5):
            for k in range(5):
                if lp[kpp, kp, k] > max_lp:
                    max_lp = lp[kpp, kp, k]
                    best_end_state = (kpp, kp, k)
    opt_path[n_obs-1] = best_end_state[2]
    opt_path[n_obs-2] = best_end_state[1]
    opt_path[n_obs-3] = best_end_state[0]
    for n in range(n_obs - 4, -1, -1):
        kpp_idx = opt_path[n+1]
        kp_idx  = opt_path[n+2]
        k_idx   = opt_path[n+3]
        opt_path[n] = amax[n+3, kpp_idx, kp_idx, k_idx]
    return (opt_path + 1).astype(np.int32)


def run_viterbi(notes: np.ndarray, params: HMMParameters, hand: int = 0) -> np.ndarray:
    """
    Main entry point. Automatically dispatches to the correct Numba function
    based on the HMM order loaded in parameters.
    """
    # Weights (hardcoded defaults from C++)
    w1_2 = 0.5
    w2_2 = 0.5
    
    w1_3 = 0.667
    w2_3 = 0.5
    w3_3 = 0.2
    
    short_time_cost = -5.0
    if params.order == 3:
        return viterbi_3rd_order_numba(notes, params.log_initial_prob, params.log_transition1_prob, params.log_transition2_prob, params.log_transition3_prob, params.log_output1_prob, params.log_output2_prob, params.log_output3_prob, utils.PITCH_TO_KEYPOS_LUT, hand, w1_3, w2_3, w3_3, short_time_cost)
    else:
        # Note: debug_trace argument removed from call
        return viterbi_2nd_order_numba(notes, params.log_initial_prob, params.log_transition1_prob, params.log_transition2_prob, params.log_output1_prob, params.log_output2_prob, utils.PITCH_TO_KEYPOS_LUT, hand, w1_2, w2_2, short_time_cost)