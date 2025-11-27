import numpy as np
from python import utils

class HMMTrainer:
    def __init__(self, order=2, epsilon=1e-3):
        assert order in [2, 3]
        self.order = order
        self.width_x = 15
        self.n_out = 3 * (2 * self.width_x + 1)

        self.ini_prob = np.full((2, 5), epsilon)
        self.tr_prob1 = np.full((2, 5, 5), epsilon)
        self.out_prob1 = np.full((2, 5, 5, self.n_out), epsilon)

        self.tr_prob2 = np.full((2, 5, 5, 5), epsilon)
        self.out_prob2 = np.full((2, 5, 5, self.n_out), epsilon)

        if self.order == 3:
            self.tr_prob3 = np.full((2, 5, 5, 5, 5), epsilon)
            self.out_prob3 = np.full((2, 5, 5, self.n_out), epsilon)

    def train(self, score_files: list):
        for filepath in score_files:
            notes = utils.load_pig_file(filepath)

            for hand in [0, 1]: # 0=RH, 1=LH
                hand_notes_unord = utils.filter_notes_by_hand(notes, hand)
                hand_notes = utils.apply_time_dep_pitch_order(hand_notes_unord)

                if len(hand_notes) < self.order:
                    continue

                f0 = hand_notes[0]['finger']
                if 1 <= abs(f0) <= 5:
                    self.ini_prob[hand, abs(f0) - 1] += 1

                for n in range(1, len(hand_notes)):
                    fn = hand_notes[n]['finger']
                    fn1 = hand_notes[n-1]['finger']
                    if not (1 <= abs(fn) <= 5 and 1 <= abs(fn1) <= 5):
                        continue

                    self.tr_prob1[hand, abs(fn1) - 1, abs(fn) - 1] += 1

                    kp_n = utils.pitch_to_keypos(hand_notes[n]['pitch'])
                    kp_n1 = utils.pitch_to_keypos(hand_notes[n-1]['pitch'])
                    delta1 = utils.subtract_keypos(kp_n, kp_n1)
                    idx1 = utils.lattice_delta_to_index(delta1[0], delta1[1])
                    self.out_prob1[hand, abs(fn1) - 1, abs(fn) - 1, idx1] += 1

                    if n >= 2:
                        fn2 = hand_notes[n-2]['finger']
                        if not (1 <= abs(fn2) <= 5):
                            continue
                        self.tr_prob2[hand, abs(fn2) - 1, abs(fn1) - 1, abs(fn) - 1] += 1

                        kp_n2 = utils.pitch_to_keypos(hand_notes[n-2]['pitch'])
                        delta2 = utils.subtract_keypos(kp_n, kp_n2)
                        idx2 = utils.lattice_delta_to_index(delta2[0], delta2[1])
                        self.out_prob2[hand, abs(fn2) - 1, abs(fn) - 1, idx2] += 1

                    if self.order == 3 and n >= 3:
                        fn3 = hand_notes[n-3]['finger']
                        if not (1 <= abs(fn3) <= 5):
                            continue
                        self.tr_prob3[hand, abs(fn3) - 1, abs(fn2) - 1, abs(fn1) - 1, abs(fn) - 1] += 1

                        kp_n3 = utils.pitch_to_keypos(hand_notes[n-3]['pitch'])
                        delta3 = utils.subtract_keypos(kp_n, kp_n3)
                        idx3 = utils.lattice_delta_to_index(delta3[0], delta3[1])
                        self.out_prob3[hand, abs(fn3) - 1, abs(fn) - 1, idx3] += 1

        self._normalize_counts()

    def _normalize_counts(self):
        for h in range(2):
            self.ini_prob[h, :] /= np.sum(self.ini_prob[h, :])
            for i in range(5):
                self.tr_prob1[h, i, :] /= np.sum(self.tr_prob1[h, i, :])
                for j in range(5):
                    self.out_prob1[h, i, j, :] /= np.sum(self.out_prob1[h, i, j, :])
                    self.tr_prob2[h, i, j, :] /= np.sum(self.tr_prob2[h, i, j, :])
                    self.out_prob2[h, i, j, :] /= np.sum(self.out_prob2[h, i, j, :])
                    if self.order == 3:
                        for k in range(5):
                            self.tr_prob3[h, i, j, k, :] /= np.sum(self.tr_prob3[h, i, j, k, :])
                        self.out_prob3[h, i, j, :] /= np.sum(self.out_prob3[h, i, j, :])

    def save_parameters(self, filepath):
        with open(filepath, 'w') as f:
            f.write("### Initial Prob Right\n")
            f.write('\t'.join(map(str, self.ini_prob[0])) + '\n')
            f.write("### Initial Prob Left\n")
            f.write('\t'.join(map(str, self.ini_prob[1])) + '\n')

            f.write("### Transition Prob Right\n")
            for row in self.tr_prob1[0]:
                f.write('\t'.join(map(str, row)) + '\n')
            f.write("### Transition Prob Left\n")
            for row in self.tr_prob1[1]:
                f.write('\t'.join(map(str, row)) + '\n')

            f.write("### Transition Prob 2nd Right\n")
            for i in range(5):
                for j in range(5):
                    f.write('\t'.join(map(str, self.tr_prob2[0, i, j])) + '\n')
            f.write("### Transition Prob 2nd Left\n")
            for i in range(5):
                for j in range(5):
                    f.write('\t'.join(map(str, self.tr_prob2[1, i, j])) + '\n')

            if self.order == 3:
                f.write("### Transition Prob 3rd Right\n")
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            f.write('\t'.join(map(str, self.tr_prob3[0, i, j, k])) + '\n')
                f.write("### Transition Prob 3rd Left\n")
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            f.write('\t'.join(map(str, self.tr_prob3[1, i, j, k])) + '\n')

            f.write("### Output Prob Right\n")
            for i in range(5):
                for j in range(5):
                    f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob1[0, i, j])) + '\n')
            f.write("### Output Prob Left\n")
            for i in range(5):
                for j in range(5):
                    f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob1[1, i, j])) + '\n')

            f.write("### Output Prob 2nd Right\n")
            for i in range(5):
                for j in range(5):
                    f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob2[0, i, j])) + '\n')
            f.write("### Output Prob 2nd Left\n")
            for i in range(5):
                for j in range(5):
                    f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob2[1, i, j])) + '\n')

            if self.order == 3:
                f.write("### Output Prob 3rd Right\n")
                for i in range(5):
                    for j in range(5):
                        f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob3[0, i, j])) + '\n')
                f.write("### Output Prob 3rd Left\n")
                for i in range(5):
                    for j in range(5):
                        f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob3[1, i, j])) + '\n')
