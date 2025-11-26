import numpy as np
from python import utils

def multi_gt_error(fins_gt: list, fin_est, subst_cost, soft_switch_cost, hard_switch_cost):
    n_gt = len(fins_gt)
    total_cost = 0.0

    for hand in [0, 1]:
        est_sequence = utils.filter_notes_by_hand(fin_est, hand)['finger']
        gt_sequences = [utils.filter_notes_by_hand(gt, hand)['finger'] for gt in fins_gt]

        if len(est_sequence) == 0:
            continue

        len_seq = len(est_sequence)
        cost = np.zeros(n_gt)
        amin = np.zeros((len_seq, n_gt), dtype=int)

        for z in range(n_gt):
            cost[z] = subst_cost if est_sequence[0] != gt_sequences[z][0] else 0

        for n in range(1, len_seq):
            pre_cost = cost.copy()
            for z in range(n_gt):
                min_trans_cost = pre_cost[z]
                amin[n, z] = z

                for zp in range(n_gt):
                    if zp == z: continue

                    switch_cost = soft_switch_cost if gt_sequences[zp][n-1] == gt_sequences[z][n-1] else hard_switch_cost

                    if pre_cost[zp] + switch_cost < min_trans_cost:
                        min_trans_cost = pre_cost[zp] + switch_cost
                        amin[n, z] = zp

                cost[z] = min_trans_cost + (subst_cost if est_sequence[n] != gt_sequences[z][n] else 0)

        total_cost += np.min(cost)

    return total_cost

def calculate_simple_match_rate(gt_file: str, est_file: str):
    gt_notes = utils.load_pig_file(gt_file)
    est_notes = utils.load_pig_file(est_file)

    gt_fingers = gt_notes['finger']
    est_fingers = est_notes['finger']

    return np.sum(gt_fingers == est_fingers) / len(gt_fingers)


def calculate_metrics(gt_files: list, est_file: str):
    fins_gt = [utils.load_pig_file(f) for f in gt_files]
    fin_est = utils.load_pig_file(est_file)

    n_notes = len(fin_est)

    m_high = (n_notes - multi_gt_error(fins_gt, fin_est, 1, 10000, 10000)) / n_notes
    m_soft = (n_notes - multi_gt_error(fins_gt, fin_est, 1, 0, 0)) / n_notes
    m_recomb = (n_notes - multi_gt_error(fins_gt, fin_est, 1, 1, 10000)) / n_notes

    # m_gen requires AveragePairwiseMatchRate
    match_rates = []
    for gt_file in gt_files:
        match_rates.append(calculate_simple_match_rate(gt_file, est_file))
    m_gen = np.mean(match_rates)

    return {"General": m_gen, "Highest": m_high, "Soft": m_soft, "Recomb": m_recomb}
