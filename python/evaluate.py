import numpy as np
from python import utils

def multi_gt_error(fins_gt: list, fin_est, subst_cost, soft_switch_cost, hard_switch_cost):
    n_gt = len(fins_gt)
    total_cost = 0.0

    for hand in [0, 1]:
        est_notes = utils.filter_notes_by_hand(fin_est, hand)
        gt_notes_by_hand = [utils.filter_notes_by_hand(gt, hand) for gt in fins_gt]

        if len(est_notes) == 0:
            continue

        # Create maps from original_idx to finger
        est_map = {note['original_idx']: note['finger'] for note in est_notes}
        gt_maps = [{note['original_idx']: note['finger'] for note in gt_seq} for gt_seq in gt_notes_by_hand]

        # Use the estimated notes' indices as the canonical sequence
        indices_to_compare = sorted(est_map.keys())
        len_seq = len(indices_to_compare)

        cost = np.zeros(n_gt)
        amin = np.zeros((len_seq, n_gt), dtype=int)

        # Initialization (n=0)
        idx = indices_to_compare[0]
        for z in range(n_gt):
            gt_finger = gt_maps[z].get(idx)
            est_finger = est_map.get(idx)
            cost[z] = subst_cost if est_finger != gt_finger else 0

        # Main loop
        for n in range(1, len_seq):
            pre_cost = cost.copy()
            idx = indices_to_compare[n]
            idx_prev = indices_to_compare[n-1]

            est_finger = est_map.get(idx)

            for z in range(n_gt):
                min_trans_cost = pre_cost[z]
                amin[n, z] = z

                for zp in range(n_gt):
                    if zp == z: continue

                    gt_finger_prev_z = gt_maps[z].get(idx_prev)
                    gt_finger_prev_zp = gt_maps[zp].get(idx_prev)

                    switch_cost = soft_switch_cost if gt_finger_prev_zp == gt_finger_prev_z else hard_switch_cost

                    if pre_cost[zp] + switch_cost < min_trans_cost:
                        min_trans_cost = pre_cost[zp] + switch_cost
                        amin[n, z] = zp

                gt_finger = gt_maps[z].get(idx)
                cost[z] = min_trans_cost + (subst_cost if est_finger != gt_finger else 0)

        total_cost += np.min(cost)

    return total_cost


def calculate_simple_match_rate(gt_notes, est_notes):
    # Create maps for alignment
    gt_map = {note['original_idx']: note['finger'] for note in gt_notes}
    est_map = {note['original_idx']: note['finger'] for note in est_notes}

    # Find common indices
    common_indices = gt_map.keys() & est_map.keys()

    if not common_indices:
        return 0.0

    matches = sum(1 for idx in common_indices if gt_map[idx] == est_map[idx])
    return matches / len(common_indices)


def calculate_metrics(gt_files: list, est_file: str):
    fins_gt = [utils.load_pig_file(f) for f in gt_files]
    fin_est = utils.load_pig_file(est_file)

    n_notes = len(fin_est)

    m_high = (n_notes - multi_gt_error(fins_gt, fin_est, 1, 10000, 10000)) / n_notes
    m_soft = (n_notes - multi_gt_error(fins_gt, fin_est, 1, 0, 0)) / n_notes
    m_recomb = (n_notes - multi_gt_error(fins_gt, fin_est, 1, 1, 10000)) / n_notes

    # m_gen requires AveragePairwiseMatchRate
    match_rates = []
    for gt_notes in fins_gt:
        match_rates.append(calculate_simple_match_rate(gt_notes, fin_est))
    m_gen = np.mean(match_rates)

    return {"General": m_gen, "Highest": m_high, "Soft": m_soft, "Recomb": m_recomb}
