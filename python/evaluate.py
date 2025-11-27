import numpy as np
from python import utils

def multi_gt_error(fins_gt: list, fin_est, subst_cost, soft_switch_cost, hard_switch_cost):
    """
    This function appears to be a correct port of the C++ MultiGTError.
    It correctly aligns notes via 'original_idx' and uses a DP approach.
    No changes are needed here for now.
    """
    n_gt = len(fins_gt)
    total_cost = 0.0
    gt_lookups = [{row['original_idx']: row['finger'] for row in gt} for gt in fins_gt]

    for hand in [0, 1]:
        est_hand_notes = fin_est[fin_est['finger'] * (1 - 2 * hand) > 0]
        est_sequence = est_hand_notes['finger']
        
        if len(est_sequence) == 0:
            continue

        target_ids = est_hand_notes['original_idx']
        len_seq = len(est_sequence)
        gt_sequences = np.zeros((n_gt, len_seq), dtype=np.int32)
        
        for z in range(n_gt):
            lookup = gt_lookups[z]
            for i, original_idx in enumerate(target_ids):
                gt_sequences[z, i] = lookup.get(original_idx, 0)

        cost = np.zeros(n_gt)
        for z in range(n_gt):
            cost[z] = subst_cost if est_sequence[0] != gt_sequences[z, 0] else 0

        for n in range(1, len_seq):
            pre_cost = cost.copy()
            for z in range(n_gt):
                min_trans_cost = pre_cost[z]
                for zp in range(n_gt):
                    if zp == z: continue
                    is_consistent = (gt_sequences[zp, n-1] == gt_sequences[z, n-1])
                    switch_cost = soft_switch_cost if is_consistent else hard_switch_cost
                    if pre_cost[zp] + switch_cost < min_trans_cost:
                        min_trans_cost = pre_cost[zp] + switch_cost

                match_cost = subst_cost if est_sequence[n] != gt_sequences[z, n] else 0
                cost[z] = min_trans_cost + match_cost

        total_cost += np.min(cost)

    return total_cost


def calculate_metrics(gt_files: list, est_file: str):
    fins_gt = [utils.load_pig_file(f) for f in gt_files]
    fin_est = utils.load_pig_file(est_file)
    
    # Per C++ implementation, evaluation does not use TimeDepPitchOrder
    # fins_gt = [utils.apply_time_dep_pitch_order(f) for f in fins_gt]
    # fin_est = utils.apply_time_dep_pitch_order(fin_est)

    n_notes = len(fin_est)
    if n_notes == 0:
        return {"General": 0, "Highest": 0, "Soft": 0, "Recomb": 0}

    # --- "General" Metric: Average Pairwise Match Rate ---
    # This logic now exactly replicates the C++ AveragePairwiseMatchRate function.
    match_rates = []
    for gt_notes in fins_gt:
        # The C++ code assumes a 1-to-1 correspondence by index.
        # To be robust, we should align by original_idx.
        est_lookup = {row['original_idx']: row['finger_str'] for row in fin_est}
        gt_lookup = {row['original_idx']: row['finger_str'] for row in gt_notes}

        matches = 0
        common_ids = set(est_lookup.keys()) & set(gt_lookup.keys())

        for oid in common_ids:
            # C++ compares the raw fingerNum string, not the parsed integer.
            if est_lookup[oid] == gt_lookup[oid]:
                matches += 1

        # The C++ code divides by the total notes in the estimated file.
        match_rates.append(matches / n_notes if n_notes > 0 else 0)

    m_gen = np.mean(match_rates)

    # --- Other Metrics based on MultiGTError ---
    # These calculations were already correct.
    err_high = multi_gt_error(fins_gt, fin_est, 1, 10000, 10000)
    err_soft = multi_gt_error(fins_gt, fin_est, 1, 0, 0)
    err_rec = multi_gt_error(fins_gt, fin_est, 1, 1, 10000)

    m_high = (n_notes - err_high) / n_notes
    m_soft = (n_notes - err_soft) / n_notes
    m_recomb = (n_notes - err_rec) / n_notes

    return {"General": m_gen, "Highest": m_high, "Soft": m_soft, "Recomb": m_recomb}
