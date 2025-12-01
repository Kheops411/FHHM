import numpy as np
from . import utils  # Correction de l'import

def multi_gt_error(fins_gt: list, fin_est, subst_cost, soft_switch_cost, hard_switch_cost):
    """
    Computes the minimal edit distance against multiple Ground Truths using Dynamic Programming.
    Corresponds to the "Recombination Match Rate" logic in the paper.
    """
    n_gt = len(fins_gt)
    total_cost = 0.0
    
    # Pre-build lookups for speed: original_idx -> finger (int)
    gt_lookups = [{row['original_idx']: row['finger'] for row in gt} for gt in fins_gt]

    for hand in [0, 1]: # 0=Right, 1=Left
        # Filter logic: Right hand (finger > 0), Left hand (finger < 0)
        # Note: (1 - 2*0) = 1  => finger * 1 > 0  => finger > 0
        #       (1 - 2*1) = -1 => finger * -1 > 0 => finger < 0
        mask = (fin_est['finger'] * (1 - 2 * hand)) > 0
        est_hand_notes = fin_est[mask]
        
        est_sequence = est_hand_notes['finger']
        len_seq = len(est_sequence)
        
        if len_seq == 0:
            continue

        target_ids = est_hand_notes['original_idx']
        
        # Build GT sequences aligned to the Estimate's original_indices
        gt_sequences = np.zeros((n_gt, len_seq), dtype=np.int32)
        
        for z in range(n_gt):
            lookup = gt_lookups[z]
            for i, original_idx in enumerate(target_ids):
                # Default to 0 if note missing in GT (will cause mismatch)
                gt_sequences[z, i] = lookup.get(original_idx, 0)

        # Initialize DP Cost table (shape: n_gt)
        cost = np.zeros(n_gt, dtype=np.float64)
        for z in range(n_gt):
            cost[z] = subst_cost if est_sequence[0] != gt_sequences[z, 0] else 0.0

        # DP Loop
        for n in range(1, len_seq):
            pre_cost = cost.copy()
            for z in range(n_gt):
                # Find best transition from previous step (zp) to current step (z)
                min_trans_cost = np.inf
                
                for zp in range(n_gt):
                    # Cost to switch from GT zp to GT z
                    if zp == z:
                        switch_cost = 0.0
                    else:
                        # Consistent: Did GT zp and GT z agree on the PREVIOUS note?
                        is_consistent = (gt_sequences[zp, n-1] == gt_sequences[z, n-1])
                        switch_cost = soft_switch_cost if is_consistent else hard_switch_cost
                    
                    c = pre_cost[zp] + switch_cost
                    if c < min_trans_cost:
                        min_trans_cost = c

                # Add substitution cost for current note
                match_cost = subst_cost if est_sequence[n] != gt_sequences[z, n] else 0.0
                cost[z] = min_trans_cost + match_cost

        total_cost += np.min(cost)

    return total_cost


def calculate_metrics(gt_files: list, est_file: str):
    fins_gt = [utils.load_pig_file(f) for f in gt_files]
    fin_est = utils.load_pig_file(est_file)
    
    # Note: Evaluation is strictly done on file order, NO TimeDepPitchOrder reordering.
    
    n_notes = len(fin_est)
    if n_notes == 0:
        return {"General": 0.0, "Highest": 0.0, "Soft": 0.0, "Recomb": 0.0}

    # --- "General" Metric: Average Pairwise Match Rate ---
    # Based on strict string comparison (e.g., "4_1" != "4")
    match_rates = []
    
    # Create lookup for estimate once
    est_lookup = {row['original_idx']: row['finger_str'] for row in fin_est}
    
    for gt_notes in fins_gt:
        gt_lookup = {row['original_idx']: row['finger_str'] for row in gt_notes}

        matches = 0
        # Intersection of note IDs ensures we only compare present notes
        common_ids = set(est_lookup.keys()) & set(gt_lookup.keys())

        for oid in common_ids:
            if est_lookup[oid] == gt_lookup[oid]:
                matches += 1

        # Normalized by Total ESTIMATED notes (as per C++ logic)
        match_rates.append(matches / n_notes)

    m_gen = np.mean(match_rates) if match_rates else 0.0

    # --- Advanced Metrics (Integer based) ---
    # 1. Highest Match Rate: Best single GT (Hard switch cost = inf)
    err_high = multi_gt_error(fins_gt, fin_est, subst_cost=1, soft_switch_cost=10000, hard_switch_cost=10000)
    
    # 2. Soft Match Rate: Switch allowed anytime (Switch cost = 0)
    err_soft = multi_gt_error(fins_gt, fin_est, subst_cost=1, soft_switch_cost=0, hard_switch_cost=0)
    
    # 3. Recombination Rate: Switch allowed only if GTs agree (Soft=1, Hard=inf)
    # Note: Soft cost=1 penalizes switching slightly to prefer continuity unless necessary? 
    # Actually, in the paper/C++, soft cost is often small or 0 for recombination, 
    # but the provided snippet uses 1. We stick to the snippet provided: (1, 1, 10000).
    err_rec = multi_gt_error(fins_gt, fin_est, subst_cost=1, soft_switch_cost=1, hard_switch_cost=10000)

    m_high = (n_notes - err_high) / n_notes
    m_soft = (n_notes - err_soft) / n_notes
    m_recomb = (n_notes - err_rec) / n_notes

    return {
        "General": m_gen, 
        "Highest": m_high, 
        "Soft": m_soft, 
        "Recomb": m_recomb
    }