import numpy as np
from python import utils

def multi_gt_error(fins_gt: list, fin_est, subst_cost, soft_switch_cost, hard_switch_cost):
    """
    Calcule le coût d'erreur minimal en recombinant plusieurs vérités terrain (GT).
    Gère l'alignement des notes via 'original_idx' pour supporter les cas où
    la répartition Main Droite / Main Gauche diffère entre les pianistes.
    """
    n_gt = len(fins_gt)
    total_cost = 0.0

    # Pré-calcul des dictionnaires de recherche pour chaque GT
    # {original_idx: finger} pour chaque GT
    # Cela permet un accès O(1) pour aligner les notes
    gt_lookups = []
    for gt in fins_gt:
        gt_lookups.append({row['original_idx']: row['finger'] for row in gt})

    for hand in [0, 1]:
        # 1. On définit la séquence "Cible" (Estimate)
        est_hand_notes = utils.filter_notes_by_hand(fin_est, hand)
        est_sequence = est_hand_notes['finger']
        
        if len(est_sequence) == 0:
            continue

        # IDs des notes que l'on est en train d'évaluer
        target_ids = est_hand_notes['original_idx']
        len_seq = len(est_sequence)

        # 2. On construit les séquences GT alignées sur ces IDs
        gt_sequences = np.zeros((n_gt, len_seq), dtype=np.int32)
        
        for z in range(n_gt):
            lookup = gt_lookups[z]
            # Pour chaque note de la séquence cible, on cherche le doigt correspondant dans ce GT.
            # Si le GT a joué cette note avec l'autre main, le doigt aura un signe opposé (ex: -1 vs 1).
            # Cela créera naturellement un coût de substitution, ce qui est correct.
            # Si la note est absente (cas rare/impossible dans PIG complet), on met 0.
            for i, original_idx in enumerate(target_ids):
                gt_sequences[z, i] = lookup.get(original_idx, 0)

        # 3. Algorithme de Programmation Dynamique (Viterbi sur les GTs)
        
        # Initialisation
        cost = np.zeros(n_gt)
        for z in range(n_gt):
            # Coût initial = substitution simple
            cost[z] = subst_cost if est_sequence[0] != gt_sequences[z, 0] else 0

        # Récurrence
        for n in range(1, len_seq):
            pre_cost = cost.copy()
            
            # Pour chaque GT 'z' courant
            for z in range(n_gt):
                min_trans_cost = pre_cost[z] # Coût si on reste sur le même GT (switch cost = 0 implicite ou non ?)
                # Le code original C++ implique que rester sur le même pianiste a un coût de switch de 0
                # Seulement si on change, on paie soft/hard.
                
                # On cherche le meilleur prédécesseur 'zp'
                for zp in range(n_gt):
                    if zp == z: 
                        continue

                    # Le coût de transition dépend de la cohérence entre les pianistes précédents
                    # Si zp et z ont joué le DOIGT PRÉCÉDENT de la même façon, c'est un "Soft Switch"
                    is_consistent = (gt_sequences[zp, n-1] == gt_sequences[z, n-1])
                    switch_cost = soft_switch_cost if is_consistent else hard_switch_cost

                    if pre_cost[zp] + switch_cost < min_trans_cost:
                        min_trans_cost = pre_cost[zp] + switch_cost

                # Coût final pour arriver à l'état z à l'étape n
                match_cost = subst_cost if est_sequence[n] != gt_sequences[z, n] else 0
                cost[z] = min_trans_cost + match_cost

        total_cost += np.min(cost)

    return total_cost

def calculate_simple_match_rate(gt_file: str, est_file: str):
    # Note: Cette fonction simple suppose une correspondance 1:1 stricte et naïve.
    # Elle est moins robuste que multi_gt_error.
    gt_notes = utils.load_pig_file(gt_file)
    est_notes = utils.load_pig_file(est_file)

    # Alignement par original_idx pour être sûr
    gt_dict = {row['original_idx']: row['finger'] for row in gt_notes}
    match_count = 0
    total = len(est_notes)
    
    for row in est_notes:
        oid = row['original_idx']
        if oid in gt_dict and gt_dict[oid] == row['finger']:
            match_count += 1
            
    return match_count / total if total > 0 else 0


def calculate_metrics(gt_files: list, est_file: str):
    # Chargement
    fins_gt = [utils.load_pig_file(f) for f in gt_files]
    fin_est = utils.load_pig_file(est_file)
    
    # Application de l'ordre temporel (IMPORTANT pour la cohérence interne des séquences)
    fins_gt = [utils.apply_time_dep_pitch_order(f) for f in fins_gt]
    fin_est = utils.apply_time_dep_pitch_order(fin_est)

    n_notes = len(fin_est)
    if n_notes == 0:
        return {"General": 0, "Highest": 0, "Soft": 0, "Recomb": 0}

    # Calculs robustes
    err_high = multi_gt_error(fins_gt, fin_est, 1, 10000, 10000)
    err_soft = multi_gt_error(fins_gt, fin_est, 1, 0, 0)
    err_rec = multi_gt_error(fins_gt, fin_est, 1, 1, 10000)

    m_high = (n_notes - err_high) / n_notes
    m_soft = (n_notes - err_soft) / n_notes
    m_recomb = (n_notes - err_rec) / n_notes

    # M_Gen (Moyenne simple par paires)
    # On réutilise la logique robuste d'alignement
    match_rates = []
    # On crée un dict pour l'estimation
    est_lookup = {row['original_idx']: row['finger'] for row in fin_est}
    
    for gt in fins_gt:
        matches = 0
        for row in gt:
            if row['original_idx'] in est_lookup and est_lookup[row['original_idx']] == row['finger']:
                matches += 1
        match_rates.append(matches / n_notes)
        
    m_gen = np.mean(match_rates)

    return {"General": m_gen, "Highest": m_high, "Soft": m_soft, "Recomb": m_recomb}