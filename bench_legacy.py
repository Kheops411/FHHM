import os
import glob
import numpy as np
from collections import defaultdict
import time

from python import utils
from python import evaluate
# Import de l'adaptateur
from python import legacy_adapter

SCORES_DIR = os.path.join('.', 'scores')

def group_scores_by_piece(scores_dir):
    pattern = os.path.join(scores_dir, '*_fingering.txt')
    files = glob.glob(pattern)
    groups = defaultdict(list)
    for f_path in files:
        piece_id = os.path.basename(f_path).split('-')[0]
        groups[piece_id].append(f_path)
    return groups

def run_legacy_benchmark():
    groups = group_scores_by_piece(SCORES_DIR)
    print(f"Trouvé {len(groups)} pièces. Lancement du Benchmark Legacy...")
    
    metrics = {'High': [], 'Soft': [], 'Recomb': []}
    start_time = time.time()

    for i, (piece_id, file_paths) in enumerate(groups.items()):
        # Chargement GT (Exactement comme HMM)
        gt_notes_list = []
        for f in file_paths:
            raw = utils.load_pig_file(f)
            # CRITIQUE : On applique le même tri que pour le HMM
            ordered = utils.apply_time_dep_pitch_order(raw)
            gt_notes_list.append(ordered)
        
        for idx_file, input_notes_gt in enumerate(gt_notes_list):
            est_notes = input_notes_gt.copy()
            
            for hand in [0, 1]: # 0=Right, 1=Left
                # Masque
                if hand == 0: mask = est_notes['finger'] > 0
                else:         mask = est_notes['finger'] < 0
                
                hand_notes = est_notes[mask]
                
                if len(hand_notes) == 0: continue
                
                # --- APPEL LEGACY VIA ADAPTATEUR ---
                # On passe 'M' comme taille par défaut
                pred_fingers = legacy_adapter.run_legacy_algorithm(hand_notes, hand_side=hand, hand_size='M')
                # -----------------------------------

                if hand == 1:
                    est_notes['finger'][mask] = -pred_fingers
                else:
                    est_notes['finger'][mask] = pred_fingers

            # Calcul métriques (Exactement comme HMM)
            n = len(est_notes)
            if n == 0: continue
            
            err_high = evaluate.multi_gt_error(gt_notes_list, est_notes, 1, 10000, 10000)
            err_soft = evaluate.multi_gt_error(gt_notes_list, est_notes, 1, 0, 0)
            err_rec  = evaluate.multi_gt_error(gt_notes_list, est_notes, 1, 1, 10000)

            metrics['High'].append((n - err_high) / n)
            metrics['Soft'].append((n - err_soft) / n)
            metrics['Recomb'].append((n - err_rec) / n)

        if (i+1) % 10 == 0: print(f"Legacy: {i+1}/{len(groups)}...")

    duration = time.time() - start_time
    print(f"\nRésultats LEGACY (Old School) - Temps: {duration:.2f}s")
    print("-" * 40)
    print(f"M_high (Strict)      : {np.mean(metrics['High']) * 100:.2f}%")
    print(f"M_soft (Relaxed)     : {np.mean(metrics['Soft']) * 100:.2f}%")
    print(f"M_rec  (Recombined)  : {np.mean(metrics['Recomb']) * 100:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    run_legacy_benchmark()