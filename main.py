import os
import glob
import numpy as np
from collections import defaultdict
import time

# Importation des modules du projet
# Assurez-vous que ce fichier est à la racine, au même niveau que le dossier 'python'
from python import utils
from python import model
from python import evaluate

# --- Configuration des chemins ---
SCORES_DIR = os.path.join('.', 'scores')
PARAM_FILE_HMM2 = os.path.join('.', 'cpp', 'Code', 'param_FHMM2.txt')#'./param_FHMM2_new.txt'
PARAM_FILE_HMM3 = os.path.join('.', 'cpp', 'Code', 'param_FHMM3.txt')#'./param_FHMM3_new.txt'

def group_scores_by_piece(scores_dir):
    """
    Regroupe les fichiers PIG par identifiant de morceau.
    Exemple: '001-1_fingering.txt' et '001-2_fingering.txt' -> Groupe '001'
    """
    pattern = os.path.join(scores_dir, '*_fingering.txt')
    files = glob.glob(pattern)
    groups = defaultdict(list)
    
    for f_path in files:
        filename = os.path.basename(f_path)
        # On suppose que l'ID est la partie avant le premier tiret (ex: "001" dans "001-1...")
        # Ou parfois tout ce qui est avant le dernier tiret si le format varie.
        # Basé sur le README PIG standard : "PieceID-PerformerID_fingering.txt"
        piece_id = filename.split('-')[0]
        groups[piece_id].append(f_path)
    
    return groups

def run_evaluation_for_model(param_file, score_groups, model_name):
    print(f"\n{'='*60}")
    print(f"Démarrage de l'évaluation pour : {model_name}")
    print(f"Fichier paramètres : {param_file}")
    print(f"{'='*60}")

    # 1. Chargement des paramètres HMM
    try:
        hmm_params = model.HMMParameters(param_file)
        print(f"Paramètres chargés. Ordre détecté : {hmm_params.order}")
    except FileNotFoundError:
        print(f"ERREUR: Fichier de paramètres introuvable: {param_file}")
        return

    # Stockage des résultats globaux
    metrics = {
        'General': [],
        'High': [],
        'Soft': [],
        'Recomb': []
    }

    total_pieces = len(score_groups)
    start_time = time.time()

    # 2. Boucle sur chaque groupe de morceaux (Pièce)
    for i, (piece_id, file_paths) in enumerate(score_groups.items()):
        # Chargement de toutes les Vérités Terrain (Ground Truths) pour ce morceau
        # ATTENTION : Il faut appliquer l'ordre spécifique (TimeDepPitchOrder) 
        # pour que les indices correspondent à ceux de la prédiction Viterbi.
        gt_notes_list = []
        for f in file_paths:
            raw = utils.load_pig_file(f)
            ordered = utils.apply_time_dep_pitch_order(raw)
            gt_notes_list.append(ordered)
        
        # On évalue chaque fichier du groupe individuellement comme "Entrée"
        # et on le compare à l'ensemble des GTs (y compris lui-même, protocole standard PIG)
        for idx_file, input_notes_gt in enumerate(gt_notes_list):
            
            # Copie pour stocker la prédiction (Est)
            # On part des notes triées/ordonnées
            est_notes = input_notes_gt.copy()
            
            # --- Prédiction Main Droite (0) et Main Gauche (1) ---
            for hand in [0, 1]:
                # Extraction des notes de la main concernée
                # Note: filter_notes_by_hand renvoie un sous-ensemble.
                # Pour réinsérer les doigts, il faut retrouver les indices.
                
                # Masque booléen pour identifier les notes de la main courante dans le tableau trié
                if hand == 0:
                    hand_mask = est_notes['finger'] > 0
                else:
                    hand_mask = est_notes['finger'] < 0
                
                hand_notes_subset = est_notes[hand_mask]
                
                if len(hand_notes_subset) == 0:
                    continue

                # Exécution Viterbi
                # Le modèle renvoie des entiers [1..5]
                pred_fingers = model.run_viterbi(hand_notes_subset, hmm_params, hand=hand)
                
                # Mise à jour des notes estimées
                if hand == 1:
                    # Pour la main gauche, le format PIG attend des nombres négatifs
                    est_notes['finger'][hand_mask] = -pred_fingers.astype(np.int32)
                else:
                    est_notes['finger'][hand_mask] = pred_fingers.astype(np.int32)

            # --- Calcul des Métriques ---
            n_notes = len(est_notes)
            if n_notes == 0:
                continue

            # Appel direct à evaluate.multi_gt_error (in-memory)
            # Rappel signature: multi_gt_error(fins_gt, fin_est, subst_cost, soft_switch, hard_switch)
            
            err_high = evaluate.multi_gt_error(gt_notes_list, est_notes, 1, 10000, 10000)
            err_soft = evaluate.multi_gt_error(gt_notes_list, est_notes, 1, 0, 0)
            err_rec  = evaluate.multi_gt_error(gt_notes_list, est_notes, 1, 1, 10000)

            m_high = (n_notes - err_high) / n_notes
            m_soft = (n_notes - err_soft) / n_notes
            m_rec  = (n_notes - err_rec)  / n_notes

            est_lookup = {row['original_idx']: row['finger'] for row in est_notes}
            rates = []
            for gt in gt_notes_list:
                matches = 0
                for row in gt:
                    # On compare via original_idx pour l'alignement
                    if row['original_idx'] in est_lookup and est_lookup[row['original_idx']] == row['finger']:
                        matches += 1
                if n_notes > 0:
                    rates.append(matches / n_notes)
                else:
                    rates.append(0)
            m_gen = np.mean(rates) if rates else 0
            # ---------------------------------------------
            metrics['General'].append(m_gen)
            metrics['High'].append(m_high)
            metrics['Soft'].append(m_soft)
            metrics['Recomb'].append(m_rec)

        if (i + 1) % 10 == 0:
            print(f"Traité {i + 1}/{total_pieces} pièces...")

    duration = time.time() - start_time
    
    # 3. Affichage des Moyennes
    print(f"\nRésultats pour {model_name} (Temps: {duration:.2f}s)")
    print("-" * 40)
    print(f"M_gen  (Standard)    : {np.mean(metrics['General']) * 100:.2f}%")
    print(f"M_high (Strict)      : {np.mean(metrics['High']) * 100:.2f}%")
    print(f"M_soft (Relaxed)     : {np.mean(metrics['Soft']) * 100:.2f}%")
    print(f"M_rec  (Recombined)  : {np.mean(metrics['Recomb']) * 100:.2f}%")
    print("-" * 40)


def main():
    # Vérification des dossiers
    if not os.path.exists(SCORES_DIR):
        print(f"ERREUR: Le dossier {SCORES_DIR} n'existe pas.")
        return

    # Groupement des fichiers
    groups = group_scores_by_piece(SCORES_DIR)
    print(f"Trouvé {len(groups)} pièces uniques dans {SCORES_DIR} (Total fichiers: {sum(len(g) for g in groups.values())})")
    
    if not groups:
        print("Aucun fichier PIG (*_fingering.txt) trouvé.")
        return

    # Exécution HMM Ordre 2
    if os.path.exists(PARAM_FILE_HMM2):
        run_evaluation_for_model(PARAM_FILE_HMM2, groups, "HMM Order 2")
    else:
        print(f"Skipping HMM2: {PARAM_FILE_HMM2} introuvable.")

    # Exécution HMM Ordre 3
    if os.path.exists(PARAM_FILE_HMM3):
        run_evaluation_for_model(PARAM_FILE_HMM3, groups, "HMM Order 3")
    else:
        print(f"Skipping HMM3: {PARAM_FILE_HMM3} introuvable.")

if __name__ == "__main__":
    main()