import sys
import os
import numpy as np

# Ajout du dossier racine au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python import utils
from python import model

def run_diagnostic():
    score_path = 'scores/126-2_fingering.txt'
    # Utilisation du chemin exact confirmé
    param_path = 'cpp/Code/param_FHMM2.txt'

    if not os.path.exists(score_path):
        print(f"ERREUR: Le fichier score {score_path} est introuvable.")
        return
    if not os.path.exists(param_path):
        print(f"ERREUR: Le fichier param {param_path} est introuvable.")
        return

    print(f"--- DIAGNOSTIC: {score_path} ---")

    # 1. Chargement des Paramètres
    print("1. Chargement des paramètres HMM...")
    try:
        hmm_params = model.HMMParameters(param_path)
        print(f"   -> Paramètres chargés. Ordre détecté: {hmm_params.order}")
    except Exception as e:
        print(f"   -> ERREUR chargement params: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Parsing du PIG
    print("2. Parsing du fichier PIG...")
    all_notes = utils.load_pig_file(score_path)
    print(f"   -> {len(all_notes)} notes lues.")
    
    # 3. Filtrage Main Droite (RH)
    print("3. Filtrage Main Droite (RH)...")
    # On filtre sur 'finger' > 0 (Main Droite)
    rh_notes = all_notes[all_notes['finger'] > 0]
    
    # Fallback si pas de doigtés explicites
    if len(rh_notes) == 0:
        print("   -> Fallback: Filtrage par Channel 0.")
        rh_notes = all_notes[all_notes['channel'] == 0]

    print(f"   -> {len(rh_notes)} notes conservées pour la MD.")

    if len(rh_notes) == 0:
        return

    # 4. Tri Temporel
    print("4. Application de TimeDepPitchOrder...")
    sorted_notes = utils.apply_time_dep_pitch_order(rh_notes)
    
    # --- TRACE INPUT (CORRIGÉ: ontime) ---
    print("\n=== PYTHON INPUT TRACE (First 30 notes) ===")
    print(f"{'IDX':<5} {'ONSET':<10} {'PITCH':<5}")
    limit = min(30, len(sorted_notes))
    for i in range(limit):
        n = sorted_notes[i]
        print(f"{i:<5} {n['ontime']:<10.6f} {n['pitch']:<5}")
    print("===========================================\n")

    # 5. Préparation Données Viterbi
    # Passage en Structured Array -> Arrays simples pour Numba
    # Important: viterbi_2nd_order_numba attend le structured array 'notes' complet
    # car il accède à ['pitch'] et ['ontime'] à l'intérieur.
    lut = utils.PITCH_TO_KEYPOS_LUT

    # Paramètres fixés (Defaults C++ pour FingeringHMM_2nd)
    hand = 0        # 0 = Right Hand
    w1 = 0.5
    w2 = 0.5
    short_time_cost = -5.0

    # 6. Exécution Viterbi
    print("5. Exécution de Viterbi (Ordre 2)...")
    try:
        # Appel avec les bons noms d'attributs de votre classe HMMParameters
        fingers = model.viterbi_2nd_order_numba(
            sorted_notes,                  # Le tableau de notes entier
            hmm_params.log_initial_prob,
            hmm_params.log_transition1_prob,
            hmm_params.log_transition2_prob,
            hmm_params.log_output1_prob,
            hmm_params.log_output2_prob,
            lut,
            hand,
            w1,
            w2,
            short_time_cost
        )
    except Exception as e:
        print(f"   -> CRASH dans Viterbi: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- TRACE OUTPUT ---
    print("\n=== PYTHON OUTPUT TRACE (First 30 steps) ===")
    print(f"STEP PITCH TIME FINGER")
    for i in range(min(30, len(fingers))):
        n = sorted_notes[i]
        print(f"STEP {i} PITCH:{n['pitch']} TIME:{n['ontime']:.6f} FINGER:{fingers[i]}")
    print("===========================================\n")

if __name__ == "__main__":
    run_diagnostic()