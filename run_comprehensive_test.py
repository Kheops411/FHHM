import os
import sys
import numpy as np
import argparse

# Adjust the path to import from the 'python' directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python'))

import utils
import model
import evaluate

def run_tests(scores_dir, param_file):
    # --- Configuration ---
    SCORES_DIR = scores_dir
    PARAM_FILE = param_file

    # Vérification des chemins
    if not os.path.exists(SCORES_DIR):
        print(f"Erreur: Le dossier {SCORES_DIR} n'existe pas.")
        return
    if not os.path.exists(PARAM_FILE):
        print(f"Erreur: Le fichier de paramètres {PARAM_FILE} n'existe pas.")
        return

    print("=== Chargement du Modèle HMM ===")
    hmm = model.HMMModel(w1=0.5, w2=0.5, lam1=0.0, short_time_cost=-5.0)

    try:
        hmm.load_cpp_parameters(PARAM_FILE)
        print("Paramètres chargés avec succès.\n")
    except Exception as e:
        print(f"Erreur lors du chargement des paramètres : {e}")
        return

    print(f"=== Analyse des partitions dans {SCORES_DIR} ===\n")

    try:
        all_scores = utils.parse_all_scores(SCORES_DIR)
    except Exception as e:
        print(f"Erreur critique lors du parsing : {e}")
        return

    if not all_scores:
        print("Aucun fichier .txt valide trouvé.")
        return

    total_notes = 0
    total_matches = 0
    file_count = 0

    for filename, notes_data in sorted(all_scores.items()):
        file_count += 1
        print(f"Traitement : {filename}")

        notes_rh = notes_data[notes_data['hand'] == 0]
        notes_lh = notes_data[notes_data['hand'] == 1]

        # This part of the original script was flawed.
        # We need to process each hand's notes and then merge the results
        # back into a single sequence that matches the order of notes_data.

        estimated_fingers = np.zeros(len(notes_data), dtype=int)

        if len(notes_rh) > 0:
            rh_indices = np.where(notes_data['hand'] == 0)[0]
            est_rh = model.viterbi_decode(
                notes_rh,
                hmm.transition_matrix_2nd[0], hmm.transition_matrix_1st[0], hmm.initial_probabilities[0],
                hmm.output_prob_1st[0], hmm.output_prob_2nd[0],
                hmm.widthX, hmm.w1, hmm.w2, hmm.short_time_cost,
                hand=0
            )
            estimated_fingers[rh_indices] = est_rh

        if len(notes_lh) > 0:
            lh_indices = np.where(notes_data['hand'] == 1)[0]
            est_lh = model.viterbi_decode(
                notes_lh,
                hmm.transition_matrix_2nd[1], hmm.transition_matrix_1st[1], hmm.initial_probabilities[1],
                hmm.output_prob_1st[1], hmm.output_prob_2nd[1],
                hmm.widthX, hmm.w1, hmm.w2, hmm.short_time_cost,
                hand=1
            )
            estimated_fingers[lh_indices] = est_lh

        ground_truth = notes_data['finger']

        if len(ground_truth) > 0:
            match_rate = evaluate.calculate_match_rate(ground_truth, estimated_fingers)

            matches = np.sum(ground_truth == estimated_fingers)
            total_matches += matches
            total_notes += len(ground_truth)

            print(f"  -> Notes: {len(ground_truth)} (RH: {len(notes_rh)}, LH: {len(notes_lh)})")
            print(f"  -> Précision : {match_rate:.2f}%")
        else:
            print("  -> (Fichier vide ou format incorrect)")

        print("-" * 40)

    print("\n=== RÉSULTATS GLOBAUX ===")
    print(f"Fichiers traités : {file_count}")
    if total_notes > 0:
        global_accuracy = (total_matches / total_notes) * 100
        print(f"Notes totales    : {total_notes}")
        print(f"Matchs corrects  : {total_matches}")
        print(f"Précision Moyenne: {global_accuracy:.2f}%")
    else:
        print("Aucune note traitée.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive tests for the HMM fingering model.")
    parser.add_argument('--scores_dir', type=str, default='./scores', help='Directory containing the score files.')
    parser.add_argument('--param_file', type=str, default='./cpp/param_FHMM2.txt', help='Path to the HMM parameter file.')
    args = parser.parse_args()
    run_tests(args.scores_dir, args.param_file)
