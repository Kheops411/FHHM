import os
import sys
import numpy as np

# Assurez-vous que xml_parser_v2.py est dans le même dossier
# ou que son chemin est dans PYTHONPATH.
try:
    from xml_parser_v3 import MusicXMLParser, Hand
except ImportError:
    print("Erreur: Le fichier 'xml_parser_v2.py' est introuvable.")
    sys.exit(1)

# Configuration
XML_FILE_PATH = r".\tests\resources\test2.xml"

def display_hand_data(hand_name: str, hand_data: dict):
    """Affiche les données d'une main dans un format tabulaire."""
    
    print(f"--- Données pour la main {hand_name} ---")
    
    num_notes = len(hand_data['pitch'])
    if num_notes == 0:
        print("Aucune note pour cette main.")
        return

    # En-tête du tableau
    print(f"{'Idx':<4} | {'Onset (s)':<10} | {'Pitch':<6} | {'Duration (s)':<13} | {'Velocity':<9} | {'Finger GT':<9}")
    print("-" * 75)

    # Récupération des vues NumPy
    # C'est ainsi qu'un algorithme Numba/Numpy accèderait aux données
    indices_originaux = hand_data['indices']
    onsets = hand_data['onset']
    pitches = hand_data['pitch']
    durations = hand_data['duration']
    velocities = hand_data['velocity']
    fingers_gt = hand_data['finger_gt']
    
    # Itération sur les notes de la vue triée
    for i in range(num_notes):
        # Les algorithmes travailleraient directement sur les tableaux (ex: onsets[i])
        print(f"{i:<4} | {onsets[i]:<10.4f} | {pitches[i]:<6} | {durations[i]:<13.4f} | {velocities[i]:<9} | {fingers_gt[i]:<9}")
    
    print(f"\nTotal: {num_notes} notes pour la main {hand_name}.")


def main():
    """Point d'entrée principal du script de démonstration."""
    
    if not os.path.exists(XML_FILE_PATH):
        print(f"Erreur: Le fichier '{XML_FILE_PATH}' n'a pas été trouvé.")
        sys.exit(1)
        
    print(f"1. Parsing du fichier '{XML_FILE_PATH}' avec xml_parser_v2...")
    parser = MusicXMLParser(XML_FILE_PATH)
    score_data = parser.parse()
    print(f"   ✓ Parsing terminé. {score_data.size} événements de note chargés dans ScoreData.\n")
    
    print("2. Extraction des vues par main (méthode 'get_hand_view')...")
    print("   Ces vues sont triées par temps et prêtes pour les algorithmes.\n")
    
    # --- Utilisation par un algorithme pour la main droite ---
    # L'algorithme reçoit ce dictionnaire de vues NumPy
    right_hand_data = score_data.get_hand_view(Hand.RIGHT)
    display_hand_data("Droite", right_hand_data)
    
    print("\n" + "="*75 + "\n")
    
    # --- Utilisation par un algorithme pour la main gauche ---
    left_hand_data = score_data.get_hand_view(Hand.LEFT)
    display_hand_data("Gauche", left_hand_data)


if __name__ == "__main__":
    # Pour un affichage plus propre des grands tableaux NumPy
    np.set_printoptions(threshold=10) 
    main()