import sys
import numpy as np
from typing import List

# Import de votre parser XML (assurez-vous que le fichier s'appelle xml_parser.py)
from xml_parser import process_musicxml, PlayedNote, Hand

# Import du moteur HMM
from python import utils, model

def hmm_fingering_adapter(notes: List[PlayedNote]) -> List[PlayedNote]:
    """
    Connecte le parser XML au moteur HMM.
    Cette fonction respecte la signature attendue par process_musicxml.
    """
    if not notes:
        return notes

    # 1. Chargement des paramètres HMM (Ordre 3 recommandé)
    # Assurez-vous que le fichier param existe (généré à l'étape précédente)
    try:
        params = model.HMMParameters("param_FHMM3_new.txt")
        print(f"   [HMM] Modèle Ordre {params.order} chargé.")
    except FileNotFoundError:
        print("   [ERREUR] Paramètres introuvables. Lancez l'entraînement d'abord.")
        return notes

    # 2. Conversion : List[PlayedNote] -> Numpy PIG Format
    # On crée un tableau vide avec le DTYPE attendu par le moteur
    num_notes = len(notes)
    pig_data = np.zeros(num_notes, dtype=utils.NOTE_DTYPE)
    
    # Mapping pour garder la trace de l'objet original
    # index_numpy -> objet PlayedNote
    note_mapping = {}

    for i, pn in enumerate(notes):
        # On utilise le champ 'original_idx' pour stocker l'index temporaire
        pig_data[i]['original_idx'] = i 
        pig_data[i]['ontime'] = pn.onset
        pig_data[i]['offtime'] = pn.offset
        pig_data[i]['pitch'] = pn.pitch if pn.pitch is not None else 0
        pig_data[i]['velocity'] = pn.velocity
        
        # Mapping Main: XML(0=Right, 1=Left) -> HMM(Channel 0/1)
        pig_data[i]['channel'] = 0 if pn.hand == Hand.RIGHT else 1
        
        # On stocke la référence
        note_mapping[i] = pn

    # 3. Traitement par main
    # Le parser XML nous donne déjà les notes d'une seule main dans cette fonction callback
    # Mais par sécurité, on vérifie le channel du premier élément
    hand_int = pig_data[0]['channel']
    hand_str = "Main Droite" if hand_int == 0 else "Main Gauche"
    
    print(f"   [HMM] Traitement {hand_str} ({num_notes} notes)...")

    # 4. TRI CRITIQUE (Comme en C++)
    # Le HMM s'attend à un ordre très précis pour les accords (Grave -> Aigu)
    # et pour les notes simultanées (cluster < 30ms)
    ordered_pig = utils.apply_time_dep_pitch_order(pig_data)
    
    # 5. Exécution de Viterbi
    # Le moteur renvoie un tableau d'entiers (les doigts)
    predicted_fingers = model.run_viterbi(ordered_pig, params, hand=hand_int)
    
    # 6. Ré-injection des résultats
    # Attention : ordered_pig n'est pas dans le même ordre que pig_data !
    # Il faut utiliser 'original_idx' pour retrouver le bon objet PlayedNote
    
    count_assigned = 0
    for i in range(len(ordered_pig)):
        original_idx = ordered_pig[i]['original_idx']
        finger = predicted_fingers[i]
        
        # Si c'est la main gauche, le HMM travaille en valeurs positives (1..5)
        # Mais on peut vouloir stocker -1..-5 ou 1..5 selon votre convention.
        # Le parser XML XMLParser semble attendre un int simple. 
        # On garde la convention 1..5 pour les deux mains ici.
        
        target_note = note_mapping[original_idx]
        target_note.finger = int(finger)
        count_assigned += 1

    print(f"   [HMM] {count_assigned} doigtés assignés.")
    
    # On retourne la liste originale (modifiée en place via les références objets)
    return notes

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_xml.py input.xml output.xml")
        sys.exit(1)

    input_xml = sys.argv[1]
    output_xml = sys.argv[2]

    print(f"Lecture de {input_xml}...")
    
    # On appelle votre parser en lui passant notre adaptateur comme algorithme
    final_notes = process_musicxml(
        input_xml, 
        output_xml, 
        fingering_algorithm=hmm_fingering_adapter
    )
    
    print("Terminé.")