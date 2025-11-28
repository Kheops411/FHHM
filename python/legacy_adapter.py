import numpy as np
import sys
import os

# On ajoute le dossier parent au path pour importer les modules legacy si besoin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importation du moteur Legacy
# Assurez-vous que les fichiers sont bien dans un dossier 'legacy' avec un __init__.py vide
from legacy import engine as legacy_engine
from legacy import utils as legacy_utils

class LegacyNote:
    """
    Objet Mock qui imite l'interface attendue par le moteur Legacy.
    """
    def __init__(self, pig_row, prev_ontime=None):
        self.pitch = pig_row['pitch']
        self.time = pig_row['ontime']
        self.duration = pig_row['offtime'] - pig_row['ontime']
        self.measure = 0 # Non utilisé pour l'eval PIG (on traite tout le morceau)
        
        # Calcul de la géométrie selon la logique Legacy (legacy_utils)
        # Note: legacy_utils.keypos_midi utilise une logique spécifique
        # On simule un objet avec attribut 'pitch' pour keypos_midi
        class PitchContainer:
            def __init__(self, p): self.pitch = p
        
        # Calcul de X (cm)
        self.x = legacy_utils.keypos_midi(PitchContainer(self.pitch))
        
        # Touche noire ?
        pc = self.pitch % 12
        self.isBlack = pc in [1, 3, 6, 8, 10]
        
        # Logique d'accord simplifiée (basée sur le temps)
        # Le moteur Legacy utilise chordID pour grouper
        self.isChord = False
        self.chordID = 0
        self.chordnr = 0 # index dans l'accord
        self.NinChord = 1 # taille de l'accord

        # Placeholder pour le résultat
        self.fingering = 0
        self.cost = 0

def prepare_legacy_sequence(pig_notes):
    """
    Transforme un tableau numpy PIG en liste d'objets LegacyNote.
    Gère la détection d'accords (ChordID) requise par le moteur Legacy.
    """
    legacy_notes = []
    if len(pig_notes) == 0:
        return legacy_notes

    current_chord_id = 0
    cluster_start_time = pig_notes[0]['ontime']
    cluster_notes = []

    for i in range(len(pig_notes)):
        note = LegacyNote(pig_notes[i])
        
        # Détection d'accord basique (seuil 30ms comme HMM)
        if abs(note.time - cluster_start_time) < 0.03:
            cluster_notes.append(note)
        else:
            # Finaliser le cluster précédent
            _finalize_cluster(cluster_notes, current_chord_id)
            current_chord_id += 1
            legacy_notes.extend(cluster_notes)
            
            # Nouveau cluster
            cluster_notes = [note]
            cluster_start_time = note.time
    
    # Finaliser le dernier cluster
    if cluster_notes:
        _finalize_cluster(cluster_notes, current_chord_id)
        legacy_notes.extend(cluster_notes)
        
    return legacy_notes

def _finalize_cluster(cluster, chord_id):
    is_chord = len(cluster) > 1
    for i, note in enumerate(cluster):
        note.isChord = is_chord
        note.chordID = chord_id
        note.chordnr = i + 1
        note.NinChord = len(cluster)

def run_legacy_algorithm(pig_notes, hand_side='right', hand_size='M'):
    """
    Exécute le moteur Legacy sur des données PIG.
    Retourne un tableau numpy d'entiers (doigtés).
    """
    if len(pig_notes) == 0:
        return np.array([], dtype=np.int32)

    # 1. Conversion
    legacy_seq = prepare_legacy_sequence(pig_notes)
    
    # 2. Exécution du moteur
    # Le moteur Legacy a besoin de 'left' ou 'right'
    side_str = 'right' if hand_side == 0 else 'left'
    
    # Instanciation de la main Legacy
    hand_solver = legacy_engine.Hand(legacy_seq, side=side_str, size=hand_size)
    
    # Paramètres par défaut du Legacy
    hand_solver.autodepth = True
    
    # Génération
    result_seq = hand_solver.generate()
    
    # 3. Extraction des résultats
    fingers = np.array([n.fingering for n in result_seq], dtype=np.int32)
    
    return fingers