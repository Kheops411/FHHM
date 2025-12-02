# main.py

import sys
from typing import List, Optional
from types import SimpleNamespace # <-- On utilise cet objet simple

# Importer les composants nécessaires
# (Assurez-vous que les fichiers sont dans la bonne structure)
from xml_parser import MusicXMLParser, PlayedNote, inject_fingerings
from legacy.engine import find_fingerings
from legacy.utils import keypos

# --- Section Adaptateur (simplifiée) ---

# Noms des notes pour la conversion MIDI -> Nom
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note_details(midi_pitch: int):
    """Convertit une hauteur MIDI en nom de note et octave."""
    octave = (midi_pitch // 12) - 1
    note_name = NOTE_NAMES[midi_pitch % 12]
    return note_name, octave

def legacy_fingering_algorithm(notes: List[PlayedNote]) -> List[PlayedNote]:
    """
    Adaptateur qui convertit les PlayedNote, exécute l'algorithme de doigté legacy,
    et met à jour les notes originales.
    """
    if not notes:
        return []

    # 1. Convertir les PlayedNote vers un format compatible avec l'engine
    noteseq_for_engine = []
    
    chord_onsets = {}
    for note in notes:
        if note.extra.get('chord_size', 1) > 1:
            if note.onset not in chord_onsets:
                chord_onsets[note.onset] = id(note)

    for note in notes:
        if note.pitch is None:
            continue

        note_name, octave = midi_to_note_details(note.pitch)

        # Simuler un objet "note" pour la fonction keypos
        keypos_note = SimpleNamespace(name=note_name, octave=octave)
        x_pos = keypos(keypos_note)
        
        is_black = '#' in note_name
        is_chord = note.extra.get('chord_size', 1) > 1
        chord_id = chord_onsets.get(note.onset)

        # Créer l'objet pour le moteur de doigté SANS DATACLASS
        fingering_note = SimpleNamespace(
            x=x_pos,
            duration=note.duration * 4, # Mise à l'échelle pour la logique de l'engine
            isBlack=is_black,
            isChord=is_chord,
            time=note.onset,
            measure=note.measure_number,
            chordID=chord_id,
            original_note=note,
            # L'engine ajoutera ces attributs, on peut les initialiser pour la clarté
            fingering=None, 
            cost=None
        )
        noteseq_for_engine.append(fingering_note)


    # 2. Exécuter l'algorithme de doigté legacy
    hand_side = "right" if notes[0].hand == 0 else "left"
    print(hand_side, noteseq_for_engine)
    fingered_sequence = find_fingerings(noteseq_for_engine, side=hand_side, size='M')

    # 3. Mettre à jour les PlayedNote originales
    for f_note in fingered_sequence:
        if hasattr(f_note, 'fingering') and f_note.fingering:
            f_note.original_note.finger = f_note.fingering
            
    return notes

# --- Fin de la section Adaptateur ---

def apply_fingering_to_notes(notes: List[PlayedNote]) -> List[PlayedNote]:
    """Applique l'algorithme de doigté aux notes, séparées par main."""
    by_hand = {0: [], 1: []}
    for note in notes:
        by_hand[note.hand].append(note)
    
    for hand, hand_notes in by_hand.items():
        if not hand_notes:
            continue
        hand_notes.sort(key=lambda x: (x.onset, x.pitch or -1))
        legacy_fingering_algorithm(hand_notes)
    
    return notes


if __name__ == '__main__':
    # Petite correction : la fonction dans xml_parser.py s'appelle inject_fingerings (avec un 's')
    # Je vais la renommer ici pour correspondre, si c'est le cas. Sinon, ajustez le nom.
    try:
        from xml_parser import inject_fingerings
    except ImportError:
        print("Erreur: La fonction 'inject_fingerings' n'a pas été trouvée dans xml_parser.py.")
        sys.exit(1)

    if len(sys.argv) < 3:
        print("Usage: python main.py <input.xml> <output.xml>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"1. Analyse du fichier MusicXML : {input_file}...")
    xml_parser = MusicXMLParser(input_file)
    played_notes = xml_parser.parse()
    print(f"   ✓ {len(played_notes)} événements de note trouvés.")

    print("2. Application de l'algorithme de doigté...")
    fingered_notes = apply_fingering_to_notes(played_notes)
    print("   ✓ Doigté appliqué.")

    print(f"3. Injection des doigtés dans le nouveau fichier XML : {output_file}...")
    inject_fingerings(xml_parser.tree, fingered_notes, output_file)
    print("   ✓ Terminé.")