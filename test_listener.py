import sys
import os
from lxml import etree
from midiutil import MIDIFile

# Importation du Code C
from xml_parser import process_musicxml, Hand

def get_initial_tempo(xml_path):
    """Scanne le XML pour trouver le tempo initial."""
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        sound = root.find(".//sound[@tempo]")
        if sound is not None:
            return float(sound.get("tempo"))
        per_minute = root.find(".//per-minute")
        if per_minute is not None and per_minute.text:
            return float(per_minute.text)
    except:
        pass
    return 120.0

def sanitize_events(events):
    """
    Nettoie les événements pour éviter les crashs MIDIUtil.
    1. Supprime les notes sans pitch (silences).
    2. Impose une durée minimale audible (évite l'erreur 'pop from empty list').
    3. Arrondit les temps pour éviter les instabilités flottantes.
    """
    cleaned = []
    # Durée minimale : 1/64ème de noire (environ)
    # En dessous de ça, MIDIUtil arrondit à 0 ticks et crashe.
    MIN_MIDI_DURATION = 0.05 
    
    for n in events:
        if n.pitch is None:
            continue
            
        # Création d'une copie ou modification directe
        onset = round(n.onset, 4)
        duration = max(n.duration, MIN_MIDI_DURATION)
        duration = round(duration, 4)
        
        # On ne modifie pas l'objet original du Code C (pour ne pas fausser le doigté futur),
        # on crée un dict temporaire pour le MIDI
        cleaned.append({
            'hand': n.hand,
            'pitch': int(n.pitch),
            'onset': onset,
            'duration': duration,
            'velocity': n.velocity
        })
        
    # Tri secondaire de sécurité (Temps, puis Note-ON avant Note-OFF implicite)
    cleaned.sort(key=lambda x: x['onset'])
    return cleaned

def events_to_midi(played_notes, output_midi_filename, tempo_bpm):
    midi = MIDIFile(numTracks=2)
    midi.addTempo(0, 0, tempo_bpm)
    midi.addTempo(1, 0, tempo_bpm)
    midi.addProgramChange(0, 0, 0, 0)
    midi.addProgramChange(1, 1, 0, 0)

    # ÉTAPE CRUCIALE : Nettoyage
    clean_notes = sanitize_events(played_notes)

    print(f"Conversion de {len(clean_notes)} événements nettoyés en MIDI à {tempo_bpm} BPM...")

    for note in clean_notes:
        midi.addNote(
            note['hand'],   # track
            note['hand'],   # channel
            note['pitch'],
            note['onset'],
            note['duration'],
            note['velocity']
        )

    with open(output_midi_filename, "wb") as output_file:
        midi.writeFile(output_file)
    
    print(f"✓ Fichier MIDI généré : {output_midi_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_listener.py <input.xml>")
        sys.exit(1)

    xml_input = sys.argv[1]
    base_name = os.path.splitext(xml_input)[0]
    midi_output = base_name + "_test_audio.mid"
    dummy_xml = base_name + "_temp_debug.xml"

    print(f"--- Test Audio Robuste ---")
    print(f"Lecture de : {xml_input}")

    try:
        tempo = get_initial_tempo(xml_input)
        print(f"Tempo détecté : {tempo} BPM")

        events = process_musicxml(xml_input, dummy_xml)
        events_to_midi(events, midi_output, tempo)

        print("\nSUCCÈS ! Le problème de durée zéro devrait être résolu.")
        if os.path.exists(dummy_xml): os.remove(dummy_xml)

    except Exception as e:
        print(f"\nERREUR : {e}")
        import traceback
        traceback.print_exc()