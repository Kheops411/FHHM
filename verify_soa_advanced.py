import sys
import os
import numpy as np
from glob import glob
from types import SimpleNamespace

# --- Imports Legacy Original ---
try:
    from xml_parser import MusicXMLParser
    from legacy.engine import find_fingerings as find_fingerings_legacy
    from legacy.utils import keypos
except ImportError:
    print("CRITICAL: Original Legacy modules missing.")
    sys.exit(1)

# --- Imports New SOA ---
try:
    from xml_parser_soa import musicxml_to_soa
    from legacy.engine_soa import find_fingerings_soa
    from structures import ScoreData
except ImportError:
    print("CRITICAL: New SOA modules missing.")
    sys.exit(1)

def run_legacy_original(xml_path):
    """Exécute l'ancien code et retourne une map { (onset, pitch, hand) -> finger }"""
    parser = MusicXMLParser(xml_path)
    played_notes = parser.parse()
    
    # Filtrer et trier
    notes = [n for n in played_notes if n.pitch is not None]
    notes.sort(key=lambda x: (x.onset, x.pitch))
    
    results = {}
    
    # Traiter main par main comme dans main.py
    for hand_id, hand_str in [(0, "right"), (1, "left")]:
        hand_notes = [n for n in notes if n.hand == hand_id]
        if not hand_notes: continue

        # Adapter vers format Engine Legacy
        engine_notes = []
        chord_map = {}
        for n in hand_notes:
            if n.extra.get('chord_size', 1) > 1:
                chord_map[n.onset] = int(n.onset * 1000)

        for n in hand_notes:
            note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            note_name = note_names[n.pitch % 12]
            octave = (n.pitch // 12) - 1
            kp = SimpleNamespace(name=note_name, octave=octave)
            
            engine_notes.append(SimpleNamespace(
                x=keypos(kp), 
                duration=n.duration * 4,
                isBlack='#' in note_name, 
                isChord=n.extra.get('chord_size', 1) > 1,
                time=n.onset, 
                measure=n.measure_number, 
                chordID=chord_map.get(n.onset),
                original_note=n,
                fingering=None
            ))
            
        # Run Legacy Algo
        fingered = find_fingerings_legacy(engine_notes, hand_str, "M")
        
        # Store Result
        for fn in fingered:
            # Clé unique pour comparaison : Temps, Pitch, Main
            key = (round(fn.time, 4), fn.original_note.pitch, hand_id)
            results[key] = fn.fingering

    return results

def run_legacy_soa(xml_path):
    """Exécute le nouveau code SOA et retourne une map similaire"""
    soa, _ = musicxml_to_soa(xml_path)
    
    # Run SOA Algo (Both hands)
    fingers_rh = find_fingerings_soa(soa, "right")
    fingers_lh = find_fingerings_soa(soa, "left")
    
    # Combine results
    soa.finger_out = fingers_rh + fingers_lh
    
    results = {}
    for i in range(len(soa)):
        if soa.finger_out[i] != 0:
            key = (round(soa.onset[i], 4), soa.pitch[i], soa.hand[i])
            results[key] = soa.finger_out[i]
            
    return results

def compare_implementations():
    xml_files = glob("tests/resources/*.xml")
    if not xml_files:
        print("No XML files found in tests/resources/")
        return

    total_notes = 0
    total_mismatches = 0
    
    print(f"{'FILENAME':<40} | {'NOTES':<6} | {'MATCH':<6} | {'STATUS'}")
    print("-" * 80)

    for xml_file in xml_files:
        filename = os.path.basename(xml_file)
        
        # 1. Get results from both
        res_orig = run_legacy_original(xml_file)
        res_soa = run_legacy_soa(xml_file)
        
        # 2. Compare
        # On compare sur l'intersection des clés (au cas où un algo planterait sur une note)
        keys_orig = set(res_orig.keys())
        keys_soa = set(res_soa.keys())
        
        all_keys = keys_orig.union(keys_soa)
        file_mismatches = 0
        
        # Détails des erreurs pour debug profond
        errors = []

        for k in all_keys:
            f_orig = res_orig.get(k, 0) # 0 si manquant
            f_soa = res_soa.get(k, 0)   # 0 si manquant
            
            if f_orig != f_soa:
                file_mismatches += 1
                errors.append(f"  Time {k[0]:.2f}s | Pitch {k[1]} | Hand {k[2]} : Old={f_orig} vs New={f_soa}")

        total_notes += len(all_keys)
        total_mismatches += file_mismatches
        
        status = "PASS" if file_mismatches == 0 else "FAIL"
        print(f"{filename:<40} | {len(all_keys):<6} | {file_mismatches:<6} | {status}")
        
        if file_mismatches > 0:
            print("\n".join(errors[:5])) # Affiche les 5 premières erreurs
            if len(errors) > 5: print(f"  ... and {len(errors)-5} more.")
            print("-" * 80)

    print("\n" + "="*80)
    print(f"TOTAL SUMMARY: {total_notes} notes processed.")
    if total_mismatches == 0:
        print("SUCCESS: Perfect match across all files.")
    else:
        print(f"FAILURE: {total_mismatches} mismatches found ({100 * total_mismatches / total_notes:.2f}% error rate).")
    print("="*80)

if __name__ == "__main__":
    compare_implementations()