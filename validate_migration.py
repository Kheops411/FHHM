import os
import glob
import sys
import numpy as np
import importlib.util

# ============================================================================
# Configuration
# ============================================================================

RESOURCES_DIR = os.path.join("tests", "resources")
V1_MODULE_NAME = "xml_parser"
V2_MODULE_NAME = "xml_parser_v2"

# Tolérance pour les comparaisons flottantes (secondes)
# On utilise une tolérance très fine car la logique mathématique est censée être identique.
TIME_TOLERANCE = 1e-4 

# ============================================================================
# Import Dynamique (pour gérer les noms de fichiers locaux)
# ============================================================================

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    # On suppose que les fichiers sont à la racine
    parser_v1 = import_module_from_file(V1_MODULE_NAME, f"{V1_MODULE_NAME}.py")
    parser_v2 = import_module_from_file(V2_MODULE_NAME, f"{V2_MODULE_NAME}.py")
except FileNotFoundError as e:
    print(f"❌ Erreur critique : Impossible de trouver les fichiers sources.\n{e}")
    sys.exit(1)

# ============================================================================
# Logique de Comparaison
# ============================================================================

def normalize_v1_notes(notes_v1):
    """
    Transforme la liste d'objets PlayedNote (V1) en une liste de tuples standardisés
    pour la comparaison.
    Format tuple : (onset_sec, pitch, hand, duration_sec, velocity, finger_gt)
    """
    normalized = []
    for n in notes_v1:
        # Conversion Pitch: None (Silence) -> 0
        pitch = n.pitch if n.pitch is not None else 0
        
        # Conversion Hand: Enum -> Int
        hand = int(n.hand)
        
        # Note: V1 stocke onset en 'quarters', mais a une propriété .onset_seconds
        # V2 stocke directement en seconds.
        # Idem pour duration.
        normalized.append({
            'onset': n.onset_seconds,
            'pitch': pitch,
            'hand': hand,
            'duration': n.duration_seconds,
            'velocity': n.velocity,
            'finger': n.finger if n.finger is not None else 0
        })
    
    # Tri strict pour alignement : Onset -> Pitch -> Hand
    # On arrondit l'onset pour le tri pour éviter les instabilités flottantes
    normalized.sort(key=lambda x: (round(x['onset'], 5), x['pitch'], x['hand']))
    return normalized

def normalize_v2_data(score_v2):
    """
    Extrait les données du ScoreData (V2) en une liste de dicts comparable.
    """
    normalized = []
    
    # Création d'index de tri pour aligner avec V1
    # np.lexsort trie selon la dernière clé passée en premier
    # Ordre voulu : Onset, Pitch, Hand
    # Donc on passe : (Hand, Pitch, Onset)
    # Note: On arrondit l'onset pour le tri
    rounded_onsets = np.round(score_v2.onset, 5)
    sorted_indices = np.lexsort((score_v2.hand, score_v2.pitch, rounded_onsets))
    
    for i in sorted_indices:
        normalized.append({
            'onset': score_v2.onset[i],
            'pitch': score_v2.pitch[i], # Déjà 0 si unset dans V2
            'hand': int(score_v2.hand[i]),
            'duration': score_v2.duration[i],
            'velocity': score_v2.velocity[i],
            'finger': score_v2.finger_gt[i]
        })
    
    return normalized

def compare_files(filepath):
    print(f"🔍 Testing: {os.path.basename(filepath)} ... ", end="", flush=True)
    
    # --- Run V1 ---
    try:
        p1 = parser_v1.MusicXMLParser(filepath)
        notes_v1 = p1.parse()
    except Exception as e:
        print(f"\n   ❌ V1 Crash: {e}")
        return False

    # --- Run V2 ---
    try:
        p2 = parser_v2.MusicXMLParser(filepath)
        score_v2 = p2.parse()
    except Exception as e:
        print(f"\n   ❌ V2 Crash: {e}")
        return False

    # --- Compare Meta ---
    if len(notes_v1) != score_v2.size:
        print(f"\n   ❌ Count Mismatch: V1={len(notes_v1)} vs V2={score_v2.size}")
        return False

    if len(notes_v1) == 0:
        print("✅ (Empty file)")
        return True

    # --- Compare Content ---
    data_v1 = normalize_v1_notes(notes_v1)
    data_v2 = normalize_v2_data(score_v2)
    
    errors = []
    
    for i, (n1, n2) in enumerate(zip(data_v1, data_v2)):
        # 1. Onset
        if abs(n1['onset'] - n2['onset']) > TIME_TOLERANCE:
            errors.append(f"Note {i}: Onset mismatch (V1={n1['onset']:.5f}s, V2={n2['onset']:.5f}s)")
        
        # 2. Duration
        if abs(n1['duration'] - n2['duration']) > TIME_TOLERANCE:
            errors.append(f"Note {i}: Duration mismatch (V1={n1['duration']:.5f}s, V2={n2['duration']:.5f}s)")
            
        # 3. Pitch
        if n1['pitch'] != n2['pitch']:
            errors.append(f"Note {i}: Pitch mismatch (V1={n1['pitch']}, V2={n2['pitch']}) at time {n1['onset']:.2f}s")
            
        # 4. Hand
        if n1['hand'] != n2['hand']:
            errors.append(f"Note {i}: Hand mismatch (V1={n1['hand']}, V2={n2['hand']})")

        # 5. Velocity
        if n1['velocity'] != n2['velocity']:
            errors.append(f"Note {i}: Velocity mismatch (V1={n1['velocity']}, V2={n2['velocity']})")

        # 6. Fingering (Ground Truth)
        if n1['finger'] != n2['finger']:
            errors.append(f"Note {i}: Finger GT mismatch (V1={n1['finger']}, V2={n2['finger']})")
        
        if len(errors) > 5:
            errors.append("... (too many errors, stopping comparison)")
            break
    
    if errors:
        print("\n   ❌ Content Mismatch:")
        for e in errors:
            print(f"      - {e}")
        return False
    
    print("✅ OK")
    return True

# ============================================================================
# Main Loop
# ============================================================================

def main():
    if not os.path.exists(RESOURCES_DIR):
        print(f"Directory not found: {RESOURCES_DIR}")
        print("Please create it and add .xml/.musicxml files to test.")
        sys.exit(0)

    files = glob.glob(os.path.join(RESOURCES_DIR, "*.xml")) + \
            glob.glob(os.path.join(RESOURCES_DIR, "*.musicxml"))
    
    if not files:
        print(f"No XML files found in {RESOURCES_DIR}")
        sys.exit(0)

    print(f"Found {len(files)} files. Starting validation...\n")
    
    success_count = 0
    failure_count = 0
    
    for f in files:
        if compare_files(f):
            success_count += 1
        else:
            failure_count += 1
            
    print("\n" + "="*40)
    print(f"SUMMARY")
    print(f"Total: {len(files)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {failure_count}")
    print("="*40)
    
    if failure_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()