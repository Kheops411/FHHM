import glob
import os
import time
from python.training import HMMTrainer

def run_training():
    # 1. Récupération de tous les fichiers PIG
    # Assurez-vous que le dossier scores/ contient bien les fichiers .txt
    score_files = sorted(glob.glob(os.path.join("scores", "*_fingering.txt")))
    
    if not score_files:
        print("ERREUR : Aucun fichier trouvé dans le dossier 'scores/'")
        return

    print(f"--- Démarrage de l'entraînement sur {len(score_files)} fichiers ---")

    # ==========================================
    # 2. Entraînement HMM Ordre 2
    # ==========================================
    print("\n[1/2] Entraînement HMM Ordre 2...")
    start_time = time.time()
    
    # tr_sym=True (Time Reversal) et rf_sym=True (Reflection/Miroir) 
    # sont recommandés pour augmenter artificiellement la taille du dataset 
    # et rendre le modèle plus robuste (comme fait dans le papier).
    trainer_2 = HMMTrainer(order=2, tr_sym=True, rf_sym=True)
    
    trainer_2.train(score_files)
    
    output_file_2 = "param_FHMM2_new.txt"
    trainer_2.save_parameters(output_file_2)
    
    print(f"      Terminé en {time.time() - start_time:.2f}s")
    print(f"      Paramètres sauvegardés dans : {output_file_2}")

    # ==========================================
    # 3. Entraînement HMM Ordre 3
    # ==========================================
    print("\n[2/2] Entraînement HMM Ordre 3...")
    start_time = time.time()
    
    trainer_3 = HMMTrainer(order=3, tr_sym=True, rf_sym=True)
    
    trainer_3.train(score_files)
    
    output_file_3 = "param_FHMM3_new.txt"
    trainer_3.save_parameters(output_file_3)
    
    print(f"      Terminé en {time.time() - start_time:.2f}s")
    print(f"      Paramètres sauvegardés dans : {output_file_3}")

if __name__ == "__main__":
    run_training()