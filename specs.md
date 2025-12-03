# Spécification Technique : Système Unifié de Doigté Piano (SoA)

## 1. Modèle de Données (`structures.py`)

Cette structure est le conteneur unique des données musicales et structurelles. Elle ne contient aucune information de positionnement spatial ou de topologie clavier.

### 1.1 Constantes Globales

```python
import numpy as np

# Tolérance temporelle pour considérer deux notes comme simultanées
# 1e-3s (1ms) couvre les imprécisions MIDI et les arrondis MusicXML.
EPSILON_ONSET = 1e-3

# Constantes pour l'identification des mains
HAND_RIGHT = 0
HAND_LEFT = 1
HAND_UNKNOWN = -1
```

### 1.2 La Classe `ScoreData`

Utilisation de `__slots__` pour figer la structure mémoire. Tous les champs sont des tableaux NumPy 1D alignés (C-Contiguous).

```python
class ScoreData:
    __slots__ = (
        'onset', 'offset', 'pitch', 'velocity', 
        'source_idx', 'hand', 
        'finger_gt', 'finger_out'
    )

    def __init__(self, n: int):
        """
        Allocation stricte.
        n : Nombre total de notes.
        """
        # --- Données Temporelles (Secondes) ---
        self.onset = np.zeros(n, dtype=np.float64)
        self.offset = np.zeros(n, dtype=np.float64)
        
        # --- Données Musicales Abstraites ---
        self.pitch = np.zeros(n, dtype=np.int16)        # MIDI 0-127
        self.velocity = np.zeros(n, dtype=np.int8)      # 0-127
        
        # --- Traçabilité & Structure ---
        # Index vers une liste externe stable d'objets sources (XML elements ou dicts)
        self.source_idx = np.zeros(n, dtype=np.int64)
        self.hand = np.full(n, HAND_UNKNOWN, dtype=np.int8)
        
        # --- Entrées / Sorties Algorithmes ---
        self.finger_gt = np.zeros(n, dtype=np.int8)     # Vérité terrain (Ground Truth)
        self.finger_out = np.zeros(n, dtype=np.int8)    # Résultat calculé (0 si vide)

    @property
    def size(self):
        return len(self.onset)

    def sort_canonical(self):
        """
        Tri Canonique : Onset (ascendant) -> Pitch (ascendant).
        Garantit que les données envoyées aux algorithmes sont chronologiques 
        et stockées de manière contiguë en mémoire.
        """
        # Clé de tri : dernier argument est la clé primaire
        sorter = np.lexsort((self.pitch, self.onset))
        
        for attr in self.__slots__:
            arr = getattr(self, attr)
            # np.ascontiguousarray force la création d'un nouveau buffer aligné
            sorted_arr = np.ascontiguousarray(arr[sorter])
            setattr(self, attr, sorted_arr)

    def validate(self):
        """
        Validation technique et logique avant traitement.
        """
        # 1. Validation Mémoire
        for attr in self.__slots__:
            arr = getattr(self, attr)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Champ '{attr}' corrompu (pas un ndarray).")
            if not arr.flags['C_CONTIGUOUS']:
                raise ValueError(f"Champ '{attr}' non contigu en mémoire.")

        # 2. Validation Logique
        if np.any(self.offset < self.onset):
            raise ValueError("Corruption : Durée négative détectée.")
        
        # Vérification du tri (avec tolérance flottante)
        if np.any(np.diff(self.onset) < -EPSILON_ONSET):
            raise ValueError("Corruption : Données non triées temporellement.")
```

---

## 2. Pipeline d'Entrée (Parseurs)

Les parseurs doivent dissocier les données vectorisables des objets sources originaux.

### 2.1 Stratégie de Mapping Source
1.  Créer une liste Python **persistante** : `source_objects_list = []`.
2.  Cette liste stocke les objets originaux (ex: `etree.Element` pour XML).
3.  Le champ `ScoreData.source_idx[i]` stocke l'index entier `k` tel que `source_objects_list[k]` correspond à la note `i`.

### 2.2 Séquence de Parsing
1.  **Extraction** : Lecture du fichier source, peuplement de `source_objects_list`, et stockage des valeurs musicales dans des listes temporaires.
2.  **Allocation** : Instanciation de `soa = ScoreData(N)`.
3.  **Remplissage** : Copie des listes temporaires dans les tableaux NumPy (`soa.pitch[:] = temp_list`).
4.  **Tri** : Appel de `soa.sort_canonical()`.
5.  **Validation** : Appel de `soa.validate()`.

---

## 3. Intégration des Algorithmes

Le module d'E/S ne fournit que les données brutes. Toute transformation en données physiques (coordonnées spatiales, statut touche noire/blanche) est de la responsabilité exclusive de l'algorithme qui en a besoin.

### 3.1 Modèle Commun d'Exécution

Pour tout algorithme, la procédure d'interface est :

1.  **Filtrage par Main** : Sélection des indices concernés.
    ```python
    mask = (soa.hand == target_hand)
    indices = np.where(mask)[0]
    if len(indices) == 0: return
    ```
2.  **Extraction Contiguë (Copie)** : Création de sous-tableaux pour passage à Numba ou autre moteur.
    ```python
    # Extraction des données musicales pures
    sub_pitch = np.ascontiguousarray(soa.pitch[indices])
    sub_onset = np.ascontiguousarray(soa.onset[indices])
    # Extraction optionnelle selon besoins (durée, vélocité)
    sub_duration = np.ascontiguousarray(soa.offset[indices] - soa.onset[indices])
    ```
3.  **Traitement Algorithmique** :
    *   L'algorithme reçoit `sub_pitch`, `sub_onset`, etc.
    *   Si l'algorithme requiert des données physiques (ex: "Legacy"), il doit convertir `sub_pitch` en coordonnées internes via sa propre table de correspondance (Look-Up Table) ou logique de calcul.
4.  **Réinjection (Scatter)** : Écriture du résultat dans le conteneur global.
    ```python
    soa.finger_out[indices] = result_fingers
    ```

### 3.2 Spécificités d'Adaptation

*   **Algorithme 1 (Legacy/Heuristique)** :
    *   Ne reçoit plus d'objets `Hand` pré-calculés par l'extérieur.
    *   Reçoit le vecteur `sub_pitch`.
    *   **Interne** : Doit implémenter/intégrer sa propre logique `pitch_to_x_coordinates` et `is_black_key(pitch)` avant d'exécuter ses calculs de coûts.
*   **Algorithme 2 (HMM)** :
    *   Reçoit directement `sub_pitch` et `sub_onset`.
    *   Utilise sa propre LUT interne (`PITCH_TO_KEYPOS_LUT`) déjà présente dans son code pour la conversion spatiale nécessaire à Viterbi.