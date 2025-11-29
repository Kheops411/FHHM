
# Spécification Technique : Algorithme Soft-Position HMM pour Doigtés de Piano

## 1. Vue d'ensemble et Architecture

L'algorithme repose sur un Modèle de Markov Caché (HMM) hybride intégrant une variable latente continue représentant la position de la main. Il vise à optimiser conjointement l'agilité des doigts (modèle discret) et l'inertie biomécanique du bras (modèle spatial continu).

### 1.1 Espace d'États
L'état du système à l'instant $t$ est défini par le triplet :
$$ S_t = (f_{t-1}, f_t, k_t) $$

*   **$f_{t-1}, f_t \in \{1..5\}$** : Les doigts utilisés à l'instant précédent et actuel (supportant une logique de transition d'ordre 3 pour l'agilité digitale).
*   **$k_t \in \{0..N_{ancres}-1\}$** : L'index de l'ancre spatiale représentant le centre de gravité de la main $x_t$.

La dimension de l'espace d'états pour Viterbi est : $5 \times 5 \times N_{ancres}$.

### 1.2 Découplage des Ordres
L'algorithme utilise une architecture asymétrique pour maintenir la performance temps-réel :
*   **Composante Digitale ($f$) :** Ordre 3 (Dépend de $f_{t-1}, f_{t-2}$). Gère l'agilité, les trilles et répétitions.
*   **Composante Spatiale ($x$) :** Ordre 1 (Dépend de $x_{t-1}$ uniquement). Gère l'inertie et la stabilité du bras.

---

## 2. Modélisation Géométrique (Moteur RBF)

La probabilité qu'un doigt atteigne une note est calculée via une projection sur des Bases Radiales (RBF).

### 2.1 Paramètres
*   **Ancres ($c$) :** Vecteur fixe de 9 positions relatives (en demi-tons) : `[-12, -9, -6, -3, 0, +3, +6, +9, +12]`.
*   **Poids Morphologiques ($W$) :** Matrice apprenable de dimension $(5 \times 9)$. Chaque ligne $W_f$ représente la morphologie du doigt $f$.
*   **Normalisation ($\mu_f, \sigma_f$) :** Paramètres scalaires (moyenne et écart-type glissants) pour centrer les activations.
*   **Température ($\tau$) :** Paramètre scalaire apprenable contrôlant l'influence relative de la géométrie.

### 2.2 Calcul du Score d'Émission
Pour une note de pitch $p_t$, un doigt $f_t$ et une position de main $x_t$ (définie par l'ancre $c_{k_t}$) :

1.  **Calcul du Delta :** $\delta = p_t - x_t$.
2.  **Activation RBF ($\Phi$) :** Calcul du vecteur d'activation sur les ancres (Interpolation linéaire ou Gaussienne).
3.  **Normalisation L2 du Champ :**
    $$ \tilde\Phi(\delta) = \frac{\Phi(\delta)}{\|\Phi(\delta)\|_2} $$
4.  **Projection et Centrage :**
    $$ z = \frac{W_{f_t} \cdot \tilde\Phi(\delta) - \mu_{f_t}}{\sigma_{f_t} + \epsilon} $$
5.  **Transformation Positive (Softplus) :**
    $$ S(f_t, \delta) = \ln(1 + e^z) $$
6.  **Énergie Finale (Log-Probabilité) :**
    $$ \log P_{emit} = -\frac{1}{\tau} \log(S(f_t, \delta)) $$

---

## 3. Modélisation Temporelle et Contraintes

Le modèle distingue deux régimes physiques basés sur l'intervalle de temps $\Delta t$ entre les notes.

### 3.1 Régime Topologique ($\Delta t < 0.03s$ - Accords)
Pour les notes simultanées, aucune pénalité "soft" n'est appliquée. Une contrainte stricte est imposée :
1.  **Position Unique :** Toutes les notes de l'accord doivent partager le même état spatial $x_t$.
2.  **Masque de Non-Croisement :** Pour toute paire de notes $(p_a, p_b)$ dans l'accord avec $p_a < p_b$, si les doigts associés sont $(f_a, f_b)$, alors l'état est invalide ($-\infty$) si $f_a > f_b$.

### 3.2 Régime Cinématique ($\Delta t \ge 0.03s$ - Séquences)
Le coût de transition remplace le `short_time_cost` par une modulation dynamique de l'inertie.

$$ \text{Coût}_{trans} = \log P_{agilité}(f_t | f_{t-1}, f_{t-2}) + \lambda(\Delta t) \cdot |x_t - x_{t-1}| $$

**Fonction de Rigidité $\lambda(\Delta t)$ :**
Une sigmoïde inversée paramétrée :
$$ \lambda(\Delta t) = \frac{1}{1 + e^{\alpha(\Delta t - t_0)}} $$
*   Si $\Delta t \to 0$ (Legato), $\lambda \approx 1$ (Inertie maximale, mouvement de bras pénalisé).
*   Si $\Delta t$ grand (Staccato/Silence), $\lambda \to 0$ (Inertie nulle, mouvement de bras gratuit).

---

## 4. Algorithmes

### 4.1 Inférence (Viterbi Découplé)
L'algorithme cherche le chemin optimal dans le treillis $5 \times 5 \times 9$.

*   **Boucle Principale :** Pour chaque temps $t$.
*   **Boucle États Actuels :** Pour chaque $(f_{prev}, f_{curr}, k_{curr})$.
*   **Boucle États Précédents :** Pour chaque $(f_{prev-1}, k_{prev})$.
    *   *Note :* On itère sur $f_{prev-1}$ pour l'ordre 3 des doigts.
    *   *Note :* On itère sur $k_{prev}$ pour l'ordre 1 de l'espace (Inertie).
*   **Score :** Somme de l'émission (Section 2.2) et de la transition unifiée (Section 3.2).

### 4.2 Entraînement (Hard EM / Viterbi Training)
Puisque la position $x_t$ est latente (non annotée), l'entraînement suit un processus itératif.

**Phase A : Initialisation (Biomécanique)**
La matrice $W$ est initialisée avec des gaussiennes asymétriques pour refléter l'anatomie, particulièrement pour le pouce (Doigt 1).
*   **Doigt 1 (MD) :** $\mu \approx -4.0, \sigma \approx 4.0$. Distribution "lourde à gauche" (basses) mais autorisant le centre (passage pouce). Chute stricte à droite.
*   **Doigts 2, 3, 4 :** Distributions étroites centrées autour de positions relatives $(-2, 0, +2)$.
*   **Doigt 5 :** Distribution centrée à droite ($\mu \approx +5.0$).

**Phase B : Boucle EM**
Répéter jusqu'à convergence :
1.  **E-Step (Alignement) :** Exécuter un Viterbi Contraint sur le dataset d'entraînement.
    *   On force le chemin à passer par les doigts annotés $f_{true}$.
    *   Le modèle "devine" les positions optimales $x_t$ associées.
    *   Collecte des paires $(\text{Doigt}, \delta_{observé})$.
2.  **M-Step (Optimisation) :**
    *   Mise à jour des poids $W$ et de $\tau$ par descente de gradient pour maximiser la log-vraisemblance des $\delta_{observé}$.
    *   Mise à jour des paramètres de normalisation $\mu_f, \sigma_f$.

---

## 5. Détails d'Implémentation Critiques

1.  **Suppression des Outputs Obsolètes :** Les matrices `log_output_prob` (1er, 2e, 3e ordre) basées sur les intervalles relatifs sont supprimées. Seule la matrice `RBF_Weights` est utilisée pour l'émission.
2.  **Stabilité Numérique ($\epsilon$) :** Ajouter $\epsilon = 10^{-8}$ dans toutes les divisions (normalisation L2, centrage) et dans le `softplus`.
3.  **Interpolation Spatiale :** Bien que l'état $k_t$ soit discret (ancres), le calcul de l'inertie $|x_t - x_{t-1}|$ utilise la distance physique en demi-tons entre les ancres, pas la différence d'indices.
4.  **Miroir Main Gauche :** Pour l'entraînement et l'inférence Main Gauche, inverser le signe des $\delta$ avant l'entrée dans le moteur RBF (ou inverser l'ordre des ancres), afin d'utiliser la même matrice morphologique $W$.

