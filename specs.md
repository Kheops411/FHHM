## 1. Représentation de la position (feature spatiale)

Le Soft-position HMM repose sur une représentation probabiliste (ou lissée) des positions de notes.

Voici l'approche à adopter :

### Représentation Hybride (RBF / Bases Radiales)

**Le Principe :**

On définit un petit nombre d'ancres fixes (par exemple 9 centres $c_k$ répartis sur le clavier).
La position d'une note $x$ n'est pas un index, mais un **vecteur d'activation**.

Si la note est à la position $x$, on calcule son activation pour chaque ancre $k$ :
$$ \phi_k(x) = \text{Activation de l'ancre } k \text{ par la position } x $$
La probabilité finale est une somme pondérée (produit scalaire) :
$$ \log P = \sum_{k=1}^{9} w_k \cdot \phi_k(x) $$

**Fonctionnement :**
1.  **Paramètres :** Le modèle apprend un vecteur de poids $w$ de taille 9 pour chaque doigt (très léger : 45 paramètres au total pour l'émission).
2.  **Viterbi :** Au lieu de faire `table[x]`, on fait un produit scalaire `dot(weights, activations)`.

**Conséquences Concrètes :**
*   **Généralisation (Parfaite) :** Comme l'Option A, c'est continu. Mais c'est plus flexible qu'une gaussienne simple (on peut apprendre des formes bizarres).
*   **Vitesse (Très Rapide) :** Un produit scalaire de vecteurs taille 9 est extrêmement rapide pour un CPU (instructions SIMD), bien plus que des exponentielles.
*   **Lutte contre l'Overfitting :** C'est l'option qui a le **moins de paramètres à apprendre**. Au lieu d'apprendre 128 valeurs (Option B), on en apprend 9. Avec le petit dataset PIG, c'est un avantage mathématique décisif.
*   **Continuité :** Permet à la position de la main d'être une valeur réelle flottante (ex: 60.4) et d'être interpolée proprement entre les ancres.

## 2. Structure de sortie : un ou plusieurs modèles soft

Le HMM peut utiliser :
* une seule distribution soft pour les deltas,
ou
* plusieurs distributions différentes selon la profondeur (1st-order, 2nd-order, 3rd-order).

Voici l'approche à adopter :

### Modèle Soft UNIQUE (Partagé / Shared Morphology)

**Le Concept :**
On fait l'hypothèse que la **morphologie de la main est constante**.
L'émission $P(\text{Note} | \text{Doigt}, \text{Position Main})$ représente la capacité physique d'un doigt à s'écarter du centre de la main.

*   Cette capacité est anatomique. La longueur de votre auriculaire (doigt 5) ne change pas, que vous veniez de jouer une note il y a 100ms (ordre 1) ou 500ms (ordre 3).
*   On apprend donc **une seule matrice de poids $W$** (taille $5 \times 9$ pour les 5 doigts et 9 ancres RBF).
*   Cette matrice est utilisée pour calculer le score de *toutes* les notes passées dans le contexte.

**Analyse Technique :**
1.  **Réduction de la Variance (Avantage PIG) :** Le dataset PIG est petit. En combinant toutes les observations (la note précédente, l'avant-dernière, etc.) pour apprendre cette unique matrice, on multiplie artificiellement la quantité de données d'entraînement par 2 ou 3. Les courbes RBF seront beaucoup plus lisses et robustes.
2.  **Stabilité de l'Inférence :** Lors du calcul Viterbi, le coût d'émission pour une position donnée est stable. Cela aide l'algorithme à "verrouiller" une position de main $x_t$ cohérente.
3.  **Sens Biologique :** Cela force le modèle à apprendre la "forme moyenne idéale" de la main, indépendamment du contexte musical.

**Justification :**

1.  **Cohérence avec la définition du Soft-Position :** L'émission dans ce modèle représente $P(\text{Note} | \text{Doigt}, \text{CentreMain})$. C'est une propriété anatomique statique (la longueur des doigts par rapport au poignet). Elle ne dépend pas de l'ordre $n-k$.
2.  **Robustesse statistique :** Avec le dataset PIG limité, apprendre une seule morphologie robuste vaut mieux que d'apprendre trois morphologies bruitées.
3.  **Rôle des Transitions :** Laissez les matrices de **Transition** ($T$) gérer la logique séquentielle ("après un 2, je mets un 3"). Laissez la matrice d'**Émission** ($W$) gérer la logique spatiale pure ("le 3 peut atteindre cette note si la main est là"). Ne mélangez pas les responsabilités.


## 3. Nature de la distribution soft

Puisque nous avons validé l'architecture **RBF (Bases Radiales)** à l'étape précédente, la question ici est précisément : **Comment déterminons-nous les poids ($w$) de ces bases radiales ?**

Sont-ils fixés par une équation mathématique rigide (Option A) ou sont-ils ajustés par l'intelligence artificielle sur les données (Option B) ?

Voici l'approche à adopter :

### Distribution Apprise (Trainable RBF Weights) :

**Le Principe :**
C'est l'approche **Data-Driven**.
Puisque nous avons choisi l'Option (A) précédente ("Un seul modèle partagé"), nous allons apprendre une seule matrice de morphologie $W$ de taille $(5 \text{ doigts} \times 9 \text{ ancres})$.

*   Pour chaque doigt, le modèle dispose de 9 curseurs (les poids des RBF).
*   L'algorithme d'entraînement va ajuster ces curseurs pour coller à la réalité du dataset PIG.

**Analyse Critique :**
1.  **Réalisme Biologique (Avantage Décisif) :** Le modèle va "sculpter" la forme de la main.
    *   Il apprendra que le doigt 2 a une zone de confort étroite.
    *   Il apprendra que le doigt 1 a une "traîne" (probabilité non nulle) très longue vers la gauche (extensions), mais une chute brutale vers la droite.
    *   C'est la seule option qui permet de capturer la vraie morphologie asymétrique de la main.
2.  **Faisabilité PIG :** Avec seulement $5 \times 9 = 45$ paramètres à apprendre au total pour l'émission, le dataset PIG est largement suffisant. Il n'y a quasiment aucun risque de sur-apprentissage.

**Nuances d'implémentation & importance critique de l'initialisation (Warm Start):**

1.  **Architecture :** Utiliser des poids apprenables : une matrice $W$ de taille $5 \times 9$.
2.  **Initialisation (Le rôle de l'Option A) :** Ne pas commencer pas avec des poids aléatoires. Initialiser la matrice $W$ en utilisant la formule Gaussienne $$ W_{f,k} = \exp\left( - \frac{(c_k - \mu_f)^2}{2\sigma_f^2} \right) $$ en utilisant des paramètres $\mu$ (moyenne) et $\sigma$ (écart-type) **différents pour chaque doigt**.
    *   Cela donne au modèle un point de départ "physiquement sain".
    *   L'entraînement ne fera ensuite qu'ajuster (tordre) cette courbe pour refléter la réalité asymétrique des pianistes du PIG dataset.

Cela garantit une convergence rapide, évite les aberrations, et capture la finesse biologique.


Voici les valeurs d'initialisation recommandées mour la main droite, il faut, evidentiellement, se souvenir que la main gauche est une "image mirror" de la main droite.

* A. Le Pouce (Doigt 1)
 Doigt le plus mobile, articulé à la base (articulation trapézo-métacarpienne). Sa position de repos est à gauche du centre de la main (Main Droite). Il possède une capacité unique d'adduction et de flexion qui lui permet de se déplacer latéralement sous la paume ("Thumb Under") pour atteindre des notes situées au centre, voire légèrement à droite de l'axe médian de la main.

    *   **Paramètres d'init :**
        *   $\mu \approx -4.0$ (Le centre de gravité moyen est à gauche).
        *   $\sigma \approx 4.0$ (Distribution très large et aplatie).
        *   **Résultat dans la matrice :**
            *   Probabilité élevée sur les ancres de gauche `[-9, -6, -3]`.
            *   Probabilité significative sur l'ancre centrale `[0]` (pour permettre l'apprentissage du passage du pouce).
            *   Probabilité faible mais non nulle sur l'ancre `[+3]`.
            *   Chute rapide vers zéro au-delà de `+3`.

* B. Les Doigts Centraux (2, 3, 4)
**Anatomie :** Doigts longs dont le mouvement principal est la flexion/extension. Leur écartement latéral (abduction/adduction) est limité par les ligaments intermétacarpiens.
    *   **Paramètres d'init :**
        *   **Doigt 3 (Majeur) :** $\mu = 0.0$ (Définit l'axe central de la main), $\sigma \approx 1.0$ (Distribution étroite, sert de pivot stable).
        *   **Doigt 2 (Index) :** $\mu \approx -2.0$ (Situé à gauche du majeur), $\sigma \approx 1.5$ (Assez mobile).
        *   **Doigt 4 (Annulaire) :** $\mu \approx +2.0$ (Situé à droite du majeur), $\sigma \approx 1.2$ (Anatomiquement moins indépendant, souvent lié au majeur ou au petit doigt).

* C. Le Petit Doigt (Doigt 5) **Anatomie :** Situé à l'extrémité droite de la main. Il possède une mobilité d'abduction importante pour aller chercher les extensions vers les aigus.
    *   **Paramètres d'init :**
        *   $\mu \approx +5.0$.
        *   $\sigma \approx 2.5$ (Plus mobile que les doigts centraux, permet d'atteindre les octaves).

## 4. Façon d’intégrer les soft positions dans le HMM

Puisque nous avons opté pour la Distribution Apprise (Trainable RBF) pour sculpter la morphologie de la main, la logique mathématique impose une direction claire :

### Remplacer entièrement les output*_prob

#### A. Changement des Paramètres (Simplification massive)
*   **Avant :** Nous avions des tenseurs énormes `(Doigt_prev, Doigt_cur, Intervalle=128)`. Pour l'ordre 3, c'était gigantesque.
*   **Maintenant :** Nous aurons une seule matrice `RBF_Weights` de taille `(5, 9)`.
    *   5 lignes (une par doigt).
    *   9 colonnes (poids pour chaque ancre RBF).
    *   Ces poids sont **appris** (Trainable), initialisés avec la logique asymétrique du pouce vue précédemment.

#### B. Changement du Viterbi (Calcul du Score)
Dans la boucle Viterbi, au moment de calculer `log_output_prob` :

1.  **Entrée :** On teste un état candidat $(f_t, x_t)$ pour une note de pitch $p_t$.
2.  **Calcul de la Position Relative :** $\delta = p_t - x_t$.
    *   *Exemple :* Note=64 (Mi), CentreMain=60 (Do). $\delta = +4$.
3.  **Activation RBF :** On calcule l'activation des 9 ancres pour la valeur $\delta$.
    *   Soit via interpolation linéaire (Option C de l'étape 1), soit via gaussiennes.
    *   On obtient un vecteur d'activation $\Phi$ de taille 9.
4.  **Produit Scalaire (Dot Product) :**
    $$ \text{Score}_{\text{emission}} = \sum_{k=1}^{9} W[f_t, k] \times \Phi[k] $$
5.  **Résultat :** Ce score remplace directement la valeur qu'on lisait auparavant dans la table `log_output1_prob`.

#### C. Gestion de la Polyphonie et des Accords
C'est le seul point où l'ancien système avait un avantage (il savait gérer des paires de notes simultanées).
Avec le nouvel algorithme, pour un accord, le modèle doit trouver une position $x_t$ unique qui maximise le score pour *toutes* les notes de l'accord simultanément.
*   $Score_{accord} = \sum_{\text{notes } i} (\text{Score RBF du doigt } f_i \text{ pour la position } x_t)$.
*   Cela force naturellement la main à se centrer au milieu de l'accord pour satisfaire tout le monde.

#### Analyse de la Précision et des Risques

En théorie, l'ancien modèle "Output 2nd/3rd order" capturait des corrélations subtiles entre doigts (ex: "Le 3 aime suivre le 2 sur un demi-ton").
En passant au Soft-Position avec une morphologie partagée, on perd cette corrélation *fine* entre doigts successifs dans le terme d'émission.

**La Solution : La Matrice de Transition**
Cette perte est compensée par la matrice de **Transition** ($P(f_t | f_{t-1})$) qui reste active.
*   La Transition gère l'**Agilité** (Le 3 aime suivre le 2).
*   L'Émission Soft gère la **Géométrie** (Le 3 est au milieu de la main).
*   L'Inertie gère la **Stabilité** (La main ne bouge pas).

Cette séparation des rôles est beaucoup plus propre et efficace pour l'apprentissage.

**Implémentation requise :**
1.  **Supprimer** le chargement et l'entraînement des matrices `log_output*_prob` dans `model.py` et `train.py`.
2.  **Ajouter** la matrice `RBF_weights` (5x9).
3.  **Réécrire** la partie "Emission" du kernel Viterbi Numba pour faire le produit scalaire RBF au lieu du lookup.
4.  **Ajouter** le calcul du coût d'inertie ($|x_t - x_{t-1}| \times \text{poids}$) dans la partie "Transition" du Viterbi.

## 5. Stratégie d’entraînement

C'est ici que le modèle va ajuster ses poids W (les courbes morphologiques des doigts) pour qu'ils correspondent à la réalité. Dans nos données d'entrainement, nous avons les Doigts ($f_t$), mais pas la Position de la Main ($x_t$).

La position de la main est une variable latente (cachée). Voici donc l'approche à adopter :


### EM (Expectation-Maximization) - variante "Viterbi Training" (Hard EM)

**Le Principe :**
C'est la méthode standard pour apprendre quand il manque des données (ici, $x$). C'est un processus cyclique :

1.  **Expectation (Deviner $x$) :** Avec les poids actuels, quelle est la position de main la plus probable que le pianiste a dû utiliser pour jouer ces doigts ?
2.  **Maximization (Améliorer $W$) :** Maintenant qu'on a "deviné" les positions, on met à jour les poids $W$ pour rendre ces positions encore plus probables.

Cette solution permet de découvrir la morphologie de la main sans jamais qu'un humain n'ait eu à annoter la position du poignet.

Plutôt que l'EM complet (Baum-Welch) qui calcule des nuages de probabilités flous (Soft EM), nous optons pour le **Viterbi Training (Hard EM)**. C'est plus rapide, plus simple à coder, et souvent plus efficace pour des distributions piquées comme la position de la main.

Voici l'algorithme précis à implémenter :

#### Phase 1 : Initialisation (Cruciale)
*   Initialiser la matrice $W$ ($5 \times 9$) avec la méthode Gaussienne asymétrique décrite précédemment.
*   Initialiser la matrice de Transition $T$ par comptage simple sur les données (comme HMM classique).

#### Phase 2 : La Boucle d'Entraînement

Pour chaque itération :

**1. Étape E (Alignment / Constrained Viterbi) :**
Pour chaque morceau du dataset d'entraînement :
*   On connaît la séquence de notes ($p_1...p_N$) et la **vraie** séquence de doigts ($f_1...f_N$).
*   On exécute un **Viterbi Contraint** : On cherche la meilleure séquence de positions de main ($x_1...x_N$) qui maximise le score, **en forçant le modèle à utiliser les vrais doigts annotés**.
    *   *Note :* C'est beaucoup plus rapide que le Viterbi de prédiction, car on ne teste pas tous les doigts. On teste seulement les positions $x$ possibles pour le doigt imposé.
*   On stocke les paires résultantes : $(\text{Doigt}, \text{Delta})$ où $\text{Delta} = p_t - x_t$.

**2. Étape M (Optimization / Update) :**
Maintenant, on a une liste géante de milliers de "Deltas" observés pour chaque doigt (générés par notre meilleure estimation à l'étape E).
*   On veut ajuster les poids $W$ des RBF pour que ces Deltas aient la probabilité maximale.
*   Cela revient à faire une **Régression Logistique** ou une simple **Descente de Gradient** sur les poids $W$.
    *   Loss = $-\sum \log(\text{ProduitScalaire}(W[f], \text{RBF}(\text{Delta})))$
*   On met à jour la matrice $W$.
*   (Optionnel) On peut aussi mettre à jour les coûts d'inertie (Transition) en observant les sauts $|x_t - x_{t-1}|$ générés.

#### Phase 3 : Convergence
On arrête quand les poids $W$ ne bougent plus beaucoup.

#### Avantages

1.  **Auto-Correction :** Au début, le modèle "devine" les positions de main un peu au hasard (guidé par l'initialisation Gaussienne). Mais dès qu'il aligne un peu mieux les mains, il obtient des données plus propres pour affiner la forme des doigts ($W$). Ce qui permet d'encore mieux aligner les mains au tour suivant. C'est un cercle vertueux.
2.  **Résolution du problème de Bach :** L'Étape E va se rendre compte que pour le passage de Bach, la seule façon de maximiser la probabilité avec les doigts annotés (si on a des exemples similaires stables) est de garder $x$ constant. Il va donc apprendre/renforcer les paramètres qui favorisent la stabilité.
3.  **Simplicité Mathématique :** L'Étape M est une optimisation convexe simple (facile à coder avec PyTorch, ou même à la main avec NumPy). L'Étape E est juste votre fonction Viterbi légèrement modifiée.

## 6. Coût de short-time (pénalité de croisements à vitesse élevée)


Cette approche remplace totalement l'ancien paramètre `short_time_cost` par un système hybride divisant le problème en deux régimes physiques distincts : la **Topologie (Notes simultanées)** et la **Cinématique (Notes séquentielles)**.

### 1. Principe Fondamental : Séparation des Régimes

L'ancien modèle appliquait une pénalité floue dès que deux notes étaient proches dans le temps. La nouvelle architecture distingue rigoureusement :
*   **$\Delta t \approx 0$ (Accords)** : C'est une contrainte de **Géométrie Statique**. La main n'a pas le temps de bouger entre les notes ; les doigts doivent respecter l'ordre des touches.
*   **$\Delta t > 0$ (Séquence)** : C'est une contrainte de **Dynamique**. La main se déplace, et le coût de ce déplacement dépend du temps disponible.

---

### 2. Régime 1 : La Topologie des Accords ($\Delta t < 0.03s$)

Pour les notes considérées comme simultanées (accords), nous n'utilisons plus de "coût" (valeur soustraite). Nous appliquons un **Masque Binaire Strict** (Filtre d'exclusion).

**Logique d'Implémentation :**
Dans la boucle Viterbi, lorsque le système détecte un cluster de notes simultanées (ex: Do et Sol joués ensemble) :

1.  **Unicité de la Position :** Le modèle force une position de main $x_t$ unique pour toutes les notes de l'accord.
2.  **Filtre de Croisement (Collision) :**
    On vérifie l'ordre des doigts par rapport à l'ordre des notes.
    *   Soit $P_1 < P_2$ (Pitch de la note 1 et 2).
    *   Soit $F_1, F_2$ les doigts associés.
    *   **Règle :** Si $F_1 > F_2$ (ex: Petit doigt sur note grave, Pouce sur note aiguë), l'état est déclaré **Invalide** (Probabilité $= 0$ ou Log-Prob $= -\infty$).
    *   *Note :* Ce filtre remplace avantageusement le `short_time_cost` car il élimine totalement les hypothèses physiquement impossibles de l'espace de recherche, au lieu de les pénaliser faiblement.

### 3. Régime 2 : La Cinématique Séquentielle ($\Delta t > 0.03s$)

Pour les notes qui se suivent (même très rapidement), nous **supprimons totalement** le `short_time_cost`. Il est remplacé par la **Modulation Temporelle de l'Inertie**.

Le contrôle du réalisme (Legato vs Staccato, trilles, traits rapides) est désormais géré par le coût de déplacement du centre de gravité de la main $x$.

**Formule du Coût de Transition Unifié :**

$$ \text{CoûtTotal} = \text{CoûtAgilité}(f_{t-1} \to f_t) + \lambda(\Delta t) \times |x_t - x_{t-1}| $$

*   **$\text{CoûtAgilité}$ :** Matrice de transition classique (5x5) qui capture la facilité des doigts à s'enchaîner (ex: 2-3 est facile).
*   **$|x_t - x_{t-1}|$ :** Distance euclidienne entre la position précédente de la main et la nouvelle.
*   **$\lambda(\Delta t)$ :** Coefficient de rigidité du bras, fonction du temps écoulé entre les notes.

**Comportement de la fonction $\lambda(\Delta t)$ :**

1.  **Jeu Rapide / Legato ($\Delta t \to 0$) :**
    *   $\lambda \approx \text{Max}$ (ex: 1.0).
    *   Le déplacement du bras coûte très cher.
    *   **Conséquence :** Le modèle refuse de changer $x$. Il est forcé d'utiliser l'agilité des doigts et la morphologie RBF (extensions) pour atteindre la note suivante. Si l'extension est impossible sans bouger la main, le chemin est rejeté naturellement par les RBF. Plus besoin de `short_time_cost`.

2.  **Jeu Lent / Staccato / Silence ($\Delta t$ grand) :**
    *   $\lambda \to 0$.
    *   Le déplacement du bras devient gratuit.
    *   **Conséquence :** Le modèle autorise la main à "sauter" (Hand Shift) pour se recentrer confortablement sur la nouvelle note. Cela permet les resets de main après une phrase ou sur des accords plaqués détachés.

### 4. Synthèse du Fonctionnement Global

Voici comment cette combinaison traite les situations critiques sans paramètres arbitraires :

| Situation | Ancien Modèle (avec `short_time_cost`) | Nouveau Modèle Unifié (Masque + Inertie Dynamique) |
| :--- | :--- | :--- |
| **Accord (Do+Sol)** | Pénalité fixe si doigts croisés. Risque d'accepter l'impossible si le reste du score est haut. | **Masque Strict :** Croisement impossible rejeté immédiatement. Position $x$ unifiée forcée. |
| **Trille Rapide (Do-Ré-Do-Ré)** | Pénalité `short_time` appliquée car $\Delta t$ petit. Risque de pénaliser un trille valide. | **Inertie Max :** Le coût de mouvement de main est élevé. Le modèle garde $x$ fixe et utilise l'agilité des doigts 2-3 (Transition) qui est favorable. Aucune pénalité parasite. |
| **Saut Rapide (Do -> Do Octave)** | Pénalité `short_time`. Le modèle hésite. | **Inertie Max :** Le bras ne peut pas bouger si vite. Le modèle vérifie si les RBF (Morphologie) du pouce et du petit doigt permettent l'octave avec $x$ fixe. Si oui (main large), c'est validé. Si non, c'est rejeté. |
| **Saut Staccato (Do ... Do Octave)** | Pas de pénalité (car $\Delta t$ grand). | **Inertie Nulle :** Le bras bouge gratuitement. Le modèle recentre $x$ sur chaque note. Doigté identique (ex: 2 ... 2) autorisé et probable. |

### Conclusion Technique

Cette architecture unifiée est mathématiquement supérieure car :
1.  Elle respecte la **physique des solides** pour les accords (pas d'interpénétration des doigts).
2.  Elle respecte la **mécanique des fluides/solides** pour le mouvement (relation Vitesse/Énergie).
3.  Elle élimine un hyperparamètre "magique" (`short_time_cost`) difficile à régler, au profit d'une loi comportementale continue ($\lambda(\Delta t)$).

