#include "../Code/FingeringHMM_v180925.hpp"
#include <iostream>
#include <iomanip>

// Cette sonde exécute Viterbi mais imprime les Log-Probs (LP) 
// pour chaque note et chaque doigt AVANT de faire le backtracking.
// Cela nous permet de voir exactement où Python et C++ divergent.

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./debug_viterbi_step <param_file> <score_file>" << std::endl;
        return 1;
    }

    string paramFile = argv[1];
    string scoreFile = argv[2];

    FingeringHMM_2nd hmm;
    hmm.ReadParamFile(paramFile);
    hmm.testData.clear();
    
    PianoFingering fingering;
    fingering.ReadFile(scoreFile);
    hmm.testData.push_back(fingering);

    // Force TimeDepPitchOrder pour être sûr de l'ordre des notes
    // Note: Dans le code original, cela est appelé avant Viterbi
    for(int i=0; i<hmm.testData.size(); i++) {
        for(int n=0; n<hmm.testData[i].evts.size(); n++) {
            hmm.testData[i].evts[n].ext1 = n;
        }
        hmm.testData[i].TimeDepPitchOrder();
    }

    // On va "ouvrir" la fonction Viterbi ici pour dumper les valeurs
    // C'est une copie simplifiée de la logique de Viterbi(0) (Main Droite) pour le debug
    int hand = 0; // Right Hand only for debug
    int i = 0; // First file
    
    // Préparation des données comme dans Viterbi()
    hmm.testData[i].SetChannelFromFingerNumber(); // Important si le fichier a des infos de main
    hmm.testData[i].ConvertFingerNumberToInt();
    
    vector<int> pos;
    for(int n=0; n<hmm.testData[i].evts.size(); n++) {
        // Filtrage main droite standard
        if(hand==0 && hmm.testData[i].evts[n].channel==0){ pos.push_back(n); }
    }

    if (pos.size() < 2) {
        cout << "Not enough notes for RH" << endl;
        return 0;
    }

    // --- DÉBUT LOGIQUE VITERBI (Instrumentée) ---
    vector<double> LP(5);
    vector<vector<double> > amax;
    amax.resize(pos.size());
    amax[0].resize(5);

    cout << setprecision(8) << fixed;

    // Initialization (Step 0)
    cout << "--- STEP 0 (Init) ---" << endl;
    cout << "Note: " << hmm.testData[i].evts[pos[0]].pitch << endl;
    for(int k=0; k<5; k++) {
        LP[k] = hmm.iniProb[hand].LP[k];
        cout << "F" << k+1 << ": " << LP[k] << endl;
    }

    // Recursion
    for(int n=1; n<pos.size(); n++) {
        // Limitons le debug aux 10 premières notes pour ne pas spammer
        if (n > 10) break; 

        cout << "--- STEP " << n << " ---" << endl;
        cout << "Note: " << hmm.testData[i].evts[pos[n]].pitch << endl;

        amax[n].resize(5);
        vector<double> preLP(LP);
        
        KeyPos kpCurrent = PitchToKeyPos(hmm.testData[i].evts[pos[n]].pitch);
        KeyPos kpPrev = PitchToKeyPos(hmm.testData[i].evts[pos[n-1]].pitch);
        KeyPos keyInt = SubtrKeyPos(kpCurrent, kpPrev);
        
        if(keyInt.x < -hmm.widthX) keyInt.x = -hmm.widthX;
        if(keyInt.x > hmm.widthX) keyInt.x = hmm.widthX;

        // Calcul index Lattice pour vérification
        int latticeIdx = 3*(keyInt.x + hmm.widthX) + keyInt.y + 1;
        cout << "Lattice Delta: dx=" << keyInt.x << " dy=" << keyInt.y << " idx=" << latticeIdx << endl;

        bool shortTime = abs(hmm.testData[i].evts[pos[n]].ontime - hmm.testData[i].evts[pos[n-1]].ontime) < 0.03;
        int delPitch = hmm.testData[i].evts[pos[n]].pitch - hmm.testData[i].evts[pos[n-1]].pitch;

        cout << "ShortTime: " << shortTime << " DelPitch: " << delPitch << endl;

        for(int k=0; k<5; k++) { // k est le doigt ACTUEL (0-4)
            // Calculer la meilleure transition vers k
            double bestVal = -1e200; // init très bas
            
            // On recalcule temporairement pour affichage
            for(int kp=0; kp<5; kp++) { // kp est le doigt PRECEDENT (0-4)
                // Attention: dans le code original, kp boucle de 1 à 5 si on regarde Viterbi(), 
                // mais les indices de tableau sont 0..4.
                // Le code original C++ Viterbi boucle: for(int kp=1;kp<5;kp+=1) pour trouver le max
                // MAIS il calcule le premier terme (kp=0) hors de la boucle.
                // Reconstruisons la logique exacte.
                
                double trans = hmm.trProb[hand][kp].LP[k];
                double emiss = hmm.w1 * hmm.outProb[hand][kp][k].LP[latticeIdx];
                
                // Penalités
                double penalty = 0;
                // Logique C++: ((shortTime&&((hand==0&&(k-kp)*delPitch<0)||(hand==1&&(k-kp)*delPitch>0)))? shortTimeCost:0)
                // k et kp sont des indices 0-4. Le code original compare les indices.
                if (shortTime) {
                    if (hand==0 && (k-kp)*delPitch < 0) penalty = hmm.shortTimeCost;
                }

                double val = preLP[kp] + trans + emiss + penalty;
                
                if (val > bestVal) bestVal = val;
            }
            LP[k] = bestVal;
            cout << "  -> Best LP to Finger " << k+1 << ": " << LP[k] << endl;
        }
    }

    return 0;
}