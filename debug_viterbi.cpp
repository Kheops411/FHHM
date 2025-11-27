#include "cpp/Code/FingeringHMM_v180925.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

// This is a near-direct copy of the Viterbi function from FingeringHMM_v180925.hpp
// with added print statements for debugging.
void Viterbi_Debug(FingeringHMM_2nd& hmm, PianoFingering& testData, int hand, const std::vector<int>& pos) {
    if(pos.size()<3){return;}

    std::vector<std::vector<double>> LP(5, std::vector<double>(5));

    std::cout << std::fixed << std::setprecision(10);

    // Initialization step (n=1)
    KeyPos keyInt_1_0 = SubtrKeyPos(PitchToKeyPos(testData.evts[pos[1]].pitch), PitchToKeyPos(testData.evts[pos[0]].pitch));
    if(keyInt_1_0.x < -hmm.widthX) keyInt_1_0.x = -hmm.widthX;
    if(keyInt_1_0.x > hmm.widthX) keyInt_1_0.x = hmm.widthX;
    int idx1 = 3 * (keyInt_1_0.x + hmm.widthX) + keyInt_1_0.y + 1;

    for(int kp=0; kp<5; ++kp) {
        for(int k=0; k<5; ++k) {
            LP[kp][k] = hmm.iniProb[hand].LP[kp] + hmm.trProb[hand][kp].LP[k] + hmm.outProb[hand][kp][k].LP[idx1];
        }
    }

    std::cout << "--- C++ Trellis at n=1 ---" << std::endl;
    for(int kp=0; kp<5; ++kp) {
        for (int k=0; k<5; ++k) {
            std::cout << LP[kp][k] << "\t";
        }
        std::cout << std::endl;
    }

    // Main loop
    for(int n=2; n<pos.size(); ++n) {
        std::vector<std::vector<double>> preLP = LP;
        double logP;
        KeyPos keyInt_n_n1=SubtrKeyPos(PitchToKeyPos(testData.evts[pos[n]].pitch),PitchToKeyPos(testData.evts[pos[n-1]].pitch));
        if(keyInt_n_n1.x<-hmm.widthX){keyInt_n_n1.x=-hmm.widthX;}
        if(keyInt_n_n1.x>hmm.widthX){keyInt_n_n1.x=hmm.widthX;}
        int idx1 = 3*(keyInt_n_n1.x+hmm.widthX)+keyInt_n_n1.y+1;

        KeyPos keyInt_n_n2=SubtrKeyPos(PitchToKeyPos(testData.evts[pos[n]].pitch),PitchToKeyPos(testData.evts[pos[n-2]].pitch));
        if(keyInt_n_n2.x<-hmm.widthX){keyInt_n_n2.x=-hmm.widthX;}
        if(keyInt_n_n2.x>hmm.widthX){keyInt_n_n2.x=hmm.widthX;}
        int idx2 = 3*(keyInt_n_n2.x+hmm.widthX)+keyInt_n_n2.y+1;

        bool shortTime=abs(testData.evts[pos[n]].ontime-testData.evts[pos[n-1]].ontime)<0.03;
        int delPitch=testData.evts[pos[n]].pitch-testData.evts[pos[n-1]].pitch;
        bool shortTime2=abs(testData.evts[pos[n]].ontime-testData.evts[pos[n-2]].ontime)<0.03;
        int delPitch2=testData.evts[pos[n]].pitch-testData.evts[pos[n-2]].pitch;

        for(int kp=0; kp<5; ++kp) {
            for(int k=0; k<5; ++k) {
                double max_log_prob = -1e100; // Large negative number
                for(int kpp=0; kpp<5; ++kpp) {
                    double st_cost = 0;
                    if(shortTime && ((hand==0 && (k-kp)*delPitch<0) || (hand==1 && (k-kp)*delPitch>0))) {
                        st_cost += hmm.shortTimeCost;
                    }
                    if(shortTime2 && ((hand==0 && (k-kpp)*delPitch2<0) || (hand==1 && (k-kpp)*delPitch2>0))) {
                        st_cost += hmm.shortTimeCost;
                    }

                    logP = preLP[kpp][kp] + hmm.trProb2[hand][kpp][kp].LP[k] +
                           hmm.w1 * hmm.outProb[hand][kp][k].LP[idx1] +
                           hmm.w2 * hmm.outProb2[hand][kpp][k].LP[idx2] + st_cost;

                    if(logP > max_log_prob) {
                        max_log_prob = logP;
                    }
                }
                LP[kp][k] = max_log_prob;
            }
        }

        // Print trellis at middle and end
        if (n == pos.size() / 2 || n == pos.size() - 1) {
            std::cout << "--- C++ Trellis at n=" << n << " ---" << std::endl;
            for(int r=0; r<5; ++r) {
                for (int c=0; c<5; ++c) {
                    std::cout << LP[r][c] << "\t";
                }
                std::cout << std::endl;
            }
        }
    }
}


int main(int argc, char** argv) {
    if(argc < 3){ cout << "Usage: $./this param_file score_file" << endl; return -1; }
    string paramFile = string(argv[1]);
    string scoreFile = string(argv[2]);

    FingeringHMM_2nd hmm;
    hmm.ReadParamFile(paramFile);

    PianoFingering pf;
    pf.ReadFile(scoreFile);
    pf.TimeDepPitchOrder();
    pf.ConvertFingerNumberToInt();

    // Right Hand
    PianoFingering pf_rh = pf;
    pf_rh.SelectHandByFingerNum(0);
    vector<int> pos_rh;
    for(int n=0; n<pf_rh.evts.size(); ++n) pos_rh.push_back(n);
    cout << "--- RIGHT HAND ---" << endl;
    Viterbi_Debug(hmm, pf_rh, 0, pos_rh);

    return 0;
}
