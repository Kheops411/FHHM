#include "FingeringHMM_v180925.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 3) return 1;

    FingeringHMM_2nd hmm;
    hmm.w1 = 0.5;
    hmm.w2 = 0.5;
    hmm.shortTimeCost = -5.0;
    hmm.ReadParamFile(argv[1]);

    PianoFingering pf;
    pf.ReadFile(argv[2]);
    pf.SelectHandByFingerNum(0);
    pf.TimeDepPitchOrder();

    hmm.testData.clear();
    hmm.testData.push_back(pf);
    hmm.Viterbi(0); // This will now print the matrix at n=5

    return 0;
}
