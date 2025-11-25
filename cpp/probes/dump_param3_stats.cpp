#include "FingeringHMM_v180925.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if(argc < 2) return 1;

    // Instantiate the 3rd Order HMM Class
    FingeringHMM_3rd hmm;
    hmm.ReadParamFile(argv[1]);

    std::cout << std::setprecision(10);

    // Dump a specific 3rd-order transition:
    // Hand 0 (RH), Prev3=0(1), Prev2=1(2), Prev1=2(3) -> Curr=3(4)
    // Indices in C++ vector: [hand][prev3][prev2][prev1].P[curr]
    double val = hmm.trProb3[0][0][1][2].P[3];

    std::cout << "TR3_SAMPLE " << val << "\n";

    // Dump 3rd order output prob sample
    // Hand 0, Prev3=0, Curr=0, dx=0(15+0), dy=0
    // Index = 3*(15)+1 = 46
    double out_val = hmm.outProb3[0][0][0].P[46];
    std::cout << "OUT3_SAMPLE " << out_val << "\n";

    return 0;
}
