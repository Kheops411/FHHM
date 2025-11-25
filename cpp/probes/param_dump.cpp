#include "FingeringHMM_v180925.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <param_file>" << std::endl;
        return 1;
    }
    FingeringHMM_2nd hmm;
    hmm.ReadParamFile(argv[1]);

    std::cout << std::fixed << std::setprecision(15);

    // --- Dump specific probabilities ---

    // Initial probability: Right Hand, finger 2 (index 1)
    std::cout << "INIT_R_2 " << hmm.iniProb[0].P[1] << "\n";

    // Transition (1st order): Right Hand, finger 2 -> 3 (indices 1 -> 2)
    std::cout << "TR1_R_2_3 " << hmm.trProb[0][1].P[2] << "\n";

    // Transition (2nd order): Right Hand, finger 1 -> 2 -> 3 (indices 0 -> 1 -> 2)
    std::cout << "TR2_R_1_2_3 " << hmm.trProb2[0][0][1].P[2] << "\n";

    // Output probability (1st order): Right Hand, finger 1 -> 2, interval C4->E4 (dx=2, dy=0)
    // C++ index formula: 3*(dx+widthX)+dy+1, where widthX=15
    // Index = 3*(2+15)+0+1 = 52
    int idx_C4E4 = 52;
    std::cout << "OUT1_R_1_2_C4E4 " << hmm.outProb[0][0][1].P[idx_C4E4] << "\n";

    // Output probability (2nd order): Left Hand, f_n-2 = -1, f_n = -2, interval C4->G3 (dx=-3, dy=0)
    // Index = 3*(-3+15)+0+1 = 37
    int idx_C4G3 = 37;
    std::cout << "OUT2_L_1_2_C4G3 " << hmm.outProb2[1][0][1].P[idx_C4G3] << "\n";

    return 0;
}
