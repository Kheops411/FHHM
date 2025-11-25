#include "FingeringHMM_v180925.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <score_file>" << std::endl;
        return 1;
    }

    PianoFingering pf;
    pf.ReadFile(argv[1]);

    // We are testing the Right Hand ordering, which corresponds to finger numbers > 0
    pf.SelectHandByFingerNum(0);

    // This is the critical step that reorders the notes
    pf.TimeDepPitchOrder();

    std::cout << std::fixed << std::setprecision(6);
    for(size_t i = 0; i < pf.evts.size(); ++i) {
        // Dump the index, ontime, and pitch of the reordered notes
        std::cout << i << " " << pf.evts[i].ontime << " " << pf.evts[i].pitch << "\n";
    }

    return 0;
}
