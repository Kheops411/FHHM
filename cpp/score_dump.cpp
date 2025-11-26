#include "Code/PianoFingering_v170101_2.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: ./bin file.txt\n"; return 1; }

    PianoFingering pf;
    pf.ReadFile(argv[1]);

    // Mimic the exact pipeline:
    // 1. Select Right Hand (0)
    pf.SelectHandByFingerNum(0);
    // 2. Convert Strings ("4_1") to Ints (4)
    pf.ConvertFingerNumberToInt();
    // 3. Apply Time ordering (essential context)
    pf.TimeDepPitchOrder();

    // Dump: Index | OnTime | Pitch | Finger(Int)
    for(size_t i=0; i<pf.evts.size(); ++i) {
        std::cout << i << " "
                  << pf.evts[i].ontime << " "
                  << pf.evts[i].pitch << " "
                  << pf.evts[i].finger << "\n";
    }
    return 0;
}
