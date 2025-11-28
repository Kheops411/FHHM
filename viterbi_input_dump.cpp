
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include "cpp/Code/PianoFingering_v170101_2.hpp"
#include "cpp/Code/Midi_v170101.hpp"

// We need to provide a definition for this C++ function, otherwise linking will fail.
void InitPitchName(){};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./viterbi_input_dump <score_file.txt>" << std::endl;
        return 1;
    }

    std::string score_file = argv[1];

    PianoFingering fingering;
    fingering.ReadFile(score_file);

    // Filter for right hand (hand=0)
    fingering.SelectHandByFingerNum(0);

    // Apply the sorting logic
    fingering.TimeDepPitchOrder();

    // Print the fields that are used by the Viterbi algorithm
    std::cout << std::fixed << std::setprecision(6);
    for (const auto& evt : fingering.evts) {
        std::cout << evt.ID << "\t" << evt.ontime << "\t" << evt.pitch << std::endl;
    }

    return 0;
}
