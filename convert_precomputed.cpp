// filepath: /home/florian/Documents/CodeProjectsGit/GNNGED/convert_precomputed.cpp
// Small CLI to run the ConvertPrecomputedMatchingsToBin function for a dataset.

#include <iostream>
#include <string>
#include "src/convert.h"

int main(int argc, char** argv) {
    std::string precomputed_dir = "../Results/Mappings/Precomputed/LINUX";
    std::string db = "LINUX";
    std::string out_root = "../Results/Mappings/Precomputed/"; // will write into out_root/db/
    std::string data_dir = "../Data/ProcessedGraphs/";

    if (argc > 1) precomputed_dir = argv[1];
    if (argc > 2) out_root = argv[2];
    if (argc > 3) db = argv[3];

    std::cout << "Converting precomputed matchings from: " << precomputed_dir << "\n";
    std::cout << "Output root: " << out_root << " db: " << db << "\n";

    // convert graphs to own format
    GraphData<UDataGraph> graphs;
    //
    graphs.Load(graphs.graphData, precomputed_dir + "/graphs.txt", GraphFormat::GRAPHLIST);
    SaveParams saveParams = {
        .graphPath = data_dir,
        .Name = db,
        .Format = GraphFormat::BGF
    };
    graphs.Save(saveParams);


    int ret = ConvertPrecomputedMatchingsToBin(precomputed_dir, db, out_root);
    if (ret != 0) {
        std::cerr << "Conversion failed with code: " << ret << "\n";
        return ret;
    }
    std::cout << "Conversion finished successfully.\n";
    return 0;
}

