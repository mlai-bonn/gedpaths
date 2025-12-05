// filepath: /home/florian/Documents/CodeProjectsGit/GNNGED/convert_precomputed.cpp
// Small CLI to run the ConvertPrecomputedMatchingsToBin function for a dataset.

#include <iostream>
#include <string>
#include "src/convert.h"

int main(int argc, char** argv) {
    std::string precomputed_dir = "../Results/Mappings/Precomputed/";
    std::string db = "IMDB-16";
    std::string data_dir = "../Data/ProcessedGraphs/";

    if (argc > 1) precomputed_dir = argv[1];
    if (argc > 2) db = argv[2];
    if (argc > 3) data_dir = argv[3];

    // ad db to precomputed dir path
    precomputed_dir += "/" + db + "/";


    std::cout << "Converting precomputed matchings from: " << precomputed_dir << "\n";

    // convert graphs to own format if not already done
    if (!std::filesystem::exists(data_dir + "/" + db + ".bgf")) {
        std::cout << "Converting graphs to BGF format in: " << data_dir << "\n";

        GraphData<UDataGraph> graphs;
        //
        graphs.Load(graphs.graphData, precomputed_dir + "graphs.txt", GraphFormat::GRAPHLIST, db);
        SaveParams saveParams = {
            .graphPath = data_dir,
            .Name = db,
            .Format = GraphFormat::BGF
        };
        graphs.Save(saveParams);
    }
    else {
        std::cout << "Graph BGF file already exists: " << data_dir + "/" + db + ".bgf" << "\n";
    }


    int ret = ConvertPrecomputedMatchingsToBin(precomputed_dir, data_dir, db);
    if (ret != 0) {
        std::cerr << "Conversion failed with code: " << ret << "\n";
        return ret;
    }
    std::cout << "Conversion finished successfully.\n";
    return 0;
}

