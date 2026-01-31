// Small CLI to run the ConvertPrecomputedMatchingsToBin function for a dataset.

#include <iostream>
#include <string>
#include <filesystem>
#include "src/convert.h"

int main(int argc, char** argv) {
    std::filesystem::path precomputed_dir = "../Results/Mappings/Precomputed";
    std::string db = "IMDB-16";
    std::filesystem::path data_dir = "../Data/ProcessedGraphs";

    // parse arguments: support flags -db, -precomputed_dir (or -pre), -data_dir (or -data)
    int pos = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-db") {
            if (i + 1 < argc) db = argv[++i];
            else {
                std::cerr << "Error: -db requires a value\n";
                return 1;
            }
        } else if (arg == "-precomputed_dir" || arg == "-pre") {
            if (i + 1 < argc) precomputed_dir = argv[++i];
            else {
                std::cerr << "Error: -precomputed_dir requires a value\n";
                return 1;
            }
        } else if (arg == "-data_dir" || arg == "-data") {
            if (i + 1 < argc) data_dir = argv[++i];
            else {
                std::cerr << "Error: -data_dir requires a value\n";
                return 1;
            }
        } else {
            // treat as positional: precomputed_dir, db, data_dir
            if (pos == 1) precomputed_dir = arg;
            else if (pos == 2) db = arg;
            else if (pos == 3) data_dir = arg;
            ++pos;
        }
    }

    // append db to precomputed dir path
    precomputed_dir /= db;

    std::cout << "Converting precomputed matchings from: " << precomputed_dir.string() << "\n";

    // convert graphs to own format if not already done
    std::filesystem::path bgf_file = data_dir / (db + ".bgf");
    if (!std::filesystem::exists(bgf_file)) {
        std::cout << "Converting graphs to BGF format in: " << data_dir.string() << "\n";

        GraphData<UDataGraph> graphs;
        //
        graphs.Load(graphs.graphData, (precomputed_dir / "graphs.txt").string(), GraphFormat::GRAPHLIST, db);
        SaveParams saveParams = {
            .graphPath = data_dir.string() + "/",
            .Name = db,
            .Format = GraphFormat::BGF
        };
        graphs.Save(saveParams);
    }
    else {
        std::cout << "Graph BGF file already exists: " << bgf_file.string() << "\n";
    }


    int ret = ConvertPrecomputedMatchingsToBin(precomputed_dir.string(), data_dir.string(), db);
    if (ret != 0) {
        std::cerr << "Conversion failed with code: " << ret << "\n";
        return ret;
    }
    std::cout << "Conversion finished successfully." << "\n";
    return 0;
}
