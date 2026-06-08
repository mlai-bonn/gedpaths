
#include "src/create_edit_paths.h"

#include <filesystem>
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>
#include <string>
#include <libGraph.h>

namespace {

using PathStrategySet = std::vector<std::string>;

std::vector<std::string> GetAllDatasetsForMethod(const std::string& mappings_path, const std::string& method) {
    const std::filesystem::path method_path = std::filesystem::path(mappings_path) / method;
    if (!std::filesystem::exists(method_path)) {
        throw std::runtime_error("Mappings directory does not exist: " + method_path.string());
    }
    if (!std::filesystem::is_directory(method_path)) {
        throw std::runtime_error("Mappings path is not a directory: " + method_path.string());
    }

    std::vector<std::string> datasets;
    for (const auto& entry : std::filesystem::directory_iterator(method_path)) {
        if (entry.is_directory()) {
            datasets.push_back(entry.path().filename().string());
        }
    }
    std::ranges::sort(datasets);
    return datasets;
}

std::vector<PathStrategySet> GetCanonicalPathStrategies() {
    return {
        {"Random"},
        {"Random", "DeleteIsolatedNodes"},
        {"InsertEdges", "DeleteIsolatedNodes"},
        {"DeleteEdges", "DeleteIsolatedNodes"}
    };
}

}

// source_id and target_id as args
int main(int argc, const char * argv[]) {
    // -db argument for the database
    std::string db = "MUTAG";
    // -processed argument for the processed data path
    std::string processed_graph_path = "../Data/ProcessedGraphs/";
    // -mappings argument for loading the mappings
    std::string mappings_path = "../Results/Mappings/";
    // -num_mappings argument for the number of valid mappings to create edit paths for
    int num_mappings = 5000;
    // -seed argument for the random seed
    int seed = 42;
    // -edit_paths argument for the path to store the edit paths
    std::string edit_path_output = "../Results/";
    // -t arguments for the threads to use
    int num_threads = 1;
    // -method
    std::string method = "REFINE";
    std::vector<std::string> path_strategies = {"Random"};
    bool connected_only = false;

    int source_id = -1;
    int target_id = -1;
    std::string method_options;
    bool use_all_db = false;
    bool use_all_path_strategies = false;
    bool db_was_set = false;
    bool path_strategy_was_set = false;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-db" || std::string(argv[i]) == "-data" || std::string(argv[i]) == "-dataset" || std::string(argv[i]) == "-database") {
            if (use_all_db) {
                std::cerr << "Error: -db cannot be combined with -all_db." << std::endl;
                return 1;
            }
            db = argv[i+1];
            db_was_set = true;
            ++i;
        }
        else if (std::string(argv[i]) == "-all_db") {
            if (db_was_set) {
                std::cerr << "Error: -all_db cannot be combined with -db." << std::endl;
                return 1;
            }
            use_all_db = true;
        }
        else if (std::string(argv[i]) == "-processed") {
            processed_graph_path = argv[i+1];
            ++i;
        }
        else if (std::string(argv[i]) == "-mappings") {
            mappings_path = argv[i+1];
            ++i;
        }
        else if (std::string(argv[i]) == "-method") {
            method = argv[i+1];
            ++i;
        }
        else if (std::string(argv[i]) == "-num_mappings") {
            num_mappings = std::stoi(argv[i+1]);
            ++i;
        }
        // if in source_id or source
        else if (std::string(argv[i]) == "-source_id" || std::string(argv[i]) == "-source") {
            source_id = std::stoi(argv[i+1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-target_id" || std::string(argv[i]) == "-target") {
            target_id = std::stoi(argv[i+1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-path_strategy") {
            if (use_all_path_strategies) {
                std::cerr << "Error: -path_strategy cannot be combined with -all_path_strategies." << std::endl;
                return 1;
            }
            path_strategy_was_set = true;
            path_strategies.clear();
            std::string path_strategy;
            // get all entries until next arg (starting with -) or end of argv
            ++i;
            while (i < argc && std::string(argv[i]).rfind('-', 0) != 0) {
                path_strategy = argv[i];
                path_strategies.push_back(path_strategy);
                ++i;
            }
            --i;
            // if no path strategies given, use Random
            if (path_strategies.empty()) {
                path_strategies.push_back("Random");
            }

        }
        else if (std::string(argv[i]) == "-all_path_strategies") {
            if (path_strategy_was_set) {
                std::cerr << "Error: -all_path_strategies cannot be combined with -path_strategy." << std::endl;
                return 1;
            }
            use_all_path_strategies = true;
        }
        else if (std::string(argv[i]) == "-connected_only") {
            connected_only = true;
        }
        // add help
        else if (std::string(argv[i]) == "-help") {
            std::cout << "Create edit paths from GED mappings" << std::endl;
            std::cout << "Arguments:" << std::endl;
            std::cout << "-db | -data | -dataset | -database <database name>" << std::endl;
            std::cout << "-all_db <use all datasets found in <mappings>/<method>/>" << std::endl;
            std::cout << "-processed <processed data path>" << std::endl;
            std::cout << "-mappings <mappings path>" << std::endl;
            std::cout << "-num_mappings <number of mappings to consider>" << std::endl;
            std::cout << "-method <GED method name>" << std::endl;
            std::cout << "-path_strategy <list of path strategy tokens>" << std::endl;
            std::cout << "-all_path_strategies <use Random, Random DeleteIsolatedNodes, InsertEdges DeleteIsolatedNodes, DeleteEdges DeleteIsolatedNodes>" << std::endl;
            std::cout << "-source_id <source graph id>" << std::endl;
            std::cout << "-target_id <target graph id>" << std::endl;
            std::cout << "-help <show this help message>" << std::endl;
             return 0;
        }
        else {
            std::cout << "Unknown argument: " << argv[i] << std::endl;
            return 1;
        }
    }

    std::vector<std::string> databases = {db};
    if (use_all_db) {
        try {
            databases = GetAllDatasetsForMethod(mappings_path, method);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    std::vector<PathStrategySet> strategy_sets;
    if (use_all_path_strategies) {
        strategy_sets = GetCanonicalPathStrategies();
    } else {
        strategy_sets.push_back(path_strategies);
    }

    bool processed_any_dataset = false;
    for (const auto& current_db : databases) {
        const std::filesystem::path mapping_file = std::filesystem::path(mappings_path) / method / current_db / (current_db + "_ged_mapping.bin");
        if (!std::filesystem::exists(mapping_file)) {
            std::cerr << "Warning: mapping file not found for dataset '" << current_db
                      << "' at " << mapping_file << ". Skipping." << std::endl;
            continue;
        }

        processed_any_dataset = true;
        for (const auto& current_path_strategies : strategy_sets) {
            std::cout << "Running CreatePaths for dataset '" << current_db
                      << "' with strategy '" << EditPathStrategiesToStringShort(StringsToEditPathStrategies(current_path_strategies))
                      << "'." << std::endl;

            std::string current_mappings_path = mappings_path;
            std::string current_edit_path_output = edit_path_output;
            const int status = create_edit_paths(current_db,
                                                 processed_graph_path,
                                                 current_mappings_path,
                                                 current_edit_path_output,
                                                 method,
                                                 num_mappings,
                                                 seed,
                                                 connected_only,
                                                 current_path_strategies,
                                                 source_id,
                                                 target_id);
            if (status != 0) {
                return status;
            }
        }
    }

    if (!processed_any_dataset) {
        std::cerr << "Error: no datasets with mapping files found for method '" << method << "'." << std::endl;
        return 1;
    }

    return 0;
}
