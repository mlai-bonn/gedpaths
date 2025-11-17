
#include "src/create_edit_paths.h"

#include <filesystem>
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>
#include <libGraph.h>


// source_id and target_id as args
int main(int argc, const char * argv[]) {
    // -db argument for the database
    std::string db = "MUTAG";
    // -processed argument for the processed data path
    std::string processed_graph_path = "../Data/ProcessedGraphs/";
    // -mappings argument for loading the mappings
    std::string mappings_path = "../Results/Mappings/";
    // -num_mappings argument for the number of valid mappings to create edit paths for
    int num_mappings = -1;
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

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-db" || std::string(argv[i]) == "-data" || std::string(argv[i]) == "-dataset" || std::string(argv[i]) == "-database") {
            db = argv[i+1];
            ++i;
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
        else if (std::string(argv[i]) == "-connected_only") {
            connected_only = true;
        }
        // add help
        else if (std::string(argv[i]) == "-help") {
            std::cout << "Create edit paths from GED mappings" << std::endl;
            std::cout << "Arguments:" << std::endl;
            std::cout << "-db | -data | -dataset | -database <database name>" << std::endl;
            std::cout << "-processed <processed data path>" << std::endl;
            std::cout << "-mappings <mappings path>" << std::endl;
            std::cout << "-num_mappings <number of mappings to consider>" << std::endl;
            std::cout << "-method <GED method name>" << std::endl;
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

    return create_edit_paths(db,
                             processed_graph_path,
                             mappings_path,
                             edit_path_output,
                             method,
                             num_mappings,
                             seed,
                             connected_only,
                             path_strategies,
                             source_id,
                             target_id);
}
