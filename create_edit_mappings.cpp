// define gurobi
#define GUROBI
// use gedlib
#define GEDLIB

#include <filesystem>
#include <iostream>
#include <vector>
#include "GraphDataStructures/GraphBase.h"
#include "src/env/ged_env.hpp"
#include <LoadSave.h>
#include "LoadSaveGraphDatasets.h"
#include "Algorithms/GED/GEDLIBWrapper.h"
#include "Algorithms/GED/GEDFunctions.h"
// OpenMP for parallel execution
#include "src/create_edit_mappings.h"

#include <omp.h>
#include <memory>
#include <atomic>
#include <chrono>
#include <iomanip>



// source_id and target_id as args
int main(const int argc, const char * argv[]) {
    // -db argument for the database (accepts several synonyms)
    std::string db = "MUTAG";
    // -raw argument for the raw data path
    std::string input_path = "../Data/Graphs/";
    // -processed argument for the processed data path
    std::string processed_graph_path = "../Data/ProcessedGraphs/";
    // -mappings argument for the path to store the mappings
    std::string output_path = "../Results/Mappings/";
    // -t arguments for the threads to use
    int num_threads = 1;
    // -method
    auto method = "F2";
    std::string method_options = "";
    auto ged_method = GEDMethodFromString(method);
    // -cost
    auto cost = "CONSTANT";
    auto edit_cost = EditCostsFromString(cost);
    // -s
    auto seed = 42;
    // -ids_path
    std::string graph_ids_path;
    // -num_pairs to randomly sample from the dataset and create mappings for (-1 for all)
    int num_pairs = 5000;


    // Add single source/target arguments
    int single_source = -1;
    int single_target = -1;

    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "-db" || std::string(argv[i]) == "-data" || std::string(argv[i]) == "-dataset" || std::string(argv[i]) == "-database") {
            db = argv[i+1];
        }
        else if (std::string(argv[i]) == "-raw") {
            input_path = argv[i+1];
        }
        else if (std::string(argv[i]) == "-processed") {
            processed_graph_path = argv[i+1];
        }
        else if (std::string(argv[i]) == "-mappings") {
            output_path = argv[i+1];
        }
        else if (std::string(argv[i]) == "-t") {
            num_threads = std::stoi(argv[i+1]);
        }
        else if (std::string(argv[i]) == "-method") {
            method = argv[i+1];
            ged_method = GEDMethodFromString(method);
        }
        else if (std::string(argv[i]) == "-method_options") {
            // read method options in format option value option value ... until next  leading - or end of argv
            int counter = 0;
            while (i + 1 < argc && std::string(argv[i+1]).rfind('-', 0) != 0) {
                // if current argv is option add -- prefix otherwise just add value
                if (counter % 2 == 0) {
                    method_options += "--" + std::string(argv[i+1]) + " ";
                }
                else {
                    method_options += std::string(argv[i+1]) + " ";
                }
                counter++;
                i++;
            }
        }
        else if (std::string(argv[i]) == "-cost") {
            cost = argv[i+1];
            edit_cost = EditCostsFromString(cost);
        }
        else if (std::string(argv[i]) == "-seed") {
            seed = std::stoi(argv[i+1]);
        }
        else if (std::string(argv[i]) == "-ids_path") {
            graph_ids_path = argv[i+1];
        }
        else if (std::string(argv[i]) == "-num_pairs") {
            num_pairs = std::stoi(argv[i+1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-single_source") {
            single_source = std::stoi(argv[i+1]);
        }
        else if (std::string(argv[i]) == "-single_target") {
            single_target = std::stoi(argv[i+1]);
        }
        // add help
        else if (std::string(argv[i]) == "-help") {
            std::cout << "Create edit mappings for a given database/dataset" << std::endl;
            std::cout << "Arguments:" << std::endl;
            std::cout << "-db | -data | -dataset | -database <database name>" << std::endl;
            std::cout << "-raw <raw data path where db can be found>" << std::endl;
            std::cout << "-processed <processed data path>" << std::endl;
            std::cout << "-mappings <mappings path>" << std::endl;
            std::cout << "-help <show this help message>" << std::endl;
            std::cout << "Usage: " << argv[0] << " -db <database name> -raw <raw data path where db can be found> -processed <processed data path> -mappings <mappings path>" << std::endl;
            return 0;
        }
        else if (std::string(argv[i]) == "-") {
            // do nothing for lone -
        }
        // if -something else then print error
        else if (std::string(argv[i]).rfind('-', 0) == 0) {
            std::cout << "Unknown argument: " << argv[i] << std::endl;
            return 1;
        }

    }

    // create mapping output directory
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directory(output_path);
    }
    // create folder for method under output path
    output_path = output_path + method + "/";

    std::filesystem::create_directory(output_path + "/");
    std::filesystem::create_directory(output_path + "/" + db + "/");
    std::filesystem::create_directory(output_path + "/" + db + "/tmp/");


    return create_edit_mappings(db, output_path, input_path, processed_graph_path,
        edit_cost, ged_method, method_options, graph_ids_path, num_pairs, num_threads, seed, single_source, single_target);
}
