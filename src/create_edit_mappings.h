//
// Created by florian on 28.10.25.
//

// define gurobi
#define GUROBI
// use gedlib
#define GEDLIB

#ifndef GEDPATHS_CREATE_EDIT_MAPPINGS_H
#define GEDPATHS_CREATE_EDIT_MAPPINGS_H
#include <utility>
#include <vector>
#include <libGraph.h>
#include <src/env/ged_env.hpp>

int create_edit_mappings(const std::string& db,
                            const std::string& output_path,
                            const std::string& input_path,
                            const std::string& processed_graph_path,
                          ged::Options::EditCosts edit_cost,
                          ged::Options::GEDMethod ged_method,
                          const std::string& method_options = "",
                          const std::string& graph_ids_path = "",
                          int num_pairs = -1,
                          int num_threads = 1,
                          int seed = 42,
                          int single_source = -1,
                          int single_target = -1);

GEDEvaluation<UDataGraph> create_edit_mappings_single(INDEX source_id, INDEX target_id, GraphData<UDataGraph>& graphs, ged::Options::EditCosts edit_cost, ged::Options::GEDMethod ged_method, const std::string& method_options = "", bool print = false);


inline GEDEvaluation<UDataGraph> create_edit_mappings_single(INDEX source_id, INDEX target_id, GraphData<UDataGraph>& graphs, ged::Options::EditCosts edit_cost, ged::Options::GEDMethod ged_method, const std::string& method_options, bool print) {
    if (source_id >= graphs.graphData.size() || target_id >= graphs.graphData.size()) {
        std::cerr << "Single source/target IDs out of range: " << source_id << ", " << target_id << std::endl;
        exit(1);
    }
    std::pair<INDEX, INDEX> pair = std::minmax(source_id, target_id);
    std::vector<std::pair<INDEX, INDEX>> single_pair{pair};
    auto ged_env = ged::GEDEnv<ged::LabelID, ged::LabelID, ged::LabelID>();
    InitializeGEDEnvironment(ged_env, graphs, edit_cost, ged_method, method_options);
    std::vector<GEDEvaluation<UDataGraph>> results;
    ged_env.run_method(pair.first, pair.second);
    GEDEvaluation<UDataGraph> result = ComputeGEDResult(ged_env, graphs, pair.first, pair.second);
    if (print) {
        // Print result for debugging
        std::cout << "Computed mapping for pair (" << pair.first << ", " << pair.second << ")" << std::endl;
        // Optionally print more details if available
        std::cout << "Distance: " << result.distance << std::endl;
        std::cout << "Lower Bound: " << result.lower_bound << std::endl;
        std::cout << "Upper Bound: " << result.upper_bound << std::endl;
        // also return the mapping
        std::cout << "Node Mapping (source -> target):" << std::endl;
        for (INDEX i = 0; i < result.node_mapping.first.size(); ++i) {
            std::cout << "  " << i << " -> " << result.node_mapping.first[i] << std::endl;
        }
        std::cout << "  Target to Source:" << std::endl;
        for (INDEX i = 0; i < result.node_mapping.second.size(); ++i) {
            std::cout << "  " << i << " -> " << result.node_mapping.second[i] << std::endl;
        }
    }
    return result;

}


inline void fixInvalidMappings(std::vector<GEDEvaluation<UDataGraph>>& results,
                               GraphData<UDataGraph>& graphs,
                               ged::Options::EditCosts edit_cost,
                               ged::Options::GEDMethod ged_method,
                               const std::string& method_options) {

    // manipulate the method options as errors seem to come from parallelization in F2/F1
    std::string modified_method_options = method_options;
    if (ged_method == ged::Options::GEDMethod::F2 || ged_method == ged::Options::GEDMethod::F1) {
        // if --threads is in method_options with some number n, with k digits replace the number by 1 otherwise add --threads 1
        if (modified_method_options.find("--threads") != std::string::npos) {
            size_t pos = modified_method_options.find("--threads");
            size_t end_of_threads = pos + 9; // length of "--threads"
            // skip spaces
            while (end_of_threads < modified_method_options.size() && modified_method_options[end_of_threads] == ' ') {
                end_of_threads++;
            }
            // replace the number after --threads with 1
            size_t start_of_number = end_of_threads;
            while (end_of_threads < modified_method_options.size() && isdigit(modified_method_options[end_of_threads])) {
                end_of_threads++;
            }
            modified_method_options.replace(start_of_number, end_of_threads - start_of_number, "1");
        }
        else {
            modified_method_options += " --threads 1";
        }
    }
    // try to correct invalid mappings
    auto invalid_mappings = CheckResultsValidity(results);
    std::cout << "Found " << invalid_mappings.size() << " invalid mappings.\n";
    if (invalid_mappings.empty()) {
        return;
    }

    // recalulate the mappings for the invalid results
    std::cout << "Recalculating mappings for invalid results...\n";
    std::vector<std::pair<INDEX, GEDEvaluation<UDataGraph>>> fixed_results;
    for (const auto &id : invalid_mappings) {
        auto source_id = results[id].graph_ids.first;
        auto target_id = results[id].graph_ids.second;
        auto fixed_result = create_edit_mappings_single(source_id, target_id, graphs, edit_cost, ged_method, modified_method_options, true);
        if (CheckResultsValidity(std::vector<GEDEvaluation<UDataGraph>>{fixed_result}).empty()) {
            fixed_results.emplace_back(id, fixed_result);
            std::cout << "  Fixed mapping for result id " << id << " (Graph IDs: " << source_id << ", " << target_id << ")\n";
        }
        else {
            // if F1 fails try F2 and vice versa
            ged::Options::GEDMethod alternative_method = (ged_method == ged::Options::GEDMethod::F1) ? ged::Options::GEDMethod::F2 : ged::Options::GEDMethod::F1;
            auto alternative_fixed_result = create_edit_mappings_single(source_id, target_id, graphs, edit_cost, alternative_method, modified_method_options, true);
            if (CheckResultsValidity(std::vector<GEDEvaluation<UDataGraph>>{alternative_fixed_result}).empty()) {
                fixed_results.emplace_back(id, alternative_fixed_result);
                std::cout << "  Fixed mapping for result id " << id << " (Graph IDs: " << source_id << ", " << target_id << ") using alternative method.\n";
            }
            else {
                std::cout << "Try manual fixing for result id " << id << " (Graph IDs: " << source_id << ", " << target_id << ")\n";
                auto source_mapping = results[id].node_mapping.first;
                auto target_mapping = results[id].node_mapping.second;
                std::cout << "  Failed to fix mapping for result id " << id << " (Graph IDs: " << source_id << ", " << target_id << ")\n";
            }
        }
    }
    // replace invalid results with fixed results
    for (auto &[id, result] : fixed_results) {
        results[id] = result;
    }
    std::cout << "Finished recalculating invalid mappings.\n";
    std::cout << "Total fixed mappings: " << fixed_results.size() << " of " << invalid_mappings.size() << "\n";
}

inline void get_existing_mappings(const std::string& output_path,
                                  const std::string& db,
                                  GraphData<UDataGraph>& graphs,
                                  std::vector<std::pair<INDEX, INDEX>>& existing_graph_ids,
                                  std::vector<GEDEvaluation<UDataGraph>>& results) {
    // load mappings
    std::string mapping_file =output_path + "/" + db + "/" + db + "_ged_mapping.bin";
    std::string tmp_path = output_path + db + "/tmp/";
    if (std::filesystem::exists(mapping_file)) {
        BinaryToGEDResult(mapping_file, graphs, results);
        for (const auto& res : results) {
            existing_graph_ids.emplace_back(res.graph_ids);
        }
        // add possible existing graph ids from the tmp folder
        // Merge all mappings in tmp folder
        std::vector<GEDEvaluation<UDataGraph>> tmp_results;
        if (std::filesystem::exists(tmp_path) && !std::filesystem::is_empty(tmp_path)) {
            MergeBinaries(tmp_path, db + "_", graphs, tmp_results);
            // append new tmp_results
            for (const auto& res : tmp_results) {
                if (ranges::find(existing_graph_ids, res.graph_ids) == existing_graph_ids.end()) {
                    results.emplace_back(res);
                    existing_graph_ids.emplace_back(res.graph_ids);
                }
            }
        }

        // sort existing graph ids
        ranges::sort(existing_graph_ids, [](const std::pair<INDEX, INDEX>& a, const std::pair<INDEX, INDEX>& b) {
            return a.first == b.first ? a.second < b.second : a.first < b.first;
        });
        // sort also the results in the same way
        ranges::sort(results, [](const GEDEvaluation<UDataGraph>& a, const GEDEvaluation<UDataGraph>& b) {
            return a.graph_ids.first == b.graph_ids.first ? a.graph_ids.second < b.graph_ids.second : a.graph_ids.first < b.graph_ids.first;
        });
    }
    else if (std::filesystem::exists(tmp_path) && !std::filesystem::is_empty(tmp_path)) {
        //check whether there is a tmp folder with partial results (i.e., at least one file inside the folder)
        MergeGEDResults(tmp_path, output_path + "/" + db + "/", db + "_", graphs);
        // load merged results
        BinaryToGEDResult(mapping_file, graphs, results);
        for (const auto& res : results) {
            existing_graph_ids.emplace_back(res.graph_ids);
        }
    }
    // save the updated results back to binary
    GEDResultToBinary(output_path + "/" + db + "/", results);
}

inline void fix_invalid_mappings(const std::string& output_path,
                                  const std::string& db,
                                  GraphData<UDataGraph>& graphs,
                              ged::Options::EditCosts edit_cost,
                              ged::Options::GEDMethod ged_method,
                              const std::string& method_options) {
    // load mappings
    std::string mapping_file =output_path + "/" + db + "/" + db + "_ged_mapping.bin";
    // load merged results
    auto results = std::vector<GEDEvaluation<UDataGraph>>{};
    BinaryToGEDResult(mapping_file, graphs, results);
    fixInvalidMappings(results, graphs, edit_cost, ged_method, method_options);
    // save the updated results back to binary
    GEDResultToBinary(output_path + "/" + db + "/", results);
}

inline int create_edit_mappings(const std::string& db,
                                const std::string& output_path,
                                const std::string& input_path,
                                const std::string& processed_graph_path,
                                ged::Options::EditCosts edit_cost,
                                ged::Options::GEDMethod ged_method,
                                const std::string& method_options,
                                const std::string& graph_ids_path,
                                int num_pairs,
                                int num_threads,
                                int seed,
                                int single_source,
                                int single_target) {

    
    if (const bool success = LoadSaveGraphDatasets::PreprocessTUDortmundGraphData(db, input_path, processed_graph_path); !success) {
        std::cout << "Failed to create TU dataset" << std::endl;
        return 1;
    }
    GraphData<UDataGraph> graphs;
    LoadSaveGraphDatasets::LoadPreprocessedGraphData(db, processed_graph_path, graphs);
    std::vector<std::pair<INDEX, INDEX>> graph_pairs;
    std::vector<std::pair<INDEX, INDEX>> existing_pairs;

    auto results = std::vector<GEDEvaluation<UDataGraph>>{};
    get_existing_mappings(output_path, db, graphs, existing_pairs, results);
    fixInvalidMappings(results, graphs, edit_cost, ged_method, method_options);
    // save the updated results back to binary
    GEDResultToBinary(output_path + "/" + db + "/", results);


        // If db_ged_mapping.bin already exists load it and look for existing graph ids





    // If single_source and single_target are set, only compute and print that mapping
    if (single_source >= 0 && single_target >= 0) {
        auto result = create_edit_mappings_single(single_source, single_target, graphs, edit_cost, ged_method, method_options, true);
        return 0;
    }

    // Else generate random pairs (a large number)
    size_t max_number_of_pairs = 1000000;
    // store pairs inside set for faster lookup
    std::set<std::pair<INDEX, INDEX>> g_pairs;
    // set up random device
    auto gen = std::mt19937(seed);
    std::uniform_int_distribution<INDEX> dist(0, graphs.graphData.size() - 1);
    while (g_pairs.size() < max_number_of_pairs && g_pairs.size() != graphs.graphData.size() * (graphs.graphData.size() - 1) / 2) {
        // get random integer between 0 and graphs.GraphData.size() - 1
        INDEX id1 = dist(gen);
        // id2 should between id1 + 1 and graphs.GraphData.size() - 1
        INDEX id2 = dist(gen);
        if (id1 != id2) {
            std::pair<INDEX, INDEX> pair = std::minmax(id1, id2);
            auto [fst, snd] = g_pairs.insert(pair);
            if (snd) {
                graph_pairs.emplace_back(pair);
            }
        }
    }
    // if there are not enough existing pairs (computation has been interrupted) and num_pairs is set, only generate that many pairs
    std::vector<std::pair<INDEX, INDEX>> next_graph_pairs;
    // iterate through the graph pairs and add those to next_graph_pairs that are not in existing_pairs
    // get the last index in graph_pairs that is bigger than an entry occuring in existing_pairs (to avoid unnecessary iterations)
    size_t max_index = 0;
    for (const auto& pair : existing_pairs) {
        auto it = ranges::find(graph_pairs, pair);
        if (it != graph_pairs.end()) {
            size_t index = std::distance(graph_pairs.begin(), it);
            if (index > max_index) {
                max_index = index;
            }
        }
    }
    // iterate over graph_pairs starting with max_index + 1
    for (size_t index = max_index + 1; index < graph_pairs.size(); ++index) {
        const auto& pair = graph_pairs[index];
        if (ranges::find(existing_pairs, pair) == existing_pairs.end()) {
            next_graph_pairs.emplace_back(pair);
        }
    }

    // std::cout number of pairs to compute
    std::cout << "Number of GED mappings to compute: " << num_pairs - existing_pairs.size() << std::endl;
    size_t number_of_pairs_to_compute = num_pairs - existing_pairs.size();
    if (number_of_pairs_to_compute == 0) {
        std::cout << "All requested GED mappings already exist. Exiting." << std::endl;
        return 0;
    }

    graph_pairs = next_graph_pairs;

    // Ensure base tmp directory exists
    std::filesystem::path base_tmp = output_path + db + "/tmp/";
    std::filesystem::create_directories(base_tmp);

    // If threads is one do not chunk

    auto ged_env = ged::GEDEnv<ged::LabelID, ged::LabelID, ged::LabelID>();
    InitializeGEDEnvironment(ged_env, graphs, edit_cost, ged_method, method_options);
    ComputeGEDResults(ged_env, graphs, graph_pairs, number_of_pairs_to_compute, base_tmp.string(), ged_method, method_options);


    std::string search_string = "_ged_mapping";
    MergeGEDResults(output_path + db + "/tmp/", output_path + db + "/", search_string, graphs);
    // load mappings
    results.clear();
    const std::string path = output_path + db + "/" + db + "_ged_mapping.bin";
    BinaryToGEDResult(path, graphs, results);
    // Fix invalid mappings that are still present (due to parallel execution issues in gedlib)
    fixInvalidMappings(results, graphs, edit_cost, ged_method, method_options);
    // save the updated results back to binary
    GEDResultToBinary(output_path + "/" + db + "/", results);
    CSVFromGEDResults(output_path + db + "/" + db + "_ged_mapping.csv", results);

    return 0;
}

#endif //GEDPATHS_CREATE_EDIT_MAPPINGS_H