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
#include <unordered_set>
#include <stdexcept>
#include <libGraph.h>
#include <src/env/ged_env.hpp>

// helper for pair hash (enables O(1) lookups for pair-based containers)
struct PairHash {
    std::size_t operator()(const std::pair<INDEX, INDEX>& p) const noexcept {
        return std::hash<long long>()((static_cast<long long>(p.first) << 32) ^ static_cast<long long>(p.second));
    }
};

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
        throw std::out_of_range("Single source/target IDs out of range: " + std::to_string(source_id) + ", " + std::to_string(target_id));
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
    fixed_results.reserve(invalid_mappings.size());
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
        // Build hash set for O(1) lookup of existing graph IDs
        std::unordered_set<std::pair<INDEX, INDEX>, PairHash> existing_ids_set;
        existing_ids_set.reserve(results.size());
        existing_graph_ids.reserve(results.size());
        for (const auto& res : results) {
            existing_graph_ids.emplace_back(res.graph_ids);
            existing_ids_set.insert(res.graph_ids);
        }
        // add possible existing graph ids from the tmp folder
        // Merge all mappings in tmp folder
        std::vector<GEDEvaluation<UDataGraph>> tmp_results;
        if (std::filesystem::exists(tmp_path) && !std::filesystem::is_empty(tmp_path)) {
            MergeBinaries(tmp_path, db + "_", graphs, tmp_results);
            // append new tmp_results
            for (const auto& res : tmp_results) {
                if (existing_ids_set.find(res.graph_ids) == existing_ids_set.end()) {
                    results.emplace_back(res);
                    existing_graph_ids.emplace_back(res.graph_ids);
                    existing_ids_set.insert(res.graph_ids);
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

    if (num_pairs == -1) {
        num_pairs = graphs.graphData.size() * (graphs.graphData.size() - 1) / 2;
    }


    size_t max_number_of_pairs = 1000000;
    std::vector<std::pair<INDEX, INDEX>> random_pair_order;
    random_pair_order.reserve(max_number_of_pairs);
    std::vector<std::pair<INDEX, INDEX>> pairs_to_compute_candidates;
    pairs_to_compute_candidates.reserve(num_pairs);
    std::vector<std::pair<INDEX, INDEX>> existing_pairs;
    std::vector<std::pair<INDEX, INDEX>> valid_computed_pairs;
    std::vector<std::pair<INDEX, INDEX>> invalid_computed_pairs;

    auto results = std::vector<GEDEvaluation<UDataGraph>>{};
    // Load existing mappings if they exist and add their graph ids to existing_pairs
    get_existing_mappings(output_path, db, graphs, existing_pairs, results);





    // Fix invalid mappings that are still present (due to parallel execution issues in gedlib)
    fixInvalidMappings(results, graphs, edit_cost, ged_method, method_options);
    // save the updated results back to binary
    GEDResultToBinary(output_path + "/" + db + "/", results);

    // If single_source and single_target are set, only compute and print that mapping
    if (single_source >= 0 && single_target >= 0) {
        auto result = create_edit_mappings_single(single_source, single_target, graphs, edit_cost, ged_method, method_options, true);
        return 0;
    }

    // If existing pairs already contain all possible pairs, we can skip the computation
    if (existing_pairs.size() == graphs.graphData.size() * (graphs.graphData.size() - 1) / 2) {
        std::cout << "All possible GED mappings already exist. Exiting." << std::endl;
        return 0;
    }
    // If num_pairs == -1, graph pairs should contain all possible pairs of graph ids that are not in existing_pairs
    if (num_pairs == graphs.graphData.size() * (graphs.graphData.size() - 1) / 2) {
        for (INDEX i = 0; i < graphs.graphData.size(); ++i) {
            for (INDEX j = i + 1; j < graphs.graphData.size(); ++j) {
                std::pair<INDEX, INDEX> pair = {i, j};
                auto find_id = ranges::find(existing_pairs, pair);
                if (find_id == existing_pairs.end()) {
                    pairs_to_compute_candidates.emplace_back(pair);
                }
            }
        }
    }
    else {
        std::set<std::pair<INDEX, INDEX>> random_pairs;
        // If num_pairs is set, randomly sample pairs, always the same ones for the same seed
        // set up random device
        auto gen = std::mt19937(seed);
        std::uniform_int_distribution<INDEX> dist(0, graphs.graphData.size() - 2);
        while (random_pairs.size() < max_number_of_pairs && random_pairs.size() != graphs.graphData.size() * (graphs.graphData.size() - 1) / 2) {
            // get random integer between 0 and graphs.GraphData.size() - 1
            INDEX id1 = dist(gen);
            // id2 should between id1 + 1 and graphs.GraphData.size() - 1
            std::uniform_int_distribution<INDEX> dist2((INDEX)id1 + 1, graphs.graphData.size() - 1);
            INDEX id2 = dist2(gen);
            // with prob 0.5 add the pair (id1, id2) or (id2, id1) to random_pairs
            if (std::bernoulli_distribution(0.5)(gen)) {
                std::swap(id1, id2);
            }
            random_pairs.insert({id1, id2});
        }
        // create a fixed random order based on the seed to choose the graph_pairs
        for (auto& pair : random_pairs) {
            random_pair_order.emplace_back(pair);
        }
        ranges::shuffle(random_pair_order, gen);

        // add only those pairs to pairs_to_compute that are not in existing_pairs
        for (const auto& pair : random_pair_order) {
            auto find_id = ranges::find(existing_pairs, pair);
            if (find_id == existing_pairs.end()) {
                pairs_to_compute_candidates.emplace_back(pair);
            }
            else {
                // check if the existing pair is valid or not using the index in existing_pairs to find the corresponding result in results and check its time
                size_t index = std::distance(existing_pairs.begin(), find_id);
                if (results[index].time == -1) {
                    invalid_computed_pairs.emplace_back(pair);
                }
                else {
                    valid_computed_pairs.emplace_back(pair);
                }
            }
        }
    }
    if (num_pairs < valid_computed_pairs.size()) {
        std::cout << "The number of pairs to compute is smaller than the number of already computed valid pairs. Exiting." << std::endl;
        return 0;
    }
    size_t num_pairs_to_compute = num_pairs - valid_computed_pairs.size();
    // std::cout number of pairs to compute
    std::cout << "Number of GED mappings to compute: " << num_pairs_to_compute << std::endl;
    if (num_pairs_to_compute == 0) {
        std::cout << "All requested GED mappings already exist. Exiting." << std::endl;
        return 0;
    }

    // Ensure base tmp directory exists
    std::filesystem::path base_tmp = output_path + db + "/tmp/";
    std::filesystem::create_directories(base_tmp);

    // Chunk the pairs into batches of size 10 to allow for better parallelization
    INDEX chunk_size = 10;
    std::vector<std::vector<std::pair<INDEX, INDEX>>> chunks;
    for (INDEX i = 0; i < pairs_to_compute_candidates.size(); i += chunk_size) {
        std::vector<std::pair<INDEX, INDEX>> chunk(pairs_to_compute_candidates.begin() + i, pairs_to_compute_candidates.begin() + std::min(i + chunk_size, pairs_to_compute_candidates.size()));
        chunks.push_back(chunk);
    }

    INDEX total_computed_pairs = 0;
    bool should_stop = false;
    const INDEX num_chunks = chunks.size();

    // Compute GED results for each chunk in parallel
    #pragma omp parallel num_threads(num_threads) shared(graphs, edit_cost, ged_method, method_options, base_tmp, total_computed_pairs, num_pairs_to_compute, chunks, should_stop, num_chunks) default(none)
    {
        // Each thread creates its own GED environment ONCE
        auto ged_env = ged::GEDEnv<ged::LabelID, ged::LabelID, ged::LabelID>();
        InitializeGEDEnvironment(ged_env, graphs, edit_cost, ged_method, method_options);

        // Use omp for to distribute chunks across threads
        #pragma omp for schedule(dynamic)
        for (INDEX chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            // Early exit check: skip remaining work if we have enough pairs
            if (should_stop) continue;

            const auto& chunk = chunks[chunk_idx];
            size_t computed_pairs = ComputeGEDResults(ged_env, graphs, chunk, base_tmp.string(), ged_method, method_options);

            #pragma omp atomic
            total_computed_pairs += computed_pairs;

            // Check if we should stop
            if (total_computed_pairs >= num_pairs_to_compute) {
                should_stop = true;
            }
        }
    }

    std::string search_string = "_ged_mapping";
    MergeGEDResults(output_path + db + "/tmp/", output_path + db + "/", search_string, graphs);
    // load mappings
    results.clear();
    const std::string path = output_path + db + "/" + db + "_ged_mapping.bin";
    BinaryToGEDResult(path, graphs, results);
    // Fix invalid mappings that are still present (due to parallel execution issues in gedlib)
    fixInvalidMappings(results, graphs, edit_cost, ged_method, method_options);
    std::vector<GEDEvaluation<UDataGraph>> final_results;
    // use only the first valid num_pairs from results (according to the random order)
    // create a map from result pair ids to result index for O(1) lookup
    std::unordered_map<std::pair<INDEX, INDEX>, INDEX, PairHash> result_pair_to_index;
    for (auto i = 0; i < results.size(); ++i) {
        result_pair_to_index[results[i].graph_ids] = i;
    }
    int final_counter = 0;
    while (final_results.size() < num_pairs && final_counter < static_cast<int>(random_pair_order.size())) {
        const auto& pair = random_pair_order[final_counter];
        auto it = result_pair_to_index.find(pair);
        if (it != result_pair_to_index.end()) {
            final_results.emplace_back(results[it->second]);
        }
        ++final_counter;
    }

    if (final_results.size() < static_cast<size_t>(num_pairs)) {
        std::cout << "Warning: only " << final_results.size() << " GED mappings found for requested "
                  << num_pairs << " pairs." << std::endl;
    }

    // save the updated results back to binary
    GEDResultToBinary(output_path + "/" + db + "/", final_results);
    CSVFromGEDResults(output_path + db + "/" + db + "_ged_mapping.csv", final_results);

    return 0;
}

#endif //GEDPATHS_CREATE_EDIT_MAPPINGS_H

