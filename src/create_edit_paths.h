//
// Created by florian on 31.10.25.
//

#ifndef GEDPATHS_CREATE_EDIT_PATHS_H
#define GEDPATHS_CREATE_EDIT_PATHS_H

#include <libGraph.h>
#include <unordered_set>

inline int create_edit_paths( const std::string& db,
                              const std::string& processed_graph_path,
                              std::string& mappings_path,
                              std::string& edit_path_output,
                              const std::string& method = "REFINE",
                              const int num_mappings = -1,
                              const int seed = 42,
                              const bool connected_only = false,
                              const std::vector<std::string>& path_strategies = {"Random"},
                              const int source_id = -1,
                              const int target_id = -1) {
    std::vector<EditPathStrategy> edit_path_strategies;
    try {
        edit_path_strategies = StringsToEditPathStrategies(path_strategies);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    if (!GetValidStrategy(edit_path_strategies)) {
        std::cerr << "Error: Invalid edit path strategies specified." << std::endl;
        return 1;
    }

    edit_path_output += "Paths_" + EditPathStrategiesToStringShort(edit_path_strategies) + "/";
    if (!std::filesystem::exists(edit_path_output)) {
        std::filesystem::create_directory(edit_path_output);
    }

    // add method and db to the output path
    mappings_path = mappings_path  + method + "/" + db + "/";
    const std::string edit_path_output_db = edit_path_output + method + "/" + db + "/";
    if (!std::filesystem::exists(edit_path_output)) {
        std::filesystem::create_directory(edit_path_output);
    }
    if (!std::filesystem::exists(edit_path_output_db)) {
        std::filesystem::create_directories(edit_path_output_db);
    }



    GraphData<UDataGraph> graphs;
    LoadSaveGraphDatasets::LoadPreprocessedGraphData(db, processed_graph_path, graphs);


    // load mappings
    std::vector<GEDEvaluation<UDataGraph>> results;
    BinaryToGEDResult(mappings_path + db + "_ged_mapping.bin", graphs, results);

    // get only the valid results
    std::vector<GEDEvaluation<UDataGraph>> valid_results;
    std::vector<int> valids;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].valid) {
            valids.push_back(static_cast<int>(i));
            valid_results.push_back(std::move(results[i]));
        }
    }
    // print number of invalid and valid results
    std::cout << "Found " << results.size() - valids.size() << " invalid mappings and " << valids.size() << " valid mappings." << std::endl;
    // print percentage of invalid
    std::cout << "Percentage of invalid mappings: " << (results.size() - valids.size()) * 100.0 / results.size() << "%\n";

    // print the number of unique



    // Falls source_id und target_id gesetzt sind, nur das entsprechende Mapping verwenden
    if (source_id >= 0 && target_id >= 0) {
        std::cout << "Creating edit path for specific graph IDs: " << source_id << " and " << target_id << ".\n";
        int found_index = -1;
        for (size_t i = 0; i < valids.size(); ++i) {
            const auto& eval = results[i];
            if ((eval.graph_ids.first == source_id && eval.graph_ids.second == target_id) ||
                (eval.graph_ids.first == target_id && eval.graph_ids.second == source_id)) {
                found_index = static_cast<int>(i);
                break;
            }
        }
        if (found_index < 0) {
            std::cerr << "Error: No valid mapping found for graph IDs " << source_id << " and " << target_id << ". Exiting.\n";
            return 1;
        }
        std::vector<GEDEvaluation<UDataGraph>> single_result;
        single_result.reserve(1);
        single_result.push_back(std::move(results[found_index]));
        results.clear();
        results.shrink_to_fit();
        std::cout << "Creating edit paths for graph IDs " << source_id << " and " << target_id << ".\n";
        CreateAllEditPaths(single_result, graphs,  edit_path_output_db, seed, connected_only, edit_path_strategies);
        return 0;
    }

    if (valids.empty()) {
        std::cerr << "No valid results to process. Exiting.\n";
        return 1;
    }

    std::cout << "Creating edit paths for " << valid_results.size() << " valid mappings out of " << results.size() << " total mappings.\n";

    if (num_mappings > 0 && num_mappings < static_cast<int>(valids.size())) {
        // shuffle valid_results and take first num_mappings
        std::ranges::shuffle(valid_results, std::mt19937(seed));
        valid_results.resize(num_mappings);
        // sort valid_results by graph ids
        std::ranges::sort(valid_results, [](const GEDEvaluation<UDataGraph>& a, const GEDEvaluation<UDataGraph>& b) {
            if (a.graph_ids.first != b.graph_ids.first) {
                return a.graph_ids.first < b.graph_ids.first;
            }
            return a.graph_ids.second < b.graph_ids.second;
        });

    }
    // Print the number of unique endpoint graphs in valid_results and the percentage regarding the whole dataset
    std::unordered_set<INDEX> unique_graph_ids;
    for (const auto& result : valid_results) {
        unique_graph_ids.insert(result.graph_ids.first);
        unique_graph_ids.insert(result.graph_ids.second);
    }
    std::cout << "Number of unique endpoint graphs in valid results: " << unique_graph_ids.size() << " out of " << graphs.graphData.size() << " total graphs (" << unique_graph_ids.size() * 100.0 / graphs.graphData.size() << "%)\n";

    // Check whether the paths have been computed
    if (std::filesystem::exists(edit_path_output_db + db + "_edit_paths.bgf")) {
        std::cout << "Edit paths for " << db << " already exist at " << edit_path_output_db + db + "_edit_paths.bgf" << std::endl;
        // Mention that one have to check if the paths are all that one wanted to compute
        std::cout << "Check if the paths are those you want to use!" << std::endl;
        return 0;
    }
    // print info about number of valid results considered
    CreateAllEditPaths(valid_results, graphs,  edit_path_output_db, seed, connected_only, edit_path_strategies);

    return 0;
}

#endif //GEDPATHS_CREATE_EDIT_PATHS_H
