//
// Created by florian on 31.10.25.
//

#ifndef GEDPATHS_CREATE_EDIT_PATHS_H
#define GEDPATHS_CREATE_EDIT_PATHS_H

#include <libGraph.h>

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
        std::vector<EditPathStrategy> edit_path_strategies = StringsToEditPathStrategies(path_strategies);
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
    // Check validity and collect invalid result ids
    auto invalids = CheckResultsValidity(results);
    if (!invalids.empty()) {
        std::cerr << "Warning: Found invalid mappings for the following result ids (these will be skipped):\n";
        for (const auto &id : invalids) {
            std::cerr << "  " << id << ": " << "Graph IDs (" << results[id].graph_ids.first << ", " << results[id].graph_ids.second << ")\n";
        }
    }
    else {
        std::cout << "All loaded mappings are valid.\n";
    }


    // Filter out invalid results
    std::vector<GEDEvaluation<UDataGraph>> valid_results;
    for (size_t i = 0; i < results.size(); ++i) {
        if (std::find(invalids.begin(), invalids.end(), static_cast<int>(i)) == invalids.end()) {
            valid_results.push_back(results[i]);
        }
    }
    if (valid_results.empty()) {
        std::cerr << "No valid results to process. Exiting.\n";
        return 1;
    }
    std::cout << "Proceeding with " << valid_results.size() << " valid mappings out of " << results.size() << " total mappings.\n";

    if (num_mappings > 0 && num_mappings < static_cast<int>(valid_results.size())) {
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
    // Falls source_id und target_id gesetzt sind, nur das entsprechende Mapping verwenden
    if (source_id >= 0 && target_id >= 0) {
        std::cout << "Creating edit path for specific graph IDs: " << source_id << " and " << target_id << ".\n";
        auto it = std::find_if(results.begin(), results.end(), [&](const GEDEvaluation<UDataGraph>& eval) {
            return (eval.graph_ids.first == source_id && eval.graph_ids.second == target_id) ||
                   (eval.graph_ids.first == target_id && eval.graph_ids.second == source_id);
        });
        if (it == results.end()) {
            std::cerr << "Kein Mapping für die angegebenen Graphen-IDs gefunden.\n";
            return 1;
        }
        std::vector<GEDEvaluation<UDataGraph>> single_result{*it};
        std::cout << "Erzeuge Edit-Path nur für Mapping zwischen Graph " << source_id << " und " << target_id << ".\n";
        CreateAllEditPaths(single_result, graphs,  edit_path_output_db, seed, connected_only, edit_path_strategies);
        return 0;
    }
    // print info about number of valid results considered
    std::cout << "Creating edit paths for " << valid_results.size() << " valid mappings out of " << results.size() << " total mappings.\n";
    CreateAllEditPaths(valid_results, graphs,  edit_path_output_db, seed, connected_only, edit_path_strategies);

    return 0;
}

#endif //GEDPATHS_CREATE_EDIT_PATHS_H