//
// Created by florian on 10.10.25.
//


#include "src/analyze_edit_path_graphs.h"
#include <filesystem>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace {

using PathStrategySet = std::vector<std::string>;

std::vector<std::string> GetAllDatasetsForMethod(const std::string& edit_paths_root, const std::string& method) {
    const std::filesystem::path method_path = std::filesystem::path(edit_paths_root) / method;
    if (!std::filesystem::exists(method_path)) {
        throw std::runtime_error("Edit path directory does not exist: " + method_path.string());
    }
    if (!std::filesystem::is_directory(method_path)) {
        throw std::runtime_error("Edit path path is not a directory: " + method_path.string());
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

std::string ResolveEditPathsRoot(std::string edit_path_output, const PathStrategySet& path_strategies) {
    std::vector<EditPathStrategy> edit_path_strategies = StringsToEditPathStrategies(path_strategies);
    if (!GetValidStrategy(edit_path_strategies)) {
        throw std::runtime_error("Invalid edit path strategies specified.");
    }

    const std::string short_strategy = EditPathStrategiesToStringShort(edit_path_strategies);
    const size_t pos = edit_path_output.find("Paths/");
    if (pos != std::string::npos) {
        edit_path_output.replace(pos, 6, "Paths_" + short_strategy + "/");
    } else {
        edit_path_output += "Paths_" + short_strategy + "/";
    }

    return edit_path_output;
}

}

int main(int argc, const char * argv[]) {
    // -db argument for the database
    std::string db = "MUTAG";
    // -edit_paths base argument for the path to store the edit paths
    std::string edit_path_output = "../Results/Paths/";
    // path generation strategy
    std::vector<std::string> path_strategies = {"Random"};
    std::string method = "F2";
    bool low_memory = false;
    bool use_all_db = false;
    bool use_all_path_strategies = false;
    bool db_was_set = false;
    bool path_strategy_was_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-db" || arg == "-data" || arg == "-dataset" || arg == "-database") {
            if (use_all_db) {
                std::cerr << "Error: -db cannot be combined with -all_db." << std::endl;
                return 1;
            }
            if (i + 1 >= argc) {
                std::cout << "Error: -db requires an argument" << std::endl;
                return 1;
            }
            db = argv[i+1];
            db_was_set = true;
            ++i;
        }
        else if (arg == "-all_db") {
            if (db_was_set) {
                std::cerr << "Error: -all_db cannot be combined with -db." << std::endl;
                return 1;
            }
            use_all_db = true;
        }
        else if (arg == "-edit_paths") {
            if (i + 1 >= argc) {
                std::cout << "Error: -edit_paths requires an argument" << std::endl;
                return 1;
            }
            edit_path_output = argv[i+1];
            ++i;
        }
        else if (arg == "-method") {
            if (i + 1 >= argc) {
                std::cout << "Error: -method requires an argument" << std::endl;
                return 1;
            }
            method = argv[i+1];
            ++i;
        }
        else if (arg == "-path_strategy") {
            if (use_all_path_strategies) {
                std::cerr << "Error: -path_strategy cannot be combined with -all_path_strategies." << std::endl;
                return 1;
            }
            path_strategy_was_set = true;
            path_strategies.clear();
            ++i;
            while (i < argc && std::string(argv[i]).rfind('-', 0) != 0) {
                path_strategies.push_back(argv[i]);
                ++i;
            }
            --i;
            if (path_strategies.empty()) {
                path_strategies.push_back("Random");
            }
        }
        else if (arg == "-all_path_strategies") {
            if (path_strategy_was_set) {
                std::cerr << "Error: -all_path_strategies cannot be combined with -path_strategy." << std::endl;
                return 1;
            }
            use_all_path_strategies = true;
        }
        else if (arg == "-low_memory") {
            low_memory = true;
        }
        // add help
        else if (arg == "-help") {
            std::cout << "Analyze edit path statistics" << std::endl;
            std::cout << "Arguments:" << std::endl;
            std::cout << "-db | -data | -dataset | -database <database name>" << std::endl;
            std::cout << "-all_db <use all datasets found in <edit_paths>_<path_strategy>/<method>/>" << std::endl;
            std::cout << "-edit_paths <edit paths root>" << std::endl;
            std::cout << "-method <GED method name>" << std::endl;
            std::cout << "-path_strategy <list of path strategy tokens>" << std::endl;
            std::cout << "-all_path_strategies <use Random, Random DeleteIsolatedNodes, InsertEdges DeleteIsolatedNodes, DeleteEdges DeleteIsolatedNodes>" << std::endl;
            std::cout << "-low_memory (skip BGF graph load and graph-level metrics)" << std::endl;
            std::cout << "-all_db analyzes every dataset directory under each selected <edit_paths>_<path_strategy>/<method>/ and reports any per-dataset failures." << std::endl;
            return 0;
        }
        else {
            std::cout << "Unknown argument: " << argv[i] << std::endl;
            return 1;
        }

    }

    std::cout << "Start analysis" << std::endl;

    std::vector<PathStrategySet> strategy_sets;
    if (use_all_path_strategies) {
        strategy_sets = GetCanonicalPathStrategies();
    } else {
        strategy_sets.push_back(path_strategies);
    }

    std::vector<std::tuple<std::string, std::string, int>> failures;
    size_t attempted_runs = 0;
    size_t successes = 0;

    for (const auto& current_path_strategies : strategy_sets) {
        std::string current_edit_path_output;
        std::string strategy_name;
        try {
            current_edit_path_output = ResolveEditPathsRoot(edit_path_output, current_path_strategies);
            strategy_name = EditPathStrategiesToStringShort(StringsToEditPathStrategies(current_path_strategies));
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }

        std::vector<std::string> databases = {db};
        if (use_all_db) {
            try {
                databases = GetAllDatasetsForMethod(current_edit_path_output, method);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                return 1;
            }
        }

        if (databases.empty()) {
            std::cerr << "No dataset directories found for strategy '" << strategy_name
                      << "' under " << current_edit_path_output << method << "\n";
            return 1;
        }

        for (const auto& db_name : databases) {
            ++attempted_runs;
            std::cout << "\n== AnalyzePaths: " << method << " / " << strategy_name << " / " << db_name << " ==\n";
            int rc = analyze_edit_path_graphs(db_name, current_edit_path_output, method, low_memory);
            if (rc == 0) {
                ++successes;
            } else {
                failures.emplace_back(strategy_name, db_name, rc);
            }
        }
    }

    std::cout << "\nAnalyzePaths summary for method " << method << ":\n";
    std::cout << "  Successful runs: " << successes << "/" << attempted_runs << "\n";
    if (!failures.empty()) {
        std::cout << "  Failed runs:\n";
        for (const auto& failure : failures) {
            std::cout << "    " << std::get<0>(failure) << " / " << std::get<1>(failure)
                      << " (exit code " << std::get<2>(failure) << ")\n";
        }
    }

    return failures.empty() ? 0 : 1;
}
