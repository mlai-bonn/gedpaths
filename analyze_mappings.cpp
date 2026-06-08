// New file: analyze_mappings.cpp
// Load mappings for a given method and dataset and compare distances


#include "src/analyze_mappings.h"

int main(int argc, const char* argv[]) {
    // defaults
    std::string db = "MUTAG";
    std::string processed_graph_path = "../Data/ProcessedGraphs/";
    std::string mappings_root = "../Results/Mappings/";
    std::string method = "F2";
    std::string compare_method; // optional second method to compare against
    std::string csv_out; // optional CSV of pairwise comparisons
    bool analyze_all = false;

    // parse simple argv-style (consistent with repo tools)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-db" || arg == "-data" || arg == "-dataset" || arg == "-database") {
            if (i + 1 >= argc) {
                std::cout << "Error: " << arg << " requires an argument" << std::endl;
                return 1;
            }
            db = argv[i+1];
            ++i;
        } else if (arg == "-processed") {
            if (i + 1 >= argc) {
                std::cout << "Error: -processed requires an argument" << std::endl;
                return 1;
            }
            processed_graph_path = argv[i+1];
            ++i;
        } else if (arg == "-mappings") {
            if (i + 1 >= argc) {
                std::cout << "Error: -mappings requires an argument" << std::endl;
                return 1;
            }
            mappings_root = argv[i+1];
            ++i;
        } else if (arg == "-method") {
            if (i + 1 >= argc) {
                std::cout << "Error: -method requires an argument" << std::endl;
                return 1;
            }
            method = argv[i+1];
            ++i;
        } else if (arg == "-compare-method") {
            if (i + 1 >= argc) {
                std::cout << "Error: -compare-method requires an argument" << std::endl;
                return 1;
            }
            compare_method = argv[i+1];
            ++i;
        } else if (arg == "-csv-out") {
            if (i + 1 >= argc) {
                std::cout << "Error: -csv-out requires an argument" << std::endl;
                return 1;
            }
            csv_out = argv[i+1];
            ++i;
        } else if (arg == "-all") {
            analyze_all = true;
        } else if (arg == "-help") {
            std::cout << "analyze_mappings: load GED mappings and compare distances\n";
            std::cout << "Usage: " << argv[0] << " [-db NAME] [-all] [-method METHOD] [-compare-method OTHER_METHOD] [-mappings PATH] [-processed PATH] [-csv-out FILE]\n";
            std::cout << "  -all analyzes every dataset directory under <mappings>/<method>/ and reports any per-dataset failures.\n";
            return 0;
        }
    }

    if (analyze_all) {
        std::filesystem::path method_root = mappings_root;
        method_root /= method;

        if (!std::filesystem::exists(method_root) || !std::filesystem::is_directory(method_root)) {
            std::cerr << "Mappings method directory not found: " << method_root << "\n";
            return 1;
        }

        std::vector<std::string> dbs;
        for (const auto& entry : std::filesystem::directory_iterator(method_root)) {
            if (entry.is_directory()) {
                dbs.push_back(entry.path().filename().string());
            }
        }
        std::sort(dbs.begin(), dbs.end());

        if (dbs.empty()) {
            std::cerr << "No dataset directories found under " << method_root << "\n";
            return 1;
        }

        std::vector<std::pair<std::string, int>> failures;
        size_t successes = 0;
        for (const auto& db_name : dbs) {
            std::cout << "\n== AnalyzeMappings: " << method << " / " << db_name << " ==\n";
            int rc = analyze_mappings(db_name, processed_graph_path, mappings_root, method, compare_method, csv_out);
            if (rc == 0) {
                ++successes;
            } else {
                failures.emplace_back(db_name, rc);
            }
        }

        std::cout << "\nAnalyzeMappings summary for method " << method << ":\n";
        std::cout << "  Successful datasets: " << successes << "/" << dbs.size() << "\n";
        if (!failures.empty()) {
            std::cout << "  Failed datasets:\n";
            for (const auto& failure : failures) {
                std::cout << "    " << failure.first << " (exit code " << failure.second << ")\n";
            }
        }

        return failures.empty() ? 0 : 1;
    }

    return analyze_mappings(db, processed_graph_path, mappings_root, method, compare_method, csv_out);
}
