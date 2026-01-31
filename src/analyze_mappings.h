//
// Created by florian on 29.10.25.
//

#ifndef GEDPATHS_ANALYZE_MAPPINGS_H
#define GEDPATHS_ANALYZE_MAPPINGS_H

#include <iostream>
#include <filesystem>
#include <vector>
#include <map>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <libGraph.h>

// helper for pair hash
struct PairHash {
    std::size_t operator()(const std::pair<INDEX, INDEX>& p) const noexcept {
        return std::hash<long long>()((static_cast<long long>(p.first) << 32) ^ static_cast<long long>(p.second));
    }
};

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    size_t n = 0;
};

inline Stats compute_stats(const std::vector<double>& vals) {
    Stats s;
    if (vals.empty()) return s;
    s.n = vals.size();
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
        if (v < s.min) s.min = v;
        if (v > s.max) s.max = v;
    }
    s.mean = sum / static_cast<double>(s.n);
    double ss = 0.0;
    for (double v : vals) ss += (v - s.mean) * (v - s.mean);
    s.stddev = std::sqrt(ss / static_cast<double>(s.n));
    return s;
}

inline void print_stats(const std::string& name, const Stats& s) {
    std::cout << "Statistics for " << name << ":\n";
    std::cout << "  Count: " << s.n << "\n";
    if (s.n == 0) return;
    std::cout << "  Mean: " << s.mean << "\n";
    std::cout << "  Stddev: " << s.stddev << "\n";
    std::cout << "  Min: " << s.min << "\n";
    std::cout << "  Max: " << s.max << "\n";
}

inline int analyze_mappings(const std::string& db,
                             const std::string& processed_graph_path,
                             const std::string& mappings_root,
                             const std::string& method,
                             const std::string& compare_method = "",
                             const std::string& csv_out = "") {
        // prepare paths
    std::string mappings_path_a = mappings_root;
    if (mappings_path_a.back() != '/') mappings_path_a += '/';
    mappings_path_a += method + "/" + db + "/" + db + "_ged_mapping.bin";

    std::string mappings_path_b;
    if (!compare_method.empty()) {
        mappings_path_b = mappings_root;
        if (mappings_path_b.back() != '/') mappings_path_b += '/';
        mappings_path_b += compare_method + "/" + db + "/" + db + "_ged_mapping.bin";
    }

    // Load graphs (function returns void in this codebase; mimic usage in other tools)
    GraphData<UDataGraph> graphs;
    LoadSaveGraphDatasets::LoadPreprocessedGraphData(db, processed_graph_path, graphs);
    if (graphs.graphData.empty()) {
        std::cerr << "No graphs loaded for db='" << db << "' from '" << processed_graph_path << "'\n";
        return 1;
    }

    // Load mappings A
    std::vector<GEDEvaluation<UDataGraph>> results_a;
    if (!std::filesystem::exists(mappings_path_a)) {
        std::cerr << "Mappings file not found: " << mappings_path_a << "\n";
        return 2;
    }
    BinaryToGEDResult(mappings_path_a, graphs, results_a);
    std::cout << "Loaded " << results_a.size() << " mappings from " << mappings_path_a << "\n";

    // Count the invalid mappings
    auto invalids = CheckResultsValidity(results_a);
    if (!invalids.empty()) {
        std::cerr << "Warning: Found invalid mappings for the following result ids (these will be skipped):\n";
        for (const auto &id : invalids) {
            std::cerr << "  " << id << ": " << "Graph IDs (" << results_a[id].graph_ids.first << ", " << results_a[id].graph_ids.second << ")\n";
        }
    } else {
        std::cout << "All loaded mappings are valid.\n";
    }

    // Build map from pair->distance
    std::unordered_map<std::pair<INDEX, INDEX>, double, PairHash> map_a;
    for (const auto& r : results_a) {
        map_a[{r.graph_ids.first, r.graph_ids.second}] = r.distance;
    }

    Stats stats_a = compute_stats([&]() {
        std::vector<double> v; v.reserve(map_a.size());
        for (const auto &kv : map_a) v.push_back(kv.second);
        return v;
    }());
    print_stats(method + " (" + db + ")", stats_a);

    // If compare_method provided, load and compare
    if (!compare_method.empty()) {
        if (!std::filesystem::exists(mappings_path_b)) {
            std::cerr << "Mappings file for compare-method not found: " << mappings_path_b << "\n";
            return 3;
        }
        std::vector<GEDEvaluation<UDataGraph>> results_b;
        BinaryToGEDResult(mappings_path_b, graphs, results_b);
        std::cout << "Loaded " << results_b.size() << " mappings from " << mappings_path_b << "\n";
        std::unordered_map<std::pair<INDEX, INDEX>, double, PairHash> map_b;
        for (const auto& r : results_b) {
            map_b[{r.graph_ids.first, r.graph_ids.second}] = r.distance;
        }

        // gather pairs present in both
        std::vector<double> paired_a;
        std::vector<double> paired_b;
        paired_a.reserve(std::min(map_a.size(), map_b.size()));
        paired_b.reserve(std::min(map_a.size(), map_b.size()));
        for (const auto &kv : map_a) {
            auto key = kv.first;
            auto it = map_b.find(key);
            if (it != map_b.end()) {
                paired_a.push_back(kv.second);
                paired_b.push_back(it->second);
            }
        }
        std::cout << "Found " << paired_a.size() << " common graph pairs between methods.\n";
        if (paired_a.empty()) {
            std::cerr << "No overlapping pairs to compare.\n";
            return 4;
        }

        // compute diff statistics
        std::vector<double> diffs; diffs.reserve(paired_a.size());
        for (size_t i = 0; i < paired_a.size(); ++i) diffs.push_back(paired_a[i] - paired_b[i]);
        Stats stats_b = compute_stats(paired_b);
        Stats stats_diff = compute_stats(diffs);
        print_stats(compare_method + " (" + db + ")", stats_b);
        print_stats(std::string("Difference (") + method + " - " + compare_method + ")", stats_diff);

        // optional CSV output
        if (!csv_out.empty()) {
            std::ofstream ofs(csv_out);
            if (!ofs.is_open()) {
                std::cerr << "Failed to open CSV output: " << csv_out << "\n";
            } else {
                ofs << "id1,id2," << method << "," << compare_method << ",diff\n";
                for (const auto &kv : map_a) {
                    auto key = kv.first;
                    auto it = map_b.find(key);
                    if (it != map_b.end()) {
                        ofs << key.first << "," << key.second << "," << kv.second << "," << it->second << "," << (kv.second - it->second) << "\n";
                    }
                }
                ofs.close();
                std::cout << "Wrote comparison CSV to " << csv_out << "\n";
            }
        }
    }
    return 0;
}

#endif //GEDPATHS_ANALYZE_MAPPINGS_H