// filepath: /home/florian/Documents/CodeProjectsGit/GNNGED/src/convert.h

// Add a lightweight header providing a converter from the "matchings.txt"
// format (used under Results/Mappings/Precomputed/<DB>/matchings.txt) into the
// repository's binary mapping format written by GEDResultToBinary.
//
// The matchings format (observed in Precomputed/LINUX/matchings.txt) is a
// sequence of blocks. Each block starts with a header line:
//   <source> <target> <num_nodes> <dist> <opt>
// followed by <num_nodes> lines, each "<src_idx> <tgt_idx>" giving the node
// mapping for each source node in order. A target index of -1 denotes deletion.
//
// Usage (callable from a small tool or other code):
//   ConvertPrecomputedMatchingsToBin(precomputed_dir, db_name, out_root);
// This will read "<precomputed_dir>/matchings.txt" and write the binary
// results into out_root/<db_name>/<db_name>_ged_mapping.bin via
// GEDResultToBinary(out_root + "/" + db_name + "/", results).

#ifndef GEDPATHS_CONVERT_H
#define GEDPATHS_CONVERT_H

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <limits>

#include <libGraph.h>


inline bool ParseHeaderLine(const std::string &line, int &source, int &target, int &num_nodes, double &d1, double &d2) {
    std::istringstream iss(line);
    if (!(iss >> source >> target >> num_nodes)) return false;
    d1 = 0.0; d2 = 0.0;
    // try to read up to two doubles
    if (iss >> d1) {
        if (!(iss >> d2)) d2 = d1;
    } else {
        d1 = d2 = 0.0;
    }
    return true;
}

// Convert matchings.txt located in `precomputed_dir` into our binary format.
// precomputed_dir is expected to contain `matchings.txt` (and optionally
// `graphs.txt`). db is the dataset name (used to form the output folder),
// out_root is the root path where the method-specific mappings are stored
// (e.g., "Results/Mappings/").
inline int ConvertPrecomputedMatchingsToBin(const std::string &matching_dir,
                                            const std::string &graph_dir,
                                           const std::string &db) {
    GraphData<UDataGraph> graphs;
    graphs.Load(graph_dir + "/" + db + ".bgf");
    namespace fs = std::filesystem;
    auto dir = fs::path(matching_dir);
    if (!fs::exists(matching_dir) || !fs::is_directory(matching_dir)) {
        std::cerr << "matchings.txt not found in: " << matching_dir << "\n";
        std::cerr << "(Tried locating: " << matching_dir << ")\n";
        return 1;
    }
    fs::path matchings = dir / "matchings.txt";
    if (!fs::exists(matchings)) {
        std::cerr << "matchings.txt not found in: " << matching_dir << "\n";
        return 1;
    }

    std::ifstream ifs(matchings);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open matchings file: " << matchings << "\n";
        return 2;
    }

    std::vector<GEDEvaluation<UDataGraph>> results;
    std::string line;
    long line_no = 0;
    // get total lines for progress reporting
    size_t total_lines = 0;
    {
        std::ifstream ifs_count(matchings);
        while (std::getline(ifs_count, line)) {
            ++total_lines;
        }
    }
    // reset ifs
    ifs.clear();
    ifs.seekg(0, std::ios::beg);

    while (std::getline(ifs, line)) {
        // print progress every 1000 lines
        if (line_no % 1000 == 0) {
            std::cout << "Processing line " << line_no << " / " << total_lines << "\r" << std::endl;
        }
        ++line_no;
        // skip empty or comment lines
        if (line.empty()) continue;
        // trim leading spaces
        size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos == std::string::npos) continue;
        if (line[pos] == '#') continue;

        int source = -1, target = -1, num_nodes = 0;
        double lower_bound = 0.0, upper_bound = 0.0;
        if (!ParseHeaderLine(line.substr(pos), source, target, num_nodes, lower_bound, upper_bound)) {
            std::cerr << "Warning: failed to parse header at line " << line_no << ": '" << line << "'\n";
            continue;
        }
        double distance = lower_bound; // in this format, dist == lower_bound
        std::pair<int,int> graph_ids = { source, target };
        std::pair<std::vector<INDEX>, std::vector<INDEX>> matching;
        matching.first = std::vector<INDEX>(graphs[source].nodes(), static_cast<INDEX>(-1));
        matching.second = std::vector<INDEX>(graphs[target].nodes(), static_cast<INDEX>(-1));
        std::vector<INDEX> node_matching(graphs[source].nodes(), static_cast<INDEX>(-1));

        int read_count = 0;
        while (read_count < num_nodes && std::getline(ifs, line)) {
            ++line_no;
            // skip empty lines
            if (line.empty()) continue;
            std::istringstream iss(line);
            int a = -1, b = -1;
            if (!(iss >> a >> b)) {
                // try single integer (some files may only list target indices per source position)
                std::istringstream iss2(line);
                if (!(iss2 >> b)) {
                    std::cerr << "Warning: failed to parse mapping line at " << line_no << ": '" << line << "'\n";
                    continue;
                } else {
                    // assume source index equals read_count
                    a = read_count;
                }
            }
            node_matching[read_count] = static_cast<INDEX>(b);
            ++read_count;
        }

        INDEX idx = 0;
        for (auto val : node_matching) {
            matching.first[idx] = val;
            ++idx;
        }
        // build reverse mapping
        for (INDEX i = 0; i < matching.first.size(); ++i) {
            INDEX tgt = matching.first[i];
            if (tgt >= 0 && static_cast<size_t>(tgt) < matching.second.size()) {
                matching.second[tgt] = static_cast<INDEX>(i);
            }
        }

        if (read_count != num_nodes) {
            std::cerr << "Warning: expected " << num_nodes << " mapping lines for pair (" << source << "," << target << ") but read " << read_count << "\n";
            // continue anyway with what we have
        }


        // Create GEDEvaluation object and populate fields we know about
        GEDEvaluation<UDataGraph> ev;
        ev.graph_ids = { static_cast<INDEX>(source), static_cast<INDEX>(target) };
        ev.node_mapping = matching;
        ev.graph_data_name = db;
        ev.distance = lower_bound;
        ev.lower_bound = lower_bound;
        ev.upper_bound = upper_bound;

        // push to results (only if source != target)
        if (source != target) {
            results.push_back(std::move(ev));
        }
    }

    // Use repository's writer
    GEDResultToBinary(matching_dir + + "/", results);
    // Write the _ged_mapping.csv as well
    std::ofstream csv_ofs(matching_dir + "/" + db + "_ged_mapping.csv");
    if (!csv_ofs.is_open()) {
        std::cerr << "Failed to open ged_mapping.csv for writing in: " << matching_dir << "\n";
        return 3;
    }
    csv_ofs << "source_id,target_id,lower_bound,upper_bound,distance,time\n";
    for (const auto &res : results) {
        csv_ofs << res.graph_ids.first << "," << res.graph_ids.second << ","
                << res.lower_bound << "," << res.upper_bound << ","
                << res.distance << "," << 0 << "\n";
    }
    // Also add the graph_ids.txt
    std::ofstream gids_ofs(matching_dir + "/" + "graph_ids.txt");
    if (!gids_ofs.is_open()) {
        std::cerr << "Failed to open graph_ids.txt for writing in: " << matching_dir << "\n";
        return 3;
    }
    for (const auto &res : results) {
        gids_ofs << res.graph_ids.first << " " << res.graph_ids.second << "\n";
    }
    gids_ofs.close();
    std::cout << "Wrote " << results.size() << " mappings to binary at: "
              << matching_dir + "/" + db + "_ged_mapping.bin" << "\n";
    return 0;
}

#endif //GEDPATHS_CONVERT_H

