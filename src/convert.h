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

inline std::filesystem::path LocatePrecomputedDir(const std::string &suggested) {
    namespace fs = std::filesystem;
    fs::path p(suggested);
    if (fs::exists(p) && fs::is_directory(p)) return p;
    // Try searching upward from cwd and exe directory for the relative path
    std::vector<fs::path> start_points;
    start_points.push_back(fs::current_path());
    try {
        fs::path exe = fs::read_symlink("/proc/self/exe");
        start_points.push_back(exe.parent_path());
    } catch (...) {}

    for (const auto &sp : start_points) {
        fs::path cur = sp;
        for (int i = 0; i < 8; ++i) {
            fs::path cand = cur / suggested;
            if (fs::exists(cand) && fs::is_directory(cand)) return cand;
            if (!cur.has_parent_path()) break;
            cur = cur.parent_path();
        }
    }
    // fallback: return original path (may be relative and not exist)
    return p;
}

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
inline int ConvertPrecomputedMatchingsToBin(const std::string &precomputed_dir,
                                           const std::string &db,
                                           const std::string &out_root) {
    namespace fs = std::filesystem;
    fs::path dir = LocatePrecomputedDir(precomputed_dir);
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "matchings.txt not found in: " << precomputed_dir << "\n";
        std::cerr << "(Tried locating: " << dir << ")\n";
        return 1;
    }
    fs::path matchings = dir / "matchings.txt";
    if (!fs::exists(matchings)) {
        std::cerr << "matchings.txt not found in: " << precomputed_dir << "\n";
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
    // If present, try to open and parse graphs.txt for possible sanity checks (optional)
    fs::path graphs_txt = dir / "graphs.txt";
    if (fs::exists(graphs_txt)) {
        // we won't parse it fully here, just note it's available
        std::cerr << "Info: found graphs.txt at " << graphs_txt << " (not required)\n";
    }

    while (std::getline(ifs, line)) {
        ++line_no;
        // skip empty or comment lines
        if (line.empty()) continue;
        // trim leading spaces
        size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos == std::string::npos) continue;
        if (line[pos] == '#') continue;

        int source = -1, target = -1, num_nodes = 0;
        double d1 = 0.0, d2 = 0.0;
        if (!ParseHeaderLine(line.substr(pos), source, target, num_nodes, d1, d2)) {
            std::cerr << "Warning: failed to parse header at line " << line_no << ": '" << line << "'\n";
            continue;
        }

        // read num_nodes mapping lines
        std::vector<INDEX> src_to_tgt;
        src_to_tgt.reserve(static_cast<size_t>(std::max(0, num_nodes)));
        int read_count = 0;
        while (read_count < num_nodes && std::getline(ifs, line)) {
            ++line_no;
            // skip empty lines
            if (line.empty()) continue;
            std::istringstream iss(line);
            int a = 0, b = -1;
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
            src_to_tgt.push_back(static_cast<INDEX>(b));
            ++read_count;
        }

        if (read_count != num_nodes) {
            std::cerr << "Warning: expected " << num_nodes << " mapping lines for pair (" << source << "," << target << ") but read " << read_count << "\n";
            // continue anyway with what we have
        }

        // Build inverse mapping (target -> source)
        int max_tgt = -1;
        for (auto v : src_to_tgt) if (v >= 0 && v > max_tgt) max_tgt = static_cast<int>(v);
        std::vector<INDEX> tgt_to_src;
        if (max_tgt >= 0) {
            tgt_to_src.assign(static_cast<size_t>(max_tgt + 1), static_cast<INDEX>(-1));
            for (size_t i = 0; i < src_to_tgt.size(); ++i) {
                INDEX t = src_to_tgt[i];
                if (t >= 0) {
                    if (static_cast<size_t>(t) >= tgt_to_src.size()) {
                        tgt_to_src.resize(static_cast<size_t>(t) + 1, static_cast<INDEX>(-1));
                    }
                    tgt_to_src[static_cast<size_t>(t)] = static_cast<INDEX>(i);
                }
            }
        }

        // Create GEDEvaluation object and populate fields we know about
        GEDEvaluation<UDataGraph> ev;
        ev.graph_ids = { static_cast<INDEX>(source), static_cast<INDEX>(target) };
        ev.distance = d1;
        ev.lower_bound = d1;
        ev.upper_bound = d2;
        ev.node_mapping = { src_to_tgt, tgt_to_src };

        // push to results
        results.push_back(std::move(ev));
    }

    // Prepare output directory and call existing writer
    fs::path out_dir_path = out_root;
    if (!out_dir_path.empty() && out_dir_path.filename() != "/") {
        // ensure trailing slash semantics like other code: pass out_root + "/" + db + "/"
    }
    std::string output_base = out_root;
    if (!output_base.empty() && output_base.back() != '/') output_base += '/';
    output_base += db + '/';

    // Ensure directory exists
    try {
        std::filesystem::create_directories(output_base);
    } catch (...) {}

    // Use repository's writer
    GEDResultToBinary(output_base, results);
    std::cout << "Wrote " << results.size() << " mappings to binary at: " << output_base << "\n";
    return 0;
}

#endif //GEDPATHS_CONVERT_H

