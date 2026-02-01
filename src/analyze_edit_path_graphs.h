//
// Created by florian on 29.10.25.
//

#ifndef GEDPATHS_ANALYZE_EDIT_PATH_GRAPHS_H
#define GEDPATHS_ANALYZE_EDIT_PATH_GRAPHS_H

#include <numeric>
#include <cmath>
#include <limits>
#include <tuple>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <libGraph.h>

// helper for pair hash (used with std::unordered_map)
struct PairHash {
    std::size_t operator()(const std::pair<INDEX, INDEX>& p) const noexcept {
        return std::hash<long long>()((static_cast<long long>(p.first) << 32) ^ static_cast<long long>(p.second));
    }
};

struct BucketOperations {
    unsigned long _node_insertions = 0;
    unsigned long _node_deletions = 0;
    unsigned long _node_relabels = 0;
    unsigned long _edge_insertions = 0;
    unsigned long _edge_deletions = 0;
    unsigned long _edge_relabels = 0;
};

// Statistic class for num, average, stddev, min, max of a list of values with name
class ValueStatistics {
public:
    ValueStatistics()= default;
    explicit ValueStatistics(const std::string& name, const std::vector<double>& values);
    void PrintStatistics() const;
    // Write the stored values to a CSV file in the provided directory.
    // The filename is derived from the statistic name (spaces replaced with underscores).
    void WriteCSV(const std::string& output_dir) const;
private:
    std::string _name;
    std::vector<double> _values;
    unsigned long _num_values = 0;
    double _sum_values = 0.0;
    double _average = 0.0;
    double _stddev = 0.0;
    double _min = std::numeric_limits<double>::max();
    double _max = std::numeric_limits<double>::min();
};

ValueStatistics::ValueStatistics(const std::string &name, const std::vector<double> &values) {
    _name = name;
    _values = values;
    _num_values = values.size();
    _sum_values = std::accumulate(values.begin(), values.end(), 0.0);
    if (_num_values > 0) {
        _average = _sum_values / static_cast<double>(_num_values);
        double sum_squared_diff = 0.0;
        for (const auto& value : values) {
            sum_squared_diff += (value - _average) * (value - _average);
            if (value < _min) {
                _min = value;
            }
            if (value > _max) {
                _max = value;
            }
        }
        _stddev = std::sqrt(sum_squared_diff / static_cast<double>(_num_values));
    } else {
        _min = 0.0;
        _max = 0.0;
    }
}

void ValueStatistics::PrintStatistics() const {
    std::cout << "Statistics for " << _name << ":\n";
    std::cout << "  Number of values: " << _num_values << "\n";
    std::cout << "  Average: " << _average << "\n";
    std::cout << "  Standard Deviation: " << _stddev << "\n";
    std::cout << "  Minimum: " << _min << "\n";
    std::cout << "  Maximum: " << _max << "\n";
}

void ValueStatistics::WriteCSV(const std::string &output_dir) const {
    namespace fs = std::filesystem;
    try {
        fs::create_directories(output_dir);
    } catch (...) {
        // If directory creation fails, still attempt to write file (may fail later).
    }
    // sanitize name to filename
    std::string filename = _name;
    for (auto &c : filename) {
        if (c == ' ' || c == '/') c = '_';
    }
    std::ostringstream path;
    path << output_dir;
    if (!output_dir.empty() && output_dir.back() != '/') path << '/';
    path << filename << ".csv";

    std::ofstream ofs(path.str());
    if (!ofs.is_open()) {
        std::cerr << "Failed to write CSV file: " << path.str() << std::endl;
        return;
    }
    ofs << "value\n";
    for (const auto &v : _values) {
        // write with high precision
        ofs << std::setprecision(10) << v << "\n";
    }
    ofs.close();
}



// Statistic class for edit paths
class EditPathStatistics {
public:
    // Default constructor deleted - references must be initialized
    EditPathStatistics() = delete;
    explicit EditPathStatistics(const GraphData<UDataGraph>& edit_paths, const std::vector<std::tuple<INDEX, INDEX, INDEX, EditOperation>>& edit_path_info);
    void PrintStatistics() const;
    // Write all contained ValueStatistics to CSV files inside the provided directory.
    void WriteCSVFiles(const std::string& output_dir) const;
    void WritePositionCSVFiles(const std::string& output_dir) const;
private:
    const GraphData<UDataGraph>& _edit_paths;  // const ref to avoid copy
    const std::vector<std::tuple<INDEX, INDEX, INDEX, EditOperation>>& _edit_path_info;  // const ref to avoid copy
    ValueStatistics _num_nodes_stats;
    ValueStatistics _num_edges_stats;
    ValueStatistics _num_operations_stats;
    ValueStatistics _path_length_stats;
    ValueStatistics _node_insertions_stats;
    ValueStatistics _node_deletions_stats;
    ValueStatistics _node_relabels_stats;
    ValueStatistics _edge_insertions_stats;
    ValueStatistics _edge_deletions_stats;
    ValueStatistics _edge_relabels_stats;
    ValueStatistics _connectedness_stats;
    // Per-path position lists (each inner vector corresponds to one edit path and stores positions/indexes where that operation occurred)
    std::vector<std::vector<int>> _node_insertion_positions;
    std::vector<std::vector<int>> _node_deletion_positions;
    std::vector<std::vector<int>> _node_relabel_positions;
    std::vector<std::vector<int>> _edge_insertion_positions;
    std::vector<std::vector<int>> _edge_deletion_positions;
    std::vector<std::vector<int>> _edge_relabel_positions;
};

EditPathStatistics::EditPathStatistics(const GraphData<UDataGraph> &edit_paths, const std::vector<std::tuple<INDEX, INDEX, INDEX, EditOperation>> &edit_path_info)
    : _edit_paths(edit_paths), _edit_path_info(edit_path_info) {
    std::vector<double> num_nodes;
    std::vector<double> num_edges;
    std::vector<double> num_operations;
    std::vector<double> path_lengths;
    std::vector<double> node_insertions;
    std::vector<double> node_deletions;
    std::vector<double> node_relabels;
    std::vector<double> edge_insertions;
    std::vector<double> edge_deletions;
    std::vector<double> edge_relabels;
    std::vector<double> graphs_unconnected;
    const unsigned long bucket_size = 10;
    std::vector<BucketOperations> buckets ( bucket_size );

    // Map to count operations per (source_id, target_id)
    std::unordered_map<std::pair<INDEX, INDEX>, std::vector<EditOperation>, PairHash> operations_map;
    std::unordered_map<std::pair<INDEX, INDEX>, std::pair<INDEX, INDEX>, PairHash> graph_positions_map;
    INDEX position = -1;
    for (const auto& entry : _edit_path_info) {
        INDEX source_id = std::get<0>(entry);
        INDEX step_id = std::get<1>(entry);
        INDEX target_id = std::get<2>(entry);
        if (step_id == 0) {
            ++position;
            auto source_graph = &_edit_paths.graphData[position];
            std::string source_graph_name = source_graph->GetName();
            // print name
            std::cout << "Processing edit paths for source graph: " << source_graph_name << std::endl;
            graph_positions_map[{source_id, target_id}] = {position, position + 1};
        }
        else {
            graph_positions_map[{source_id, std::get<2>(entry)}].second += 1;
        }
        ++position;

        EditOperation operation = std::get<3>(entry);
        operations_map[{source_id, target_id}].push_back(operation);
    }

    // Now calculate statistics based on operations_map
    for (const auto& [key, operations] : operations_map) {
        INDEX source_id = key.first;
        INDEX target_id = key.second;

        // Find all graphs corresponding to the current path order in operations_map is the same as in edit_paths
        std::vector<UDataGraph*> path_graphs = std::vector<UDataGraph*>(operations.size() + 1, nullptr);
        for (INDEX i = 0; i < path_graphs.size(); ++i) {
            INDEX graph_index = graph_positions_map[{source_id, target_id}].first;
            path_graphs[i] = &_edit_paths.graphData[graph_index + i];
        }

        if (!path_graphs.empty()) {
            graphs_unconnected.push_back(0.0);
            for (const auto& g : path_graphs) {
                if (g == nullptr) continue;
                num_nodes.push_back(static_cast<double>(g->nodes()));
                num_edges.push_back(static_cast<double>(g->edges()));
                // print whether the graph is connected
                std::string graph_name = g->GetName();
                bool connected = g->GetConnectivity();
                if (!connected) {
                    graphs_unconnected.back() += 1.0;
                }
            }
            node_insertions.push_back(0.0);
            node_deletions.push_back(0.0);
            node_relabels.push_back(0.0);
            edge_insertions.push_back(0.0);
            edge_deletions.push_back(0.0);
            edge_relabels.push_back(0.0);

            unsigned long bucket_counter = 0;
            unsigned long operation_counter = 0;
            // per-path positional lists for this path
            std::vector<int> node_insert_pos;
            std::vector<int> node_delete_pos;
            std::vector<int> node_relabel_pos;
            std::vector<int> edge_insert_pos;
            std::vector<int> edge_delete_pos;
            std::vector<int> edge_relabel_pos;
            for (const auto& op : operations) {
                // make divisor explicit as double to avoid narrowing warnings
                auto ops_size_d = static_cast<double>(operations.size());
                double bucket_divisor = ops_size_d / static_cast<double>(bucket_size);
                bucket_counter = std::min(static_cast<unsigned long>(std::floor(static_cast<double>(operation_counter) / bucket_divisor)), bucket_size - 1);
                switch (op.operationObject) {
                    case OperationObject::NODE:
                        if (op.type == EditType::INSERT) {
                            node_insertions.back() += 1.0;
                            node_insert_pos.push_back(static_cast<int>(operation_counter));
                            buckets[bucket_counter]._node_insertions += 1;
                        } else if (op.type == EditType::DELETE) {
                            node_deletions.back() += 1.0;
                            node_delete_pos.push_back(static_cast<int>(operation_counter));
                            buckets[bucket_counter]._node_deletions += 1;
                        } else if (op.type == EditType::RELABEL) {
                            node_relabels.back() += 1.0;
                            node_relabel_pos.push_back(static_cast<int>(operation_counter));
                            buckets[bucket_counter]._node_relabels += 1;
                        }
                        break;
                    case OperationObject::EDGE:
                        if (op.type == EditType::INSERT) {
                            edge_insertions.back() += 1.0;
                            edge_insert_pos.push_back(static_cast<int>(operation_counter));
                            buckets[bucket_counter]._edge_insertions += 1;
                        } else if (op.type == EditType::DELETE) {
                            edge_deletions.back() += 1.0;
                            edge_delete_pos.push_back(static_cast<int>(operation_counter));
                            buckets[bucket_counter]._edge_deletions += 1;
                        } else if (op.type == EditType::RELABEL) {
                            edge_relabels.back() += 1.0;
                            edge_relabel_pos.push_back(static_cast<int>(operation_counter));
                            buckets[bucket_counter]._edge_relabels += 1;
                        }
                        break;
                    default:
                        break;
                }
                operation_counter += 1;
            }
            // store per-path positions into the global vectors
            _node_insertion_positions.push_back(std::move(node_insert_pos));
            _node_deletion_positions.push_back(std::move(node_delete_pos));
            _node_relabel_positions.push_back(std::move(node_relabel_pos));
            _edge_insertion_positions.push_back(std::move(edge_insert_pos));
            _edge_deletion_positions.push_back(std::move(edge_delete_pos));
            _edge_relabel_positions.push_back(std::move(edge_relabel_pos));
            num_operations.push_back(static_cast<double>(operations.size()));
            path_lengths.push_back(static_cast<double>(operations.size()));
        }
    }

    _num_nodes_stats = ValueStatistics("Number of Nodes", num_nodes);
    _num_edges_stats = ValueStatistics("Number of Edges", num_edges);
    _num_operations_stats = ValueStatistics("Number of Operations", num_operations);
    _path_length_stats = ValueStatistics("Path Length", path_lengths);
    _node_insertions_stats = ValueStatistics("Node Insertions", node_insertions);
    _node_deletions_stats = ValueStatistics("Node Deletions", node_deletions);
    _node_relabels_stats = ValueStatistics("Node Relabels", node_relabels);
    _edge_insertions_stats = ValueStatistics("Edge Insertions", edge_insertions);
    _edge_deletions_stats = ValueStatistics("Edge Deletions", edge_deletions);
    _edge_relabels_stats = ValueStatistics("Edge Relabels", edge_relabels);
    _connectedness_stats = ValueStatistics("Graphs Unconnected", graphs_unconnected);


}
void EditPathStatistics::PrintStatistics() const {
    std::cout << "Edit Path Statistics:\n";
    _num_nodes_stats.PrintStatistics();
    _num_edges_stats.PrintStatistics();
    _num_operations_stats.PrintStatistics();
    _path_length_stats.PrintStatistics();
    _node_insertions_stats.PrintStatistics();
    _node_deletions_stats.PrintStatistics();
    _node_relabels_stats.PrintStatistics();
    _edge_insertions_stats.PrintStatistics();
    _edge_deletions_stats.PrintStatistics();
    _edge_relabels_stats.PrintStatistics();
    _connectedness_stats.PrintStatistics();
}

void EditPathStatistics::WriteCSVFiles(const std::string &output_dir) const {
    // Create directory and write each ValueStatistics as its own CSV
    _num_nodes_stats.WriteCSV(output_dir);
    _num_edges_stats.WriteCSV(output_dir);
    _num_operations_stats.WriteCSV(output_dir);
    _path_length_stats.WriteCSV(output_dir);
    _node_insertions_stats.WriteCSV(output_dir);
    _node_deletions_stats.WriteCSV(output_dir);
    _node_relabels_stats.WriteCSV(output_dir);
    _edge_insertions_stats.WriteCSV(output_dir);
    _edge_deletions_stats.WriteCSV(output_dir);
    _edge_relabels_stats.WriteCSV(output_dir);
    _connectedness_stats.WriteCSV(output_dir);
}

// Write per-path positions CSV files: each row corresponds to a path and contains comma-separated positions (or empty if none)
void WritePositionsCSVFile(const std::string &output_dir, const std::string &name, const std::vector<std::vector<int>> &positions_vec) {
    namespace fs = std::filesystem;
    try { fs::create_directories(output_dir); } catch(...) {}
    std::string filename = name;
    for (auto &c : filename) if (c == ' ' || c == '/') c = '_';
    std::ostringstream path;
    path << output_dir; if (!output_dir.empty() && output_dir.back() != '/') path << '/';
    path << filename << ".csv";

    std::ofstream ofs(path.str());
    if (!ofs.is_open()) { std::cerr << "Failed to write positions CSV: " << path.str() << std::endl; return; }
    ofs << "positions\n";
    for (const auto &row : positions_vec) {
        if (row.empty()) { ofs << "\n"; continue; }
        bool first = true;
        for (int v : row) {
            if (!first) ofs << ',';
            ofs << v;
            first = false;
        }
        ofs << "\n";
    }
    ofs.close();
}

void EditPathStatistics::WritePositionCSVFiles(const std::string &output_dir) const {
    WritePositionsCSVFile(output_dir, "Node_Insertions_Positions", _node_insertion_positions);
    WritePositionsCSVFile(output_dir, "Node_Deletions_Positions", _node_deletion_positions);
    WritePositionsCSVFile(output_dir, "Node_Relabels_Positions", _node_relabel_positions);
    WritePositionsCSVFile(output_dir, "Edge_Insertions_Positions", _edge_insertion_positions);
    WritePositionsCSVFile(output_dir, "Edge_Deletions_Positions", _edge_deletion_positions);
    WritePositionsCSVFile(output_dir, "Edge_Relabels_Positions", _edge_relabel_positions);
}


inline int analyze_edit_path_graphs(const std::string& db,
                                    const std::string& edit_path_output,
                                    const std::string& method) {
    std::string edit_path_output_db = edit_path_output + method + "/" + db + "/";

    // Load MUTAG edit paths
    GraphData<UDataGraph> edit_paths;
    std::vector<std::tuple<INDEX, INDEX, INDEX, EditOperation>> edit_path_info;
    edit_paths.Load(edit_path_output_db + db + "_edit_paths.bgf");
    std::string info_path = edit_path_output_db + db + "_edit_paths_data.bin";
    ReadEditPathInfo(info_path, edit_path_info);
    EditPathStatistics stats(edit_paths, edit_path_info);
    stats.PrintStatistics();

    // Write evaluation CSVs under Results/Paths/<method>/<db>/Evaluation/ create directory if it does not exist
    std::string eval_dir = edit_path_output_db + "Evaluation/";
    if (!std::filesystem::exists(eval_dir)) {
        std::filesystem::create_directory(eval_dir);
    }
    stats.WriteCSVFiles(eval_dir);
    // Write per-path positions evaluation CSVs
    stats.WritePositionCSVFiles(eval_dir);
    return 0;
}

#endif //GEDPATHS_ANALYZE_EDIT_PATH_GRAPHS_H