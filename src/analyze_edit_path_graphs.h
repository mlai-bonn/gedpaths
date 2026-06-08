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
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <optional>
#include <libGraph.h>

inline std::string SanitizeMetricName(const std::string& name) {
    std::string filename = name;
    for (auto& c : filename) {
        if (c == ' ' || c == '/') c = '_';
    }
    return filename;
}

class ValueStatistics {
public:
    ValueStatistics()= default;
    explicit ValueStatistics(std::string name): _name(std::move(name)) {}
    void AddValue(double value);
    void PrintStatistics() const;
    [[nodiscard]] unsigned long GetCount() const { return _num_values; }
    [[nodiscard]] double GetAverage() const { return _average; }
    [[nodiscard]] double GetStddev() const { return _stddev; }
    [[nodiscard]] double GetMin() const { return _min; }
    [[nodiscard]] double GetMax() const { return _max; }
private:
    std::string _name;
    unsigned long _num_values = 0;
    double _average = 0.0;
    double _m2 = 0.0;
    double _stddev = 0.0;
    double _min = 0.0;
    double _max = 0.0;
};

void ValueStatistics::AddValue(const double value) {
    if (_num_values == 0) {
        _min = value;
        _max = value;
        _average = value;
        _m2 = 0.0;
        _num_values = 1;
        _stddev = 0.0;
        return;
    }

    _min = std::min(_min, value);
    _max = std::max(_max, value);
    _num_values += 1;
    const double delta = value - _average;
    _average += delta / static_cast<double>(_num_values);
    const double delta2 = value - _average;
    _m2 += delta * delta2;
    _stddev = std::sqrt(_m2 / static_cast<double>(_num_values));
}

inline void EnsureDirectory(const std::string& output_dir) {
    namespace fs = std::filesystem;
    try {
        fs::create_directories(output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to create directory '" << output_dir << "': " << e.what() << std::endl;
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

inline void WriteCSVRow(std::ofstream& ofs, const double value) {
    ofs << std::setprecision(10) << value << "\n";
}

inline bool OpenCSVWithHeader(const std::string& output_dir,
                              const std::string& metric_name,
                              std::ofstream& ofs) {
    std::ostringstream path;
    path << output_dir;
    if (!output_dir.empty() && output_dir.back() != '/') path << '/';
    path << SanitizeMetricName(metric_name) << ".csv";

    ofs.open(path.str());
    if (!ofs.is_open()) {
        std::cerr << "Failed to write CSV file: " << path.str() << std::endl;
        return false;
    }
    ofs << "value\n";
    return true;
}

inline bool OpenPositionsCSVWithHeader(const std::string& output_dir,
                                       const std::string& metric_name,
                                       std::ofstream& ofs) {
    std::ostringstream path;
    path << output_dir;
    if (!output_dir.empty() && output_dir.back() != '/') path << '/';
    path << SanitizeMetricName(metric_name) << ".csv";

    ofs.open(path.str());
    if (!ofs.is_open()) {
        std::cerr << "Failed to write positions CSV: " << path.str() << std::endl;
        return false;
    }
    ofs << "positions\n";
    return true;
}

inline void WritePositionsCSVRow(std::ofstream& ofs, const std::vector<int>& row) {
    if (row.empty()) {
        ofs << "\n";
        return;
    }
    bool first = true;
    for (const int value : row) {
        if (!first) ofs << ',';
        ofs << value;
        first = false;
    }
    ofs << "\n";
}

inline std::string JsonEscape(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

struct PathOperationCounts {
    unsigned long node_insertions = 0;
    unsigned long node_deletions = 0;
    unsigned long node_relabels = 0;
    unsigned long edge_insertions = 0;
    unsigned long edge_deletions = 0;
    unsigned long edge_relabels = 0;
    size_t num_operations = 0;
};

class EditPathStatistics {
public:
    EditPathStatistics() = delete;
    explicit EditPathStatistics(const GraphData<UDataGraph>* edit_paths,
                                const std::vector<std::tuple<INDEX, INDEX, INDEX, EditOperation>>& edit_path_info,
                                bool include_graph_metrics);
    void PrintStatistics() const;
    void WriteCSVFiles(const std::string& output_dir) const;
    void WritePositionCSVFiles(const std::string& output_dir) const;
    void WriteSummaryJson(const std::string& output_dir,
                          const std::string& db,
                          const std::string& method,
                          bool low_memory) const;
    [[nodiscard]] unsigned long GetNumGraphs() const { return _num_graphs; }
    [[nodiscard]] double GetAveragePathLength() const { return _path_length_stats.GetAverage(); }
    [[nodiscard]] double GetAverageNodeInsertions() const { return _node_insertions_stats.GetAverage(); }
    [[nodiscard]] double GetAverageNodeDeletions() const { return _node_deletions_stats.GetAverage(); }
    [[nodiscard]] double GetAverageNodeRelabels() const { return _node_relabels_stats.GetAverage(); }
    [[nodiscard]] double GetAverageEdgeInsertions() const { return _edge_insertions_stats.GetAverage(); }
    [[nodiscard]] double GetAverageEdgeDeletions() const { return _edge_deletions_stats.GetAverage(); }
    [[nodiscard]] double GetAverageEdgeRelabels() const { return _edge_relabels_stats.GetAverage(); }
    [[nodiscard]] bool HasGraphMetrics() const { return _include_graph_metrics; }
private:
    void BuildStatistics();
    void AddOperation(const EditOperation& op, PathOperationCounts& counts) const;
    void AddGraphPathMetrics(size_t graph_start_idx, size_t num_operations, ValueStatistics* num_nodes, ValueStatistics* num_edges, ValueStatistics* disconnected) const;

    const GraphData<UDataGraph>* _edit_paths;
    const std::vector<std::tuple<INDEX, INDEX, INDEX, EditOperation>>& _edit_path_info;
    bool _include_graph_metrics = false;
    unsigned long _num_graphs = 0;
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
};

EditPathStatistics::EditPathStatistics(const GraphData<UDataGraph>* edit_paths,
                                       const std::vector<std::tuple<INDEX, INDEX, INDEX, EditOperation>>& edit_path_info,
                                       const bool include_graph_metrics)
    : _edit_paths(edit_paths),
      _edit_path_info(edit_path_info),
      _include_graph_metrics(include_graph_metrics && edit_paths != nullptr),
      _num_nodes_stats("Number of Nodes"),
      _num_edges_stats("Number of Edges"),
      _num_operations_stats("Number of Operations"),
      _path_length_stats("Path Length"),
      _node_insertions_stats("Node Insertions"),
      _node_deletions_stats("Node Deletions"),
      _node_relabels_stats("Node Relabels"),
      _edge_insertions_stats("Edge Insertions"),
      _edge_deletions_stats("Edge Deletions"),
      _edge_relabels_stats("Edge Relabels"),
      _connectedness_stats("Graphs Unconnected") {
    BuildStatistics();
}

void EditPathStatistics::PrintStatistics() const {
    std::cout << "Edit Path Statistics:\n";
    _num_operations_stats.PrintStatistics();
    _path_length_stats.PrintStatistics();
    _node_insertions_stats.PrintStatistics();
    _node_deletions_stats.PrintStatistics();
    _node_relabels_stats.PrintStatistics();
    _edge_insertions_stats.PrintStatistics();
    _edge_deletions_stats.PrintStatistics();
    _edge_relabels_stats.PrintStatistics();
    if (_include_graph_metrics) {
        _num_nodes_stats.PrintStatistics();
        _num_edges_stats.PrintStatistics();
        _connectedness_stats.PrintStatistics();
    } else {
        std::cout << "Graph-level statistics skipped (low-memory mode enabled).\n";
    }
}

void EditPathStatistics::WriteCSVFiles(const std::string &output_dir) const {
    namespace fs = std::filesystem;
    EnsureDirectory(output_dir);

    std::ofstream num_nodes_csv;
    std::ofstream num_edges_csv;
    std::ofstream num_operations_csv;
    std::ofstream path_length_csv;
    std::ofstream node_insertions_csv;
    std::ofstream node_deletions_csv;
    std::ofstream node_relabels_csv;
    std::ofstream edge_insertions_csv;
    std::ofstream edge_deletions_csv;
    std::ofstream edge_relabels_csv;
    std::ofstream unconnected_csv;

    if (!OpenCSVWithHeader(output_dir, "Number of Operations", num_operations_csv)) return;
    if (!OpenCSVWithHeader(output_dir, "Path Length", path_length_csv)) return;
    if (!OpenCSVWithHeader(output_dir, "Node Insertions", node_insertions_csv)) return;
    if (!OpenCSVWithHeader(output_dir, "Node Deletions", node_deletions_csv)) return;
    if (!OpenCSVWithHeader(output_dir, "Node Relabels", node_relabels_csv)) return;
    if (!OpenCSVWithHeader(output_dir, "Edge Insertions", edge_insertions_csv)) return;
    if (!OpenCSVWithHeader(output_dir, "Edge Deletions", edge_deletions_csv)) return;
    if (!OpenCSVWithHeader(output_dir, "Edge Relabels", edge_relabels_csv)) return;
    if (_include_graph_metrics) {
        if (!OpenCSVWithHeader(output_dir, "Number of Nodes", num_nodes_csv)) return;
        if (!OpenCSVWithHeader(output_dir, "Number of Edges", num_edges_csv)) return;
        if (!OpenCSVWithHeader(output_dir, "Graphs Unconnected", unconnected_csv)) return;
    } else {
        for (const std::string& metric : {"Number of Nodes", "Number of Edges", "Graphs Unconnected"}) {
            std::ostringstream stale_path;
            stale_path << output_dir;
            if (!output_dir.empty() && output_dir.back() != '/') stale_path << '/';
            stale_path << SanitizeMetricName(metric) << ".csv";
            std::error_code ec;
            fs::remove(stale_path.str(), ec);
        }
    }

    bool has_path = false;
    size_t current_ops = 0;
    size_t graph_cursor = 0;
    PathOperationCounts counts;
    auto flush_path = [&]() {
        if (!has_path) return;
        WriteCSVRow(num_operations_csv, static_cast<double>(counts.num_operations));
        WriteCSVRow(path_length_csv, static_cast<double>(counts.num_operations));
        WriteCSVRow(node_insertions_csv, static_cast<double>(counts.node_insertions));
        WriteCSVRow(node_deletions_csv, static_cast<double>(counts.node_deletions));
        WriteCSVRow(node_relabels_csv, static_cast<double>(counts.node_relabels));
        WriteCSVRow(edge_insertions_csv, static_cast<double>(counts.edge_insertions));
        WriteCSVRow(edge_deletions_csv, static_cast<double>(counts.edge_deletions));
        WriteCSVRow(edge_relabels_csv, static_cast<double>(counts.edge_relabels));

        if (_include_graph_metrics) {
            double disconnected_graphs = 0.0;
            if (graph_cursor + current_ops < _edit_paths->graphData.size()) {
                for (size_t i = 0; i <= current_ops; ++i) {
                    const auto& graph = _edit_paths->graphData[graph_cursor + i];
                    WriteCSVRow(num_nodes_csv, static_cast<double>(graph.nodes()));
                    WriteCSVRow(num_edges_csv, static_cast<double>(graph.edges()));
                    const bool connected = const_cast<UDataGraph&>(graph).GetConnectivity();
                    if (!connected) disconnected_graphs += 1.0;
                }
            } else {
                std::cerr << "Warning: Graph index mismatch while writing CSV values." << std::endl;
            }
            WriteCSVRow(unconnected_csv, disconnected_graphs);
            graph_cursor += current_ops + 1;
        }
        counts = PathOperationCounts{};
        current_ops = 0;
    };

    for (const auto& entry : _edit_path_info) {
        const INDEX step_id = std::get<1>(entry);
        if (step_id == 0) {
            flush_path();
            has_path = true;
        }
        if (!has_path) has_path = true;
        AddOperation(std::get<3>(entry), counts);
        current_ops += 1;
    }
    flush_path();
}

void EditPathStatistics::WritePositionCSVFiles(const std::string &output_dir) const {
    EnsureDirectory(output_dir);

    std::ofstream node_insertions_pos_csv;
    std::ofstream node_deletions_pos_csv;
    std::ofstream node_relabels_pos_csv;
    std::ofstream edge_insertions_pos_csv;
    std::ofstream edge_deletions_pos_csv;
    std::ofstream edge_relabels_pos_csv;

    if (!OpenPositionsCSVWithHeader(output_dir, "Node Insertions Positions", node_insertions_pos_csv)) return;
    if (!OpenPositionsCSVWithHeader(output_dir, "Node Deletions Positions", node_deletions_pos_csv)) return;
    if (!OpenPositionsCSVWithHeader(output_dir, "Node Relabels Positions", node_relabels_pos_csv)) return;
    if (!OpenPositionsCSVWithHeader(output_dir, "Edge Insertions Positions", edge_insertions_pos_csv)) return;
    if (!OpenPositionsCSVWithHeader(output_dir, "Edge Deletions Positions", edge_deletions_pos_csv)) return;
    if (!OpenPositionsCSVWithHeader(output_dir, "Edge Relabels Positions", edge_relabels_pos_csv)) return;

    bool has_path = false;
    std::vector<int> node_insert_pos;
    std::vector<int> node_delete_pos;
    std::vector<int> node_relabel_pos;
    std::vector<int> edge_insert_pos;
    std::vector<int> edge_delete_pos;
    std::vector<int> edge_relabel_pos;
    int operation_counter = 0;

    auto flush_path = [&]() {
        if (!has_path) return;
        WritePositionsCSVRow(node_insertions_pos_csv, node_insert_pos);
        WritePositionsCSVRow(node_deletions_pos_csv, node_delete_pos);
        WritePositionsCSVRow(node_relabels_pos_csv, node_relabel_pos);
        WritePositionsCSVRow(edge_insertions_pos_csv, edge_insert_pos);
        WritePositionsCSVRow(edge_deletions_pos_csv, edge_delete_pos);
        WritePositionsCSVRow(edge_relabels_pos_csv, edge_relabel_pos);
        node_insert_pos.clear();
        node_delete_pos.clear();
        node_relabel_pos.clear();
        edge_insert_pos.clear();
        edge_delete_pos.clear();
        edge_relabel_pos.clear();
        operation_counter = 0;
    };

    for (const auto& entry : _edit_path_info) {
        const INDEX step_id = std::get<1>(entry);
        if (step_id == 0) {
            flush_path();
            has_path = true;
        }
        if (!has_path) has_path = true;

        const auto& op = std::get<3>(entry);
        switch (op.operationObject) {
            case OperationObject::NODE:
                if (op.type == EditType::INSERT) node_insert_pos.push_back(operation_counter);
                else if (op.type == EditType::DELETE) node_delete_pos.push_back(operation_counter);
                else if (op.type == EditType::RELABEL) node_relabel_pos.push_back(operation_counter);
                break;
            case OperationObject::EDGE:
                if (op.type == EditType::INSERT) edge_insert_pos.push_back(operation_counter);
                else if (op.type == EditType::DELETE) edge_delete_pos.push_back(operation_counter);
                else if (op.type == EditType::RELABEL) edge_relabel_pos.push_back(operation_counter);
                break;
            default:
                break;
        }
        operation_counter += 1;
    }
    flush_path();
}

void EditPathStatistics::WriteSummaryJson(const std::string& output_dir,
                                          const std::string& db,
                                          const std::string& method,
                                          const bool low_memory) const {
    EnsureDirectory(output_dir);
    std::ostringstream path;
    path << output_dir;
    if (!output_dir.empty() && output_dir.back() != '/') path << '/';
    path << "summary.json";

    std::ofstream ofs(path.str());
    if (!ofs.is_open()) {
        std::cerr << "Failed to write summary JSON: " << path.str() << std::endl;
        return;
    }

    const std::vector<std::pair<std::string, const ValueStatistics*>> base_metrics = {
            {"Number of Operations", &_num_operations_stats},
            {"Path Length", &_path_length_stats},
            {"Node Insertions", &_node_insertions_stats},
            {"Node Deletions", &_node_deletions_stats},
            {"Node Relabels", &_node_relabels_stats},
            {"Edge Insertions", &_edge_insertions_stats},
            {"Edge Deletions", &_edge_deletions_stats},
            {"Edge Relabels", &_edge_relabels_stats},
    };

    std::vector<std::pair<std::string, const ValueStatistics*>> metrics = base_metrics;
    if (_include_graph_metrics) {
        metrics.push_back({"Number of Nodes", &_num_nodes_stats});
        metrics.push_back({"Number of Edges", &_num_edges_stats});
        metrics.push_back({"Graphs Unconnected", &_connectedness_stats});
    }

    ofs << "{\n";
    ofs << "  \"db\": \"" << JsonEscape(db) << "\",\n";
    ofs << "  \"method\": \"" << JsonEscape(method) << "\",\n";
    ofs << "  \"low_memory\": " << (low_memory ? "true" : "false") << ",\n";
    ofs << "  \"include_graph_metrics\": " << (_include_graph_metrics ? "true" : "false") << ",\n";
    ofs << "  \"metrics\": {\n";
    for (size_t i = 0; i < metrics.size(); ++i) {
        const auto& [name, stats] = metrics[i];
        ofs << "    \"" << JsonEscape(name) << "\": {\n";
        ofs << "      \"count\": " << stats->GetCount() << ",\n";
        ofs << "      \"average\": " << std::setprecision(10) << stats->GetAverage() << ",\n";
        ofs << "      \"stddev\": " << std::setprecision(10) << stats->GetStddev() << ",\n";
        ofs << "      \"min\": " << std::setprecision(10) << stats->GetMin() << ",\n";
        ofs << "      \"max\": " << std::setprecision(10) << stats->GetMax() << "\n";
        ofs << "    }";
        if (i + 1 < metrics.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  }\n";
    ofs << "}\n";
}

void EditPathStatistics::AddOperation(const EditOperation& op, PathOperationCounts& counts) const {
    switch (op.operationObject) {
        case OperationObject::NODE:
            if (op.type == EditType::INSERT) counts.node_insertions += 1;
            else if (op.type == EditType::DELETE) counts.node_deletions += 1;
            else if (op.type == EditType::RELABEL) counts.node_relabels += 1;
            break;
        case OperationObject::EDGE:
            if (op.type == EditType::INSERT) counts.edge_insertions += 1;
            else if (op.type == EditType::DELETE) counts.edge_deletions += 1;
            else if (op.type == EditType::RELABEL) counts.edge_relabels += 1;
            break;
        default:
            break;
    }
    counts.num_operations += 1;
}

void EditPathStatistics::AddGraphPathMetrics(const size_t graph_start_idx,
                                             const size_t num_operations,
                                             ValueStatistics* num_nodes,
                                             ValueStatistics* num_edges,
                                             ValueStatistics* disconnected) const {
    if (!_include_graph_metrics || _edit_paths == nullptr) return;
    const size_t graph_count = num_operations + 1;
    if (graph_start_idx + graph_count > _edit_paths->graphData.size()) {
        std::cerr << "Warning: Path graph indices exceed loaded graph data." << std::endl;
        return;
    }

    double num_disconnected = 0.0;
    for (size_t i = 0; i < graph_count; ++i) {
        const auto& graph = _edit_paths->graphData[graph_start_idx + i];
        if (num_nodes != nullptr) num_nodes->AddValue(static_cast<double>(graph.nodes()));
        if (num_edges != nullptr) num_edges->AddValue(static_cast<double>(graph.edges()));
        const bool connected = const_cast<UDataGraph&>(graph).GetConnectivity();
        if (!connected) num_disconnected += 1.0;
    }
    if (disconnected != nullptr) disconnected->AddValue(num_disconnected);
}

void EditPathStatistics::BuildStatistics() {
    if (_edit_path_info.empty()) return;

    bool has_path = false;
    PathOperationCounts counts;
    size_t current_ops = 0;
    size_t graph_cursor = 0;

    auto flush_path = [&]() {
        if (!has_path) return;
        _num_operations_stats.AddValue(static_cast<double>(counts.num_operations));
        _path_length_stats.AddValue(static_cast<double>(counts.num_operations));
        _node_insertions_stats.AddValue(static_cast<double>(counts.node_insertions));
        _node_deletions_stats.AddValue(static_cast<double>(counts.node_deletions));
        _node_relabels_stats.AddValue(static_cast<double>(counts.node_relabels));
        _edge_insertions_stats.AddValue(static_cast<double>(counts.edge_insertions));
        _edge_deletions_stats.AddValue(static_cast<double>(counts.edge_deletions));
        _edge_relabels_stats.AddValue(static_cast<double>(counts.edge_relabels));
        _num_graphs += static_cast<unsigned long>(current_ops + 1);

        if (_include_graph_metrics) {
            AddGraphPathMetrics(graph_cursor, current_ops, &_num_nodes_stats, &_num_edges_stats, &_connectedness_stats);
            graph_cursor += current_ops + 1;
        }
        counts = PathOperationCounts{};
        current_ops = 0;
    };

    for (const auto& entry : _edit_path_info) {
        const INDEX step_id = std::get<1>(entry);
        if (step_id == 0) {
            flush_path();
            has_path = true;
        }
        if (!has_path) has_path = true;
        AddOperation(std::get<3>(entry), counts);
        current_ops += 1;
    }
    flush_path();
}

inline std::string FormatDouble2DP(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value;
    return oss.str();
}

inline std::string FormatIntWithLatexGrouping(unsigned long value) {
    std::string s = std::to_string(value);
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (i > 0 && ((s.size() - i) % 3 == 0)) out += "\\,";
        out += s[i];
    }
    return out;
}

inline bool ExtractResultsRoot(const std::string& edit_path_output,
                               std::filesystem::path& results_root) {
    namespace fs = std::filesystem;
    fs::path p(edit_path_output);
    p = p.lexically_normal();

    fs::path prefix;
    for (const auto& part : p) {
        const std::string token = part.string();
        if (token.rfind("Paths_", 0) == 0) {
            results_root = prefix;
            return true;
        }
        prefix /= part;
    }
    return false;
}

inline std::optional<double> ComputeAverageFromValueCSV(const std::filesystem::path& csv_path) {
    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) return std::nullopt;

    std::string line;
    if (!std::getline(ifs, line)) return std::nullopt; // header

    double sum = 0.0;
    unsigned long count = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        double v = 0.0;
        iss >> v;
        if (iss.fail()) continue;
        sum += v;
        ++count;
    }
    if (count == 0) return std::nullopt;
    return sum / static_cast<double>(count);
}

inline std::string StrategyLabel(const std::string& strategy) {
    if (strategy == "i-E_d-IsoN") return "$+E$";
    if (strategy == "d-E_d-IsoN") return "$-E$";
    if (strategy == "Rnd_d-IsoN") return "Rnd";
    if (strategy == "Rnd") return "Rnd (no iso cleanup)";
    return strategy;
}

struct Table2StrategyRow {
    std::string strategy_name;
    std::string order_label;
    double avg_nodes = 0.0;
    double avg_edges = 0.0;
    double avg_unconnected = 0.0;
};

inline int StrategySortKey(const std::string& strategy) {
    if (strategy == "Rnd_d-IsoN") return 0;
    if (strategy == "Rnd") return 1;
    if (strategy == "i-E_d-IsoN") return 2;
    if (strategy == "d-E_d-IsoN") return 3;
    return 100;
}

inline std::vector<Table2StrategyRow> CollectTable2Rows(const std::filesystem::path& results_root,
                                                        const std::string& method,
                                                        const std::string& db) {
    namespace fs = std::filesystem;
    std::vector<Table2StrategyRow> rows;
    if (!fs::exists(results_root) || !fs::is_directory(results_root)) return rows;

    for (const auto& entry : fs::directory_iterator(results_root)) {
        if (!entry.is_directory()) continue;
        const std::string dir_name = entry.path().filename().string();
        if (dir_name.rfind("Paths_", 0) != 0) continue;
        const std::string strategy = dir_name.substr(std::string("Paths_").size());

        fs::path eval_dir = entry.path() / method / db / "Evaluation";
        auto avg_nodes = ComputeAverageFromValueCSV(eval_dir / "Number_of_Nodes.csv");
        auto avg_edges = ComputeAverageFromValueCSV(eval_dir / "Number_of_Edges.csv");
        auto avg_unconnected = ComputeAverageFromValueCSV(eval_dir / "Graphs_Unconnected.csv");

        if (!avg_nodes || !avg_edges || !avg_unconnected) continue;

        Table2StrategyRow row;
        row.strategy_name = strategy;
        row.order_label = StrategyLabel(strategy);
        row.avg_nodes = *avg_nodes;
        row.avg_edges = *avg_edges;
        row.avg_unconnected = *avg_unconnected;
        rows.push_back(row);
    }

    std::ranges::sort(rows, [](const Table2StrategyRow& a, const Table2StrategyRow& b) {
        const int a_key = StrategySortKey(a.strategy_name);
        const int b_key = StrategySortKey(b.strategy_name);
        if (a_key != b_key) return a_key < b_key;
        return a.strategy_name < b.strategy_name;
    });
    return rows;
}

inline bool WriteTable1RowTex(const std::filesystem::path& output_path,
                              const std::string& db,
                              const EditPathStatistics& stats) {
    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to write LaTeX file: " << output_path << std::endl;
        return false;
    }

    ofs << db << " & $" << FormatIntWithLatexGrouping(stats.GetNumGraphs()) << "$ & $"
        << FormatDouble2DP(stats.GetAveragePathLength()) << "$ & $"
        << FormatDouble2DP(stats.GetAverageNodeInsertions()) << "$ & $"
        << FormatDouble2DP(stats.GetAverageNodeDeletions()) << "$ & $"
        << FormatDouble2DP(stats.GetAverageNodeRelabels()) << "$ & $"
        << FormatDouble2DP(stats.GetAverageEdgeInsertions()) << "$ & $"
        << FormatDouble2DP(stats.GetAverageEdgeDeletions()) << "$ & $"
        << FormatDouble2DP(stats.GetAverageEdgeRelabels()) << "$ \\\\\n";
    return true;
}

inline bool WriteTable2Tex(const std::filesystem::path& output_path,
                           const std::string& method,
                           const std::string& db,
                           const std::vector<Table2StrategyRow>& rows) {
    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to write LaTeX file: " << output_path << std::endl;
        return false;
    }

    ofs << "\\begin{table}[t]\n";
    ofs << "\\centering\n";
    ofs << "\\begin{tabular}{l c c c c}\n";
    ofs << "\\toprule\n";
    ofs << "Order & Unique Graphs & \\#Nodes & \\#Edges & \\#Discon.~Graphs \\\\\n";
    ofs << "\\midrule\n";
    if (rows.empty()) {
        ofs << "% no strategy data found for method " << method << " and dataset " << db << "\n";
    } else {
        for (const auto& row : rows) {
            ofs << row.order_label << " & -- & $" << FormatDouble2DP(row.avg_nodes)
                << "$ & $" << FormatDouble2DP(row.avg_edges)
                << "$ & $" << FormatDouble2DP(row.avg_unconnected) << "$ \\\\\n";
        }
    }
    ofs << "\\bottomrule\n";
    ofs << "\\end{tabular}\n";
    ofs << "\\caption{Heuristic results for " << db << ". Unique Graphs is left blank and can be filled by WL analysis.}\n";
    ofs << "\\end{table}\n";
    return true;
}


inline int
analyze_edit_path_graphs(const std::string& db,
                                    const std::string& edit_path_output,
                                    const std::string& method,
                                    const bool low_memory = false) {
    std::string edit_path_output_db = edit_path_output + method + "/" + db + "/";

    // print what you are doing in which dir
    std::cout << "Reading path data from directory: " << edit_path_output_db << "\n";
    std::cout << "Writing analysis to: " << edit_path_output_db + "Evaluation" << "\n";

    GraphData<UDataGraph> edit_paths;
    GraphData<UDataGraph>* edit_paths_ptr = nullptr;
    std::vector<std::tuple<INDEX, INDEX, INDEX, EditOperation>> edit_path_info;

    if (!low_memory) {
        edit_paths.Load(edit_path_output_db + db + "_edit_paths.bgf");
        edit_paths_ptr = &edit_paths;
    } else {
        std::cout << "Low-memory mode enabled: skipping BGF graph load and graph-level metrics.\n";
    }

    std::string info_path = edit_path_output_db + db + "_edit_paths_data.bin";
    ReadEditPathInfo(info_path, edit_path_info);
    if (edit_path_info.empty()) {
        std::cerr << "No edit path info entries loaded from: " << info_path << "\n";
        return 1;
    }
    EditPathStatistics stats(edit_paths_ptr, edit_path_info, !low_memory);
    stats.PrintStatistics();

    // Write evaluation CSVs under Results/Paths/<method>/<db>/Evaluation/ create directory if it does not exist
    std::string eval_dir = edit_path_output_db + "Evaluation/";
    if (!std::filesystem::exists(eval_dir)) {
        std::filesystem::create_directories(eval_dir);
    }
    stats.WriteSummaryJson(eval_dir, db, method, low_memory);
    stats.WriteCSVFiles(eval_dir);
    // Write per-path positions evaluation CSVs
    stats.WritePositionCSVFiles(eval_dir);

    std::filesystem::path results_root;
    bool extracted = ExtractResultsRoot(edit_path_output, results_root);
    if (!extracted) {
        std::cerr << "Warning: Could not infer Results root from edit path output '" << edit_path_output
                  << "'. Skipping LaTeX export.\n";
        return 0;
    }

    std::filesystem::path latex_dir = results_root / "Latex" / method / db;
    try {
        std::filesystem::create_directories(latex_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to create LaTeX output directory " << latex_dir
                  << ": " << e.what() << "\n";
        return 0;
    }

    const std::filesystem::path table1_row_path = latex_dir / "table1_row.tex";
    WriteTable1RowTex(table1_row_path, db, stats);

    const auto table2_rows = CollectTable2Rows(results_root, method, db);
    const std::filesystem::path table2_path = latex_dir / "table2.tex";
    WriteTable2Tex(table2_path, method, db, table2_rows);

    return 0;
}

#endif //GEDPATHS_ANALYZE_EDIT_PATH_GRAPHS_H
