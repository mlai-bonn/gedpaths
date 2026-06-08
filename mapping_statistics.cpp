// define gurobi
#define GUROBI
// use gedlib wrapper helpers
#define GEDLIB

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <libGraph.h>
#include "Algorithms/GED/GEDLIBWrapper.h"
#include "src/create_edit_mappings.h"


struct UDataGraph;

namespace {

struct PairGroupStats {
    std::optional<double> avg_nodes;
    std::optional<double> avg_edges;
    std::optional<double> avg_node_diff;
    std::optional<double> avg_edge_diff;
    std::optional<double> extreme_nodes;
    std::optional<double> extreme_edges;
};

struct DatasetStats {
    std::string db;
    int valids = 0;
    int invalids = 0;
    int total = 0;
    std::optional<double> valid_percentage;
    PairGroupStats valid_stats;
    PairGroupStats invalid_stats;
};

std::string FormatDouble2DP(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value;
    return oss.str();
}

std::string FormatLatexGroupedInt(long long value) {
    std::string s = std::to_string(value);
    std::string out;
    out.reserve(s.size() + (s.size() / 3) * 2);
    for (size_t i = 0; i < s.size(); ++i) {
        if (i > 0 && ((s.size() - i) % 3 == 0)) {
            out += "\\,";
        }
        out += s[i];
    }
    return out;
}

std::string FormatLatexNumber2DP(const std::optional<double>& value) {
    if (!value.has_value()) {
        return "--";
    }
    return FormatDouble2DP(*value);
}

std::string FormatLatexInteger(const std::optional<double>& value) {
    if (!value.has_value()) {
        return "--";
    }
    std::ostringstream oss;
    oss << static_cast<long long>(std::llround(*value));
    return oss.str();
}

std::string EscapeLatex(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const char ch : text) {
        if (ch == '_') {
            escaped += "\\_";
        } else {
            escaped += ch;
        }
    }
    return escaped;
}

PairGroupStats ComputePairGroupStats(const std::vector<std::pair<UDataGraph, UDataGraph>>& pair_graphs,
                                     bool use_max_extreme) {
    PairGroupStats stats;
    if (pair_graphs.empty()) {
        return stats;
    }

    double avg_nodes = 0.0;
    double avg_edges = 0.0;
    double avg_node_diff = 0.0;
    double avg_edge_diff = 0.0;
    double extreme_nodes = use_max_extreme ? 0.0 : std::numeric_limits<double>::max();
    double extreme_edges = use_max_extreme ? 0.0 : std::numeric_limits<double>::max();

    for (const auto& pair : pair_graphs) {
        const auto& g1 = pair.first;
        const auto& g2 = pair.second;
        avg_nodes += (static_cast<double>(g1.nodes()) + static_cast<double>(g2.nodes())) / 2.0;
        avg_edges += (static_cast<double>(g1.edges()) + static_cast<double>(g2.edges())) / 2.0;
        avg_node_diff += std::abs(static_cast<double>(g1.nodes()) - static_cast<double>(g2.nodes()));
        avg_edge_diff += std::abs(static_cast<double>(g1.edges()) - static_cast<double>(g2.edges()));

        if (use_max_extreme) {
            extreme_nodes = std::max(extreme_nodes, static_cast<double>(g1.nodes()));
            extreme_nodes = std::max(extreme_nodes, static_cast<double>(g2.nodes()));
            extreme_edges = std::max(extreme_edges, static_cast<double>(g1.edges()));
            extreme_edges = std::max(extreme_edges, static_cast<double>(g2.edges()));
        } else {
            extreme_nodes = std::min(extreme_nodes, static_cast<double>(g1.nodes()));
            extreme_nodes = std::min(extreme_nodes, static_cast<double>(g2.nodes()));
            extreme_edges = std::min(extreme_edges, static_cast<double>(g1.edges()));
            extreme_edges = std::min(extreme_edges, static_cast<double>(g2.edges()));
        }
    }

    const double count = static_cast<double>(pair_graphs.size());
    stats.avg_nodes = avg_nodes / count;
    stats.avg_edges = avg_edges / count;
    stats.avg_node_diff = avg_node_diff / count;
    stats.avg_edge_diff = avg_edge_diff / count;
    stats.extreme_nodes = extreme_nodes;
    stats.extreme_edges = extreme_edges;
    return stats;
}

void PrintDatasetStats(const DatasetStats& stats) {
    std::cout << stats.db << ": " << stats.valids << " valid mappings, " << stats.invalids
              << " invalid mappings, " << stats.total << " total mappings, ";
    if (stats.valid_percentage.has_value()) {
        std::cout << *stats.valid_percentage << "% valid." << std::endl;
    } else {
        std::cout << "--% valid." << std::endl;
    }

    std::cout << stats.db << std::endl;
    std::cout << "\tAverage size of valid pairs: " << FormatLatexNumber2DP(stats.valid_stats.avg_nodes)
              << " nodes, " << FormatLatexNumber2DP(stats.valid_stats.avg_edges) << " edges." << std::endl;
    std::cout << "\tAverage size difference of valid pairs: "
              << FormatLatexNumber2DP(stats.valid_stats.avg_node_diff) << " nodes, "
              << FormatLatexNumber2DP(stats.valid_stats.avg_edge_diff) << " edges." << std::endl;
    std::cout << "\tMax size of valid pairs: " << FormatLatexInteger(stats.valid_stats.extreme_nodes)
              << " nodes, " << FormatLatexInteger(stats.valid_stats.extreme_edges) << " edges." << std::endl;

    std::cout << stats.db << std::endl;
    std::cout << "\tAverage size of invalid pairs: " << FormatLatexNumber2DP(stats.invalid_stats.avg_nodes)
              << " nodes, " << FormatLatexNumber2DP(stats.invalid_stats.avg_edges) << " edges." << std::endl;
    std::cout << "\tAverage size difference of invalid pairs: "
              << FormatLatexNumber2DP(stats.invalid_stats.avg_node_diff) << " nodes, "
              << FormatLatexNumber2DP(stats.invalid_stats.avg_edge_diff) << " edges." << std::endl;
    std::cout << "\tMin size of invalid pairs: " << FormatLatexInteger(stats.invalid_stats.extreme_nodes)
              << " nodes, " << FormatLatexInteger(stats.invalid_stats.extreme_edges) << " edges." << std::endl;
}

bool WriteLatexTable(const std::filesystem::path& output_path,
                     const std::string& method,
                     const std::vector<DatasetStats>& all_stats) {
    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to write LaTeX file: " << output_path << std::endl;
        return false;
    }

    ofs << "\\begin{table}[t]\n";
    ofs << "\\centering\n";
    ofs << "\\resizebox{\\textwidth}{!}{%\n";
    ofs << "\\begin{tabular}{l c c c c c c c c c c c c c c c c}\n";
    ofs << "\\toprule\n";
    ofs << "Dataset & Total & Valid & Invalid & \\% Valid"
           " & \\multicolumn{2}{c}{Avg N}"
           " & \\multicolumn{2}{c}{Avg E}"
           " & \\multicolumn{2}{c}{Avg $\\Delta$N}"
           " & \\multicolumn{2}{c}{Avg $\\Delta$E}"
           " & \\multicolumn{2}{c}{Size N}"
           " & \\multicolumn{2}{c}{Size E} \\\\\n";
    ofs << "\\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11} "
           "\\cmidrule(lr){12-13} \\cmidrule(lr){14-15} \\cmidrule(lr){16-17}\n";
    ofs << " &  &  &  &  & $\\checkmark$ & $\\times$ & $\\checkmark$ & $\\times$"
           " & $\\checkmark$ & $\\times$ & $\\checkmark$ & $\\times$"
           " & $\\checkmark$ (max) & $\\times$ (min) & $\\checkmark$ (max) & $\\times$ (min) \\\\\n";
    ofs << "\\midrule\n";

    for (const auto& stats : all_stats) {
        ofs << EscapeLatex(stats.db)
            << " & $" << FormatLatexGroupedInt(stats.total) << "$"
            << " & $" << FormatLatexGroupedInt(stats.valids) << "$"
            << " & $" << FormatLatexGroupedInt(stats.invalids) << "$"
            << " & $" << FormatLatexNumber2DP(stats.valid_percentage) << "$"
            << " & $" << FormatLatexNumber2DP(stats.valid_stats.avg_nodes) << "$"
            << " & $" << FormatLatexNumber2DP(stats.invalid_stats.avg_nodes) << "$"
            << " & $" << FormatLatexNumber2DP(stats.valid_stats.avg_edges) << "$"
            << " & $" << FormatLatexNumber2DP(stats.invalid_stats.avg_edges) << "$"
            << " & $" << FormatLatexNumber2DP(stats.valid_stats.avg_node_diff) << "$"
            << " & $" << FormatLatexNumber2DP(stats.invalid_stats.avg_node_diff) << "$"
            << " & $" << FormatLatexNumber2DP(stats.valid_stats.avg_edge_diff) << "$"
            << " & $" << FormatLatexNumber2DP(stats.invalid_stats.avg_edge_diff) << "$"
            << " & $" << FormatLatexInteger(stats.valid_stats.extreme_nodes) << "$"
            << " & $" << FormatLatexInteger(stats.invalid_stats.extreme_nodes) << "$"
            << " & $" << FormatLatexInteger(stats.valid_stats.extreme_edges) << "$"
            << " & $" << FormatLatexInteger(stats.invalid_stats.extreme_edges) << "$ \\\\\n";
    }

    ofs << "\\bottomrule\n";
    ofs << "\\end{tabular}%\n";
    ofs << "}\n";
    ofs << "\\caption{Mapping statistics for method " << method
        << " across datasets, split into valid and invalid mapping subsets.}\n";
    ofs << "\\end{table}\n";
    return true;
}

} // namespace

int main(int argc, const char* argv[]) {


    std::string method = "F2";
    std::filesystem::path path = "../Results/Mappings/" + method;
    std::filesystem::path processed_graph_path = "../Data/ProcessedGraphs/";

    std::vector<std::string> dbs = {"MUTAG", "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "NCI1", "DHFR", "NCI109", "Mutagenicity"};
    std::vector<DatasetStats> all_stats;
    all_stats.reserve(dbs.size());

    for (const auto& db : dbs) {
        GraphData<UDataGraph> graphs;
        LoadSaveGraphDatasets::LoadPreprocessedGraphData(db, processed_graph_path, graphs);


        std::vector<std::pair<INDEX, INDEX>> existing_pairs;
        auto results = std::vector<GEDEvaluation<UDataGraph>>{};
        // Load existing mappings if they exist and add their graph ids to existing_pairs
        get_existing_mappings(path, db, graphs, existing_pairs, results);

        // print statistics
        int valids = 0;
        int invalids = 0;
        std::vector<std::pair<UDataGraph, UDataGraph>> valid_pair_graphs;
        std::vector<std::pair<UDataGraph, UDataGraph>> invalid_pair_graphs;



        for (auto& result : results) {
           if (result.valid) {
               ++valids;
               valid_pair_graphs.emplace_back(*result.graphs.first, *result.graphs.second);
           }
           else {
               ++invalids;
               invalid_pair_graphs.emplace_back(*result.graphs.first, *result.graphs.second);
           }
        }

        DatasetStats stats;
        stats.db = db;
        stats.valids = valids;
        stats.invalids = invalids;
        stats.total = valids + invalids;
        if (stats.total > 0) {
            stats.valid_percentage = static_cast<double>(valids) * 100.0 / static_cast<double>(stats.total);
        }
        stats.valid_stats = ComputePairGroupStats(valid_pair_graphs, true);
        stats.invalid_stats = ComputePairGroupStats(invalid_pair_graphs, false);

        PrintDatasetStats(stats);
        all_stats.push_back(stats);
    }

    const auto latex_dir = std::filesystem::path("../Results/Latex") / method;
    std::filesystem::create_directories(latex_dir);
    WriteLatexTable(latex_dir / "mapping_statistics.tex", method, all_stats);

    return 0;
}
