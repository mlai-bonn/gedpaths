// define gurobi
#define GUROBI
// use gedlib wrapper helpers
#define GEDLIB

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <libGraph.h>
#include "Algorithms/GED/GEDLIBWrapper.h"
#include "src/create_edit_mappings.h"


struct UDataGraph;

int main(int argc, const char* argv[]) {


    std::string method = "F2";
    std::filesystem::path path = "../Results/Mappings/" + method;
    std::filesystem::path processed_graph_path = "../Data/ProcessedGraphs/";

    std::vector<std::string> dbs = {"MUTAG", "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "NCI1", "DHFR", "NCI109", "Mutagenicity"};

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

        // print num valids/invalids and percentage of valids per db + statistics over graphs of valids invalids
        std::cout << db << ": " << valids << " valid mappings, " << invalids << " invalid mappings, " << (valids + invalids) << " total mappings, " << (valids * 100.0 / (valids + invalids)) << "% valid." << std::endl;

        // average size of graph of valid pairs
        double avg_size = 0;
        double avg_edges = 0;
        double avg_size_diff = 0;
        double avg_edges_diff = 0;
        double max_size = 0;
        double max_edges = 0;
        for (auto& pair : valid_pair_graphs) {
            auto& g1 = pair.first;
            auto& g2 = pair.second;
            avg_size +=  (static_cast<double>(g1.nodes()) + static_cast<double>(g2.nodes())) / 2.0f;
            avg_edges +=  static_cast<double>(g1.edges()) + static_cast<double>(g2.edges()) / 2.0f;
            avg_size_diff += std::abs(static_cast<double>(g1.nodes()) - static_cast<double>(g2.nodes()));
            avg_edges_diff += std::abs(static_cast<double>(g1.edges()) - static_cast<double>(g2.edges()));
            max_size = std::max(max_size, static_cast<double>(g1.nodes()));
            max_size = std::max(max_size, static_cast<double>(g2.nodes()));
            max_edges = std::max(max_edges, static_cast<double>(g1.edges()));
            max_edges = std::max(max_edges, static_cast<double>(g2.edges()));
        }

        std::cout << db << std::endl;
        std::cout << "\t" << "Average size of valid pairs: " << avg_size / valid_pair_graphs.size() << " nodes, " << avg_edges / valid_pair_graphs.size() << " edges." << std::endl;
        std::cout << "\t" << "Average size difference of valid pairs: " << avg_size_diff / valid_pair_graphs.size() << " nodes, " << avg_edges_diff / valid_pair_graphs.size() << " edges." << std::endl;
        std::cout << "\t" << "Max size of valid pairs: " << max_size << " nodes, " << max_edges << " edges." << std::endl;

        // invalid pairs
        avg_size = 0;
        avg_edges = 0;
        avg_size_diff = 0;
        avg_edges_diff = 0;
        max_size = std::numeric_limits<double>::max();
        max_edges = std::numeric_limits<double>::max();
        for (auto& pair : invalid_pair_graphs) {
            auto& g1 = pair.first;
            auto& g2 = pair.second;
            avg_size += static_cast<double>(g1.nodes()) + static_cast<double>(g2.nodes()) / 2.0f;
            avg_edges += static_cast<double>(g1.edges()) + static_cast<double>(g2.edges()) / 2.0f;
            avg_size_diff += std::abs(static_cast<double>(g1.nodes()) - static_cast<double>(g2.nodes()));
            avg_edges_diff += std::abs(static_cast<double>(g1.edges()) - static_cast<double>(g2.edges()));
            max_size = std::min(max_size, static_cast<double>(g1.nodes()));
            max_size = std::min(max_size, static_cast<double>(g2.nodes()));
            max_edges = std::min(max_edges, static_cast<double>(g1.edges()));
            max_edges = std::min(max_edges, static_cast<double>(g2.edges()));
        }
        std::cout << db << std::endl;
        std::cout << "\t" << "Average size of invalid pairs: " << avg_size / invalid_pair_graphs.size() << " nodes, " << avg_edges / invalid_pair_graphs.size() << " edges." << std::endl;
        std::cout << "\t" << "Average size difference of invalid pairs: " << avg_size_diff / invalid_pair_graphs.size() << " nodes, " << avg_edges_diff / invalid_pair_graphs.size() << " edges." << std::endl;
        std::cout << "\t" << "Min size of invalid pairs: " << max_size << " nodes, " << max_edges << " edges." << std::endl;

    }
    return 0;
}
