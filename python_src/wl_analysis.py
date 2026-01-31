import os
from os.path import dirname

from python_src.converter.GEDPathsInMemory import GEDPathsInMemoryDataset
from python_src.converter.bgf_to_torch_geometric import BGFInMemoryDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import argparse


def weisfeiler_lehman_graph_hash(graph, iterations=10):
    return nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr="primary_node_labels",
        edge_attr="primary_edge_labels",
        iterations=iterations,
    )


# gets a pytorch-geometric dataset and outputs the number of unique graphs using WL analysis
def wl_analysis(dataset):
    unique_hashes = set()
    for i, data in enumerate(dataset):
        # print progress
        if (i + 1) % 1000 == 0 or i == 0:
            print("Processing graph {}/{}".format(i + 1, len(dataset)))
        G = to_networkx(data, node_attrs=['primary_node_labels'], edge_attrs=['primary_edge_labels'], to_undirected=True)
        graph_hash = weisfeiler_lehman_graph_hash(G)
        unique_hashes.add(graph_hash)
    print("Total unique graphs found: {}".format(len(unique_hashes)))
    return len(unique_hashes)


def main():
    parser = argparse.ArgumentParser(description="Run WL analysis on GED path datasets.")
    parser.add_argument("-db", "--db", dest="db_name", default="Mutagenicity",
                        help="Dataset name (default: Mutagenicity)")
    parser.add_argument("-s", "--strategy", dest="strategy", default="d-E_d-IsoN",
                        help="Path strategy (default: d-E_d-IsoN)")
    parser.add_argument("--bgf-path", dest="bgf_path", default=None,
                        help="Optional explicit path to the .bgf file (overrides db and strategy)")
    args = parser.parse_args()

    db_name = args.db_name
    strategy = args.strategy

    if args.bgf_path:
        bgf_path = args.bgf_path
    else:
        bgf_path = f"Results/Paths_{strategy}/F2/{db_name}/{db_name}_edit_paths.bgf"

    db_name = "NCI1"
    strategy = "Rnd_d-IsoN"
    #173897
    #174214
    #174481
    #174941
    bgf_path = f"Results/Paths_{strategy}/F2/{db_name}/{db_name}_edit_paths.bgf"
    edit_operation_path = f"Results/Paths_{strategy}/F2/{db_name}/{db_name}_edit_paths_data.txt"
    # Use the directory containing the bgf as the dataset root so the processed file
    # will be written to <root>/processed/data.pt
    root_dir = dirname(bgf_path) or "."

    if not os.path.exists(bgf_path):
        print(f"Warning: bgf file not found at {bgf_path}")

    ds = BGFInMemoryDataset(root=root_dir, path=bgf_path)
    print(f"Loaded dataset from: {ds.processed_paths[0]}")

    ds_ged = GEDPathsInMemoryDataset(root=root_dir, path=bgf_path, edit_path_data=edit_operation_path)
    unique_hashes = wl_analysis(ds_ged)
    print(f"Total unique graphs in GEDPathsInMemoryDataset: {unique_hashes}")


if __name__ == "__main__":
    main()
