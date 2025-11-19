import sys
import os
from os.path import dirname

from python_src.converter.GEDPathsInMemory import GEDPathsInMemoryDataset
from python_src.converter.torch_geometric_exporter import BGFInMemoryDataset
import torch
from torch_geometric.utils import to_networkx
import networkx as nx

def weisfeiler_lehman_graph_hash(graph, iterations=10, node_attr=None, edge_attr=None):
    return nx.weisfeiler_lehman_graph_hash(graph,  node_attr=node_attr, edge_attr=edge_attr, iterations=iterations)

# gets a pytorch-geometric dataset and outputs the number of unique graphs using WL analysis
def wl_analysis(dataset):
    unique_hashes = set()
    for i, data in enumerate(dataset):
        # print progress
        if (i + 1) % 1000 == 0 or i == 0:
            print("Processing graph {}/{}".format(i+1, len(dataset)))
        if dataset.data.get('primary_edge_labels', None) is None:
            G = to_networkx(data, node_attrs=['primary_node_labels'], to_undirected=True)
            graph_hash = weisfeiler_lehman_graph_hash(G, node_attr='primary_node_labels')
        else:
            G = to_networkx(data, node_attrs=['primary_node_labels'], edge_attrs=['primary_edge_labels'], to_undirected=True)
            graph_hash = weisfeiler_lehman_graph_hash(G, node_attr='primary_node_labels', edge_attr='primary_edge_labels')

        unique_hashes.add(graph_hash)
    print("Total unique graphs found: {}".format(len(unique_hashes)))
    return len(unique_hashes)


def main():
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

    ds = BGFInMemoryDataset(root=root_dir, path=bgf_path)
    print(f"Loaded dataset from: {ds.processed_paths[0]}")

    ds_ged = GEDPathsInMemoryDataset(root=root_dir, path=bgf_path, edit_path_data=edit_operation_path)
    unique_hashes = wl_analysis(ds_ged)
    print(f"Total unique graphs in GEDPathsInMemoryDataset: {unique_hashes}")

if __name__ == "__main__":
    main()
