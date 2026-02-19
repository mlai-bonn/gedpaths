import os
from os.path import dirname

from python_src.converter.GEDPathsInMemory import GEDPathsInMemoryDataset
from python_src.converter.bgf_to_torch_geometric import BGFInMemoryDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import argparse
import json
from joblib import Parallel, delayed


def weisfeiler_lehman_graph_hash(graph, iterations=10):
    return nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr="primary_node_labels",
        edge_attr="primary_edge_labels",
        iterations=iterations,
    )


# gets a pytorch-geometric dataset and outputs one WL hash per graph
def wl_analysis(dataset, iterations=10, threads=1):

    chunk_size = 10000
    num_graphs = len(dataset)
    index_chunks = [
        list(range(i, min(i + chunk_size, num_graphs)))
        for i in range(0, num_graphs, chunk_size)
    ]

    def compute_hash(indices, chunk_idx, total_chunks):
        local_hashes = []
        for idx in indices:
            g = dataset[idx]
            G = to_networkx(
                g,
                node_attrs=['primary_node_labels'],
                edge_attrs=['primary_edge_labels'],
                to_undirected=True,
            )
            local_hashes.append(weisfeiler_lehman_graph_hash(G, iterations=iterations))
        print(
            "Processed chunk {}/{} ({} graphs)".format(
                chunk_idx + 1,
                total_chunks,
                len(indices),
            )
        )
        return local_hashes

    total_chunks = len(index_chunks)
    chunk_hashes = Parallel(n_jobs=threads, backend="loky")(
        delayed(compute_hash)(indices, chunk_idx, total_chunks)
        for chunk_idx, indices in enumerate(index_chunks)
    )
    graph_hashes = []
    for local_hashes in chunk_hashes:
        graph_hashes.extend(local_hashes)
    return graph_hashes


def main():
    parser = argparse.ArgumentParser(description="Run WL analysis on GED path datasets.")
    parser.add_argument("-db", "--db", dest="db_name", default="Mutagenicity",
                        help="Dataset name (default: Mutagenicity)")
    parser.add_argument("-s", "--path_strategy", dest="strategy", default="d-E_d-IsoN",
                        help="Path strategy (default: d-E_d-IsoN)")
    parser.add_argument("-m", "--method", dest="method", default="F2",
                        help="Method (default: F2)")
    parser.add_argument("--bgf-path", dest="bgf_path", default=None,
                        help="Optional explicit path to the .bgf file (overrides db and strategy)")
    parser.add_argument("--wl-iterations", dest="wl_iterations", type=int, default=10,
                        help="WL hash iterations (default: 10)")
    parser.add_argument("--threads", dest="threads", type=int, default=16,
                        help="Number of threads for WL analysis (default: 1)")
    args = parser.parse_args()

    db_name = args.db_name
    strategy = args.strategy
    method = args.method
    wl_iterations = args.wl_iterations
    threads = args.threads

    if args.bgf_path:
        bgf_path = args.bgf_path
    else:
        bgf_path = f"Results/Paths_{strategy}/{method}/{db_name}/{db_name}_edit_paths.bgf"

    edit_operation_path = f"Results/Paths_{strategy}/{method}/{db_name}/{db_name}_edit_paths_data.txt"
    # Use the directory containing the bgf as the dataset root so the processed file
    # will be written to <root>/processed/data.pt
    root_dir = dirname(bgf_path) or "."
    output_dir = os.path.join(root_dir, "WLAnalysis")
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summary.json")
    hashes_path = os.path.join(output_dir, "hashes.txt")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
            same_config = (
                existing_summary.get("db_name") == db_name
                and existing_summary.get("strategy") == strategy
                and existing_summary.get("method") == method
                and existing_summary.get("bgf_path") == bgf_path
                and existing_summary.get("edit_operation_path") == edit_operation_path
                and existing_summary.get("wl_iterations") == wl_iterations
            )
            if same_config and os.path.exists(hashes_path):
                print(f"WL analysis already exists at: {summary_path}. Skipping computation.")
                return
        except (OSError, json.JSONDecodeError):
            pass

    if not os.path.exists(bgf_path):
        print(f"Warning: bgf file not found at {bgf_path}")

    ds = BGFInMemoryDataset(root=root_dir, path=bgf_path)
    print(f"Loaded dataset from: {ds.processed_paths[0]}")

    ds_ged = GEDPathsInMemoryDataset(root=root_dir, path=bgf_path, edit_path_data=edit_operation_path)
    graph_hashes = wl_analysis(ds_ged, iterations=wl_iterations, threads=threads)

    hash_counts = {}
    with open(hashes_path, "w", encoding="utf-8") as f:
        for graph_hash in graph_hashes:
            hash_counts[graph_hash] = hash_counts.get(graph_hash, 0) + 1
            f.write(f"{graph_hash} {hash_counts[graph_hash]}\n")

    print(f"Total unique graphs in GEDPathsInMemoryDataset: {len(hash_counts)}")


    summary = {
        "db_name": db_name,
        "strategy": strategy,
        "method": method,
        "bgf_path": bgf_path,
        "edit_operation_path": edit_operation_path,
        "total_graphs": len(ds_ged),
        "unique_graphs": len(hash_counts),
        "wl_iterations": wl_iterations,
        "threads": threads,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved WL analysis to: {output_dir}")


if __name__ == "__main__":
    main()
