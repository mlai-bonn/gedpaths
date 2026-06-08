import argparse
import json
import os
import sys
from os.path import dirname

import networkx as nx
from joblib import Parallel, delayed
from torch_geometric.utils import to_networkx

from python_src.converter.GEDPathsInMemory import GEDPathsInMemoryDataset
from python_src.converter.bgf_to_torch_geometric import BGFInMemoryDataset


CANONICAL_PATH_STRATEGIES = [
    "Rnd",
    "Rnd_d-IsoN",
    "i-E_d-IsoN",
    "d-E_d-IsoN",
]
DEFAULT_DB_NAME = "Mutagenicity"
DEFAULT_METHOD = "F2"
DEFAULT_STRATEGY = "d-E_d-IsoN"
DEFAULT_RESULTS_ROOT = "Results"


def weisfeiler_lehman_graph_hash(graph, edge_labels, iterations=10):
    # if edge_labels
    if edge_labels:
        return nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="primary_node_labels",
            edge_attr="primary_edge_labels",
            iterations=iterations,
        )
    return nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr="primary_node_labels",
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
            # check if primary edge labels are None or zero tensor
            if g.primary_edge_labels is None or g.primary_edge_labels.numel() == 0:
                G = to_networkx(
                    g,
                    node_attrs=['primary_node_labels'],
                    to_undirected=True,
                )
                local_hashes.append(weisfeiler_lehman_graph_hash(G, False, iterations=iterations))
            else:
                G = to_networkx(
                    g,
                    node_attrs=['primary_node_labels'],
                    edge_attrs=['primary_edge_labels'],
                    to_undirected=True,
                )
                local_hashes.append(weisfeiler_lehman_graph_hash(G, True, iterations=iterations))
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


def get_all_datasets_for_strategy(results_root, method, strategy):
    method_path = os.path.join(results_root, f"Paths_{strategy}", method)
    if not os.path.exists(method_path):
        raise FileNotFoundError(f"Edit path directory does not exist: {method_path}")
    if not os.path.isdir(method_path):
        raise NotADirectoryError(f"Edit path path is not a directory: {method_path}")

    datasets = [
        entry
        for entry in os.listdir(method_path)
        if os.path.isdir(os.path.join(method_path, entry))
    ]
    return sorted(datasets)


def resolve_paths(results_root, method, strategy, db_name, bgf_path):
    if bgf_path:
        resolved_bgf_path = bgf_path
    else:
        resolved_bgf_path = os.path.join(
            results_root,
            f"Paths_{strategy}",
            method,
            db_name,
            f"{db_name}_edit_paths.bgf",
        )

    edit_operation_path = os.path.join(
        results_root,
        f"Paths_{strategy}",
        method,
        db_name,
        f"{db_name}_edit_paths_data.txt",
    )
    return resolved_bgf_path, edit_operation_path


def run_wl_analysis_for_config(db_name, strategy, method, results_root, bgf_path, wl_iterations, threads):
    resolved_bgf_path, edit_operation_path = resolve_paths(
        results_root=results_root,
        method=method,
        strategy=strategy,
        db_name=db_name,
        bgf_path=bgf_path,
    )

    if not os.path.exists(resolved_bgf_path):
        print(f"Warning: bgf file not found at {resolved_bgf_path}. Skipping.")
        return 1
    if not os.path.exists(edit_operation_path):
        print(f"Warning: edit path metadata not found at {edit_operation_path}. Skipping.")
        return 1

    root_dir = dirname(resolved_bgf_path) or "."
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
                and existing_summary.get("bgf_path") == resolved_bgf_path
                and existing_summary.get("edit_operation_path") == edit_operation_path
                and existing_summary.get("wl_iterations") == wl_iterations
            )
            if same_config and os.path.exists(hashes_path):
                print(f"WL analysis already exists at: {summary_path}. Skipping computation.")
                return 0
        except (OSError, json.JSONDecodeError):
            pass

    ds = BGFInMemoryDataset(root=root_dir, path=resolved_bgf_path)
    print(f"Loaded dataset from: {ds.processed_paths[0]}")

    ds_ged = GEDPathsInMemoryDataset(
        root=root_dir,
        path=resolved_bgf_path,
        edit_path_data=edit_operation_path,
    )
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
        "bgf_path": resolved_bgf_path,
        "edit_operation_path": edit_operation_path,
        "total_graphs": len(ds_ged),
        "unique_graphs": len(hash_counts),
        "wl_iterations": wl_iterations,
        "threads": threads,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved WL analysis to: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run WL analysis on GED path datasets.")
    parser.add_argument(
        "-db",
        "--db",
        dest="db_name",
        default=DEFAULT_DB_NAME,
        help=f"Dataset name (default: {DEFAULT_DB_NAME})",
    )
    parser.add_argument(
        "-all_db",
        action="store_true",
        help="Use all datasets found under Results/Paths_<path_strategy>/<method>/",
    )
    parser.add_argument(
        "-s",
        "--path_strategy",
        dest="strategy",
        default=DEFAULT_STRATEGY,
        help=f"Path strategy (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "-all_path_strategies",
        action="store_true",
        help="Use Rnd, Rnd_d-IsoN, i-E_d-IsoN, d-E_d-IsoN",
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        default=DEFAULT_METHOD,
        help=f"Method (default: {DEFAULT_METHOD})",
    )
    parser.add_argument(
        "--results-root",
        dest="results_root",
        default=DEFAULT_RESULTS_ROOT,
        help=f"Results root containing Paths_<path_strategy>/ directories (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--bgf-path",
        dest="bgf_path",
        default=None,
        help="Optional explicit path to the .bgf file (single-run only; overrides db and strategy)",
    )
    parser.add_argument(
        "--wl-iterations",
        dest="wl_iterations",
        type=int,
        default=10,
        help="WL hash iterations (default: 10)",
    )
    parser.add_argument(
        "--threads",
        dest="threads",
        type=int,
        default=16,
        help="Number of threads for WL analysis (default: 16)",
    )
    args = parser.parse_args()

    db_flags = {"-db", "--db"}
    strategy_flags = {"-s", "--path_strategy"}
    db_was_set = any(arg in db_flags for arg in sys.argv[1:])
    strategy_was_set = any(arg in strategy_flags for arg in sys.argv[1:])

    if args.all_db and db_was_set:
        parser.error("-db cannot be combined with -all_db.")
    if args.all_path_strategies and strategy_was_set:
        parser.error("-s/--path_strategy cannot be combined with -all_path_strategies.")
    if args.bgf_path and (args.all_db or args.all_path_strategies):
        parser.error("--bgf-path cannot be combined with -all_db or -all_path_strategies.")

    strategies = CANONICAL_PATH_STRATEGIES if args.all_path_strategies else [args.strategy]

    failures = []
    attempted_runs = 0
    successes = 0

    for strategy in strategies:
        try:
            databases = (
                get_all_datasets_for_strategy(args.results_root, args.method, strategy)
                if args.all_db
                else [args.db_name]
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            print(f"Error: {exc}")
            return 1

        if not databases:
            print(
                f"No dataset directories found for strategy '{strategy}' "
                f"under {os.path.join(args.results_root, f'Paths_{strategy}', args.method)}"
            )
            return 1

        for db_name in databases:
            attempted_runs += 1
            print(f"\n== WLAnalysis: {args.method} / {strategy} / {db_name} ==")
            rc = run_wl_analysis_for_config(
                db_name=db_name,
                strategy=strategy,
                method=args.method,
                results_root=args.results_root,
                bgf_path=args.bgf_path,
                wl_iterations=args.wl_iterations,
                threads=args.threads,
            )
            if rc == 0:
                successes += 1
            else:
                failures.append((strategy, db_name, rc))

    if attempted_runs == 0:
        print("Error: no datasets or strategies were processed.")
        return 1

    print(f"\nWLAnalysis summary for method {args.method}:")
    print(f"  Successful runs: {successes}/{attempted_runs}")
    if failures:
        print("  Failed runs:")
        try:
            for strategy, db_name, rc in failures:
                print(f"    {strategy} / {db_name} (exit code {rc})")
        except ValueError:
            pass

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
