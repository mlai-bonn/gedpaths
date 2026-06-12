#!/usr/bin/env python3
"""
Generator for synthetic graph datasets in TU Dortmund text format.

Currently provides the TRIANGLE_SQUARE dataset:
  100 connected Erdos-Renyi base graphs. A pendant edge (v, u) connects a
  uniformly random base node v to a new node u, and u is a vertex of the class
  motif attached behind it:
    label 0: base G(11, p) + triangle u-w1-w2      (motif: 3 nodes)
    label 1: base G(10, p) + square   u-w1-w2-w3   (motif: 4 nodes)
  The triangle base has one extra node so both classes have the same total
  number of nodes (nodes + 4 = 14 with the defaults).
  50 graphs per class, shuffled with a fixed seed.

The output is written to <dest>/<db>/ as <db>_A.txt, <db>_graph_indicator.txt,
<db>_graph_labels.txt, <db>_node_labels.txt and <db>_edge_labels.txt, so the
existing pipeline (CreateMappings) picks it up like any TU dataset and converts
it to BGF in Data/ProcessedGraphs/. Additionally, <processed_dest>/
<db>_motif_nodes.json records per graph the attachment node v and the motif
node indices (per-graph, 0-based) so the motif can be recovered downstream.

Usage:
  python python_src/synthetic_data_generator.py -db TRIANGLE_SQUARE
"""

from __future__ import annotations

import argparse
import json
import os
import random

DEFAULT_DEST = os.path.join("Data", "Graphs")
DEFAULT_PROCESSED_DEST = os.path.join("Data", "ProcessedGraphs")
SYNTHETIC_DATASETS = ["TRIANGLE_SQUARE"]


def erdos_renyi_connected(num_nodes: int, p: float, rng: random.Random,
                          max_tries: int = 1000) -> list[tuple[int, int]]:
    """Sample G(num_nodes, p) edges, resampling until the graph is connected."""
    for _ in range(max_tries):
        edges = [(i, j)
                 for i in range(num_nodes)
                 for j in range(i + 1, num_nodes)
                 if rng.random() < p]
        adjacency: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        for i, j in edges:
            adjacency[i].append(j)
            adjacency[j].append(i)
        # BFS connectivity check
        seen = {0}
        queue = [0]
        while queue:
            node = queue.pop()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if len(seen) == num_nodes:
            return edges
    raise RuntimeError(f"Could not sample a connected G({num_nodes}, {p}) "
                       f"within {max_tries} tries")


def build_triangle_square_graph(num_base_nodes: int, p: float, label: int,
                                rng: random.Random) -> tuple[int, list[tuple[int, int]], int, list[int]]:
    """Return (num_nodes, edges, attachment_node, motif_nodes) for one graph.

    A pendant edge (v, u) connects a random base node v to the new node u,
    which is a vertex of the attached motif. The triangle base gets one extra
    node so both classes end up with num_base_nodes + 4 nodes in total.
    """
    base = num_base_nodes + 1 if label == 0 else num_base_nodes
    edges = erdos_renyi_connected(base, p, rng)
    v = rng.randrange(base)
    u = base  # new node: far endpoint of the pendant edge, vertex of the motif
    edges.append((v, u))
    if label == 0:
        # triangle u-w1-w2 behind the pendant edge
        w1, w2 = base + 1, base + 2
        edges += [(u, w1), (w1, w2), (w2, u)]
        num_nodes = base + 3
        motif_nodes = [u, w1, w2]
    else:
        # square u-w1-w2-w3 behind the pendant edge
        w1, w2, w3 = base + 1, base + 2, base + 3
        edges += [(u, w1), (w1, w2), (w2, w3), (w3, u)]
        num_nodes = base + 4
        motif_nodes = [u, w1, w2, w3]
    return num_nodes, edges, v, motif_nodes


def generate_triangle_square(num_graphs: int, num_base_nodes: int, p: float,
                             seed: int) -> tuple[list[tuple[int, list[tuple[int, int]]]],
                                                 list[int], list[dict]]:
    rng = random.Random(seed)
    labels = [0] * (num_graphs // 2) + [1] * (num_graphs - num_graphs // 2)
    rng.shuffle(labels)
    graphs = []
    motif_info = []
    for graph_id, label in enumerate(labels):
        num_nodes, edges, v, motif_nodes = build_triangle_square_graph(
            num_base_nodes, p, label, rng)
        graphs.append((num_nodes, edges))
        motif_info.append({
            "graph_id": graph_id,
            "label": label,
            "attachment_node": v,
            "motif_nodes": motif_nodes,
        })
    return graphs, labels, motif_info


def write_tu_format(db: str, dest: str,
                    graphs: list[tuple[int, list[tuple[int, int]]]],
                    labels: list[int], readme: str) -> str:
    out_dir = os.path.join(dest, db)
    os.makedirs(out_dir, exist_ok=True)

    a_lines = []
    indicator_lines = []
    node_label_lines = []
    edge_label_lines = []
    node_offset = 0  # TU node ids are 1-based and global over all graphs
    for graph_id, (num_nodes, edges) in enumerate(graphs, start=1):
        indicator_lines.extend([str(graph_id)] * num_nodes)
        node_label_lines.extend(["0"] * num_nodes)
        for i, j in edges:
            a, b = node_offset + i + 1, node_offset + j + 1
            a_lines.append(f"{a}, {b}")
            a_lines.append(f"{b}, {a}")
            edge_label_lines.extend(["0", "0"])
        node_offset += num_nodes

    files = {
        f"{db}_A.txt": a_lines,
        f"{db}_graph_indicator.txt": indicator_lines,
        f"{db}_graph_labels.txt": [str(label) for label in labels],
        f"{db}_node_labels.txt": node_label_lines,
        f"{db}_edge_labels.txt": edge_label_lines,
    }
    for name, lines in files.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(readme)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic TU-format graph dataset")
    parser.add_argument("-db", "-dataset", "-data", "-database", dest="db",
                        default="TRIANGLE_SQUARE", choices=SYNTHETIC_DATASETS,
                        help="Synthetic dataset to generate (default: TRIANGLE_SQUARE)")
    parser.add_argument("--num_graphs", type=int, default=100,
                        help="Total number of graphs, split evenly over both classes (default: 100)")
    parser.add_argument("--nodes", type=int, default=10,
                        help="Base graph size of the square class; the triangle class "
                             "uses one node more so both classes have nodes+4 nodes "
                             "in total (default: 10)")
    parser.add_argument("--p", type=float, default=0.3,
                        help="Erdos-Renyi edge probability of the base graph (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"Output root directory (default: {DEFAULT_DEST})")
    parser.add_argument("--processed_dest", default=DEFAULT_PROCESSED_DEST,
                        help="Directory for the motif-nodes JSON "
                             f"(default: {DEFAULT_PROCESSED_DEST})")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the dataset directory already exists")
    args = parser.parse_args()

    out_dir = os.path.join(args.dest, args.db)
    if os.path.isdir(out_dir) and not args.force:
        print(f"Dataset {args.db} already exists at {out_dir} (use --force to regenerate).")
        return 0

    graphs, labels, motif_info = generate_triangle_square(args.num_graphs, args.nodes,
                                                          args.p, args.seed)
    readme = (
        f"{args.db}: synthetic dataset generated by python_src/synthetic_data_generator.py\n\n"
        f"{args.num_graphs} graphs, {labels.count(0)} of class 0 and {labels.count(1)} of class 1.\n"
        f"Each graph is a connected Erdos-Renyi base graph; a pendant edge (v, u) connects\n"
        f"a random base node v to a new node u, which is a vertex of the class motif:\n"
        f"  label 0: base G({args.nodes + 1}, {args.p}) + triangle u-w1-w2    -> {args.nodes + 4} nodes\n"
        f"  label 1: base G({args.nodes}, {args.p}) + square u-w1-w2-w3 -> {args.nodes + 4} nodes\n"
        f"Both classes have the same total number of nodes ({args.nodes + 4}).\n"
        f"The attachment node v and the motif node indices (per-graph, 0-based) of each\n"
        f"graph are stored in {args.processed_dest}/{args.db}_motif_nodes.json.\n"
        f"All node and edge labels are 0. Seed: {args.seed}.\n"
    )
    out_dir = write_tu_format(args.db, args.dest, graphs, labels, readme)

    os.makedirs(args.processed_dest, exist_ok=True)
    motif_path = os.path.join(args.processed_dest, f"{args.db}_motif_nodes.json")
    with open(motif_path, "w") as f:
        json.dump(motif_info, f, indent=2)
    print(f"Motif node info written to {motif_path}")

    num_nodes_total = sum(n for n, _ in graphs)
    num_edges_total = sum(len(e) for _, e in graphs)
    print(f"Generated {len(graphs)} graphs ({labels.count(0)}x label 0, {labels.count(1)}x label 1) "
          f"with {num_nodes_total} nodes and {num_edges_total} undirected edges in total.")
    print(f"Written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
