#!/usr/bin/env python3
"""
Generator for synthetic graph datasets in TU Dortmund text format.

Currently provides the TRIANGLE_SQUARE dataset:
  100 connected Erdos-Renyi base graphs G(n=10, p=0.3). Each graph gets a new
  node u attached via an edge (v, u) at a uniformly random base node v, and the
  class motif is glued onto this edge:
    label 0: a triangle through (v, u)  -> one extra node w,  edges (v,w), (u,w)
    label 1: a square   through (v, u)  -> two extra nodes,   cycle v-u-w1-w2-v
  50 graphs per class, shuffled with a fixed seed.

The output is written to <dest>/<db>/ as <db>_A.txt, <db>_graph_indicator.txt,
<db>_graph_labels.txt, <db>_node_labels.txt and <db>_edge_labels.txt, so the
existing pipeline (CreateMappings) picks it up like any TU dataset and converts
it to BGF in Data/ProcessedGraphs/.

Usage:
  python python_src/synthetic_data_generator.py -db TRIANGLE_SQUARE
"""

from __future__ import annotations

import argparse
import os
import random

DEFAULT_DEST = os.path.join("Data", "Graphs")
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
                                rng: random.Random) -> tuple[int, list[tuple[int, int]]]:
    """Return (num_nodes, edges) for one TRIANGLE_SQUARE graph with the given label."""
    edges = erdos_renyi_connected(num_base_nodes, p, rng)
    v = rng.randrange(num_base_nodes)
    u = num_base_nodes  # node attached via the new edge
    edges.append((v, u))
    if label == 0:
        # triangle through the edge (v, u)
        w = num_base_nodes + 1
        edges.append((v, w))
        edges.append((u, w))
        num_nodes = num_base_nodes + 2
    else:
        # square through the edge (v, u): cycle v-u-w1-w2-v
        w1 = num_base_nodes + 1
        w2 = num_base_nodes + 2
        edges.append((u, w1))
        edges.append((w1, w2))
        edges.append((w2, v))
        num_nodes = num_base_nodes + 3
    return num_nodes, edges


def generate_triangle_square(num_graphs: int, num_base_nodes: int, p: float,
                             seed: int) -> tuple[list[tuple[int, list[tuple[int, int]]]], list[int]]:
    rng = random.Random(seed)
    labels = [0] * (num_graphs // 2) + [1] * (num_graphs - num_graphs // 2)
    rng.shuffle(labels)
    graphs = [build_triangle_square_graph(num_base_nodes, p, label, rng)
              for label in labels]
    return graphs, labels


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
                        help="Number of nodes of the random base graph (default: 10)")
    parser.add_argument("--p", type=float, default=0.3,
                        help="Erdos-Renyi edge probability of the base graph (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"Output root directory (default: {DEFAULT_DEST})")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the dataset directory already exists")
    args = parser.parse_args()

    out_dir = os.path.join(args.dest, args.db)
    if os.path.isdir(out_dir) and not args.force:
        print(f"Dataset {args.db} already exists at {out_dir} (use --force to regenerate).")
        return 0

    graphs, labels = generate_triangle_square(args.num_graphs, args.nodes, args.p, args.seed)
    readme = (
        f"{args.db}: synthetic dataset generated by python_src/synthetic_data_generator.py\n\n"
        f"{args.num_graphs} graphs, {labels.count(0)} of class 0 and {labels.count(1)} of class 1.\n"
        f"Each graph is a connected Erdos-Renyi base graph G({args.nodes}, {args.p}).\n"
        f"A new node u is attached via an edge (v, u) at a random base node v, and the\n"
        f"class motif is glued onto this edge:\n"
        f"  label 0: triangle through (v, u) (one extra node)  -> {args.nodes + 2} nodes\n"
        f"  label 1: square through (v, u) (two extra nodes)   -> {args.nodes + 3} nodes\n"
        f"All node and edge labels are 0. Seed: {args.seed}.\n"
    )
    out_dir = write_tu_format(args.db, args.dest, graphs, labels, readme)

    num_nodes_total = sum(n for n, _ in graphs)
    num_edges_total = sum(len(e) for _, e in graphs)
    print(f"Generated {len(graphs)} graphs ({labels.count(0)}x label 0, {labels.count(1)}x label 1) "
          f"with {num_nodes_total} nodes and {num_edges_total} undirected edges in total.")
    print(f"Written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
