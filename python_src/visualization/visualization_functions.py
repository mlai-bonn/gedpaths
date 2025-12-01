import os
import ast
from typing import Optional
import numpy as np

from networkx.drawing.nx_agraph import pygraphviz_layout

try:
    import torch
    from torch_geometric.data import InMemoryDataset
    from torch_geometric.utils import to_networkx
except Exception as e:
    raise ImportError("This module requires torch and torch_geometric. Install them before using: pip install torch torch-geometric") from e

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl


def _get_tab20_palette():
    """Return a list of hex colors for the tab20 colormap in a robust way.

    Some matplotlib colormap objects expose a `.colors` attribute (ListedColormap).
    If not present, sample the colormap at integer indices or evenly spaced points.
    """
    cmap = plt.get_cmap('tab20')
    colors = None
    try:
        # Some colormap implementations (ListedColormap) expose `.colors`.
        # Use getattr to avoid static analyzer unresolved attribute warnings.
        c = getattr(cmap, 'colors', None)
        if c is not None:
            colors = list(c)
        else:
            raise AttributeError('no colors attribute')
    except Exception:
        try:
            # Fallback: sample discrete entries if available
            n = getattr(cmap, 'N', None)
            if n is not None and n > 0:
                colors = [cmap(i) for i in range(n)]
            else:
                # Last fallback: sample 20 evenly spaced points
                colors = [cmap(i / 20.0) for i in range(20)]
        except Exception:
            # As an absolute fallback, return a small fixed palette
            colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                      (1.0, 0.4980392156862745, 0.054901960784313725),
                      (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                      (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)] * 5
    return [mpl.colors.to_hex(c) for c in colors]


# Module-level cached palette to avoid repeated sampling and to satisfy static analyzers
TAB20_PALETTE = _get_tab20_palette()


def compute_layout(data, prog: str = 'neato'):
    """Compute a 2D layout dict for a torch-geometric Data object.

    Strategy:
    - If `data.x` exists and has shape (N,2), use those coords directly.
    - Otherwise convert to networkx (using to_networkx) and try `pygraphviz_layout`.
    - Fall back to `nx.spring_layout` on failure.
    Returns: dict {node_index: (x, y)}
    """
    # Try to use explicit node coordinates from data.x
    try:
        if hasattr(data, 'x') and data.x is not None:
            try:
                x = data.x
                # handle torch tensor
                if hasattr(x, 'detach'):
                    x_np = x.detach().cpu().numpy()
                else:
                    x_np = np.array(x)
                if hasattr(x_np, 'ndim') and x_np.ndim == 2 and x_np.shape[1] == 2:
                    return {i: (float(x_np[i, 0]), float(x_np[i, 1])) for i in range(x_np.shape[0])}
            except Exception:
                pass
    except Exception:
        pass

    # Otherwise build a networkx graph and compute a layout
    try:
        G = to_networkx(data, node_attrs=['primary_node_labels'], to_undirected=True)
        try:
            pos = pygraphviz_layout(G, prog=prog)
            # ensure pos values are plain floats
            return {n: (float(pos[n][0]), float(pos[n][1])) for n in pos}
        except Exception:
            # fallback to spring layout (deterministic via fixed seed)
            pos = nx.spring_layout(G, seed=42)
            return {n: (float(pos[n][0]), float(pos[n][1])) for n in pos}
    except Exception:
        # final fallback: place nodes on a circle by index
        try:
            num_nodes = int(getattr(data, 'num_nodes', 0) or 0)
            if num_nodes <= 0:
                # try to infer from primary_node_labels or from edges
                try:
                    if hasattr(data, 'primary_node_labels') and data.primary_node_labels is not None:
                        num_nodes = int(len(data.primary_node_labels))
                except Exception:
                    num_nodes = 0
            if num_nodes <= 0:
                return {}
            import math
            return {i: (math.cos(2 * math.pi * i / max(1, num_nodes)), math.sin(2 * math.pi * i / max(1, num_nodes))) for i in range(num_nodes)}
        except Exception:
            return {}


class _LoadedInMemoryDataset(InMemoryDataset):
    """Tiny wrapper that lets us load a processed .pt file and use the
    InMemoryDataset interface (dataset[idx]) to obtain Data objects.

    We only implement the minimal properties required by torch-geometric.
    """
    def __init__(self, processed_path: str):
        self._processed_path = os.path.abspath(processed_path)
        # root is two directories above processed file when following
        # <root>/processed/data.pt, but we keep it flexible.
        root = os.path.dirname(os.path.dirname(self._processed_path))
        super().__init__(root)
        # load the processed file in a way compatible with multiple PyG versions
        out = torch.load(self._processed_path, weights_only=False)
        if isinstance(out, tuple):
            if len(out) == 2:
                self.data, self.slices = out
                self.sizes = {}
            elif len(out) == 3:
                self.data, self.slices, self.sizes = out
            else:
                # len >= 4: (data, slices, sizes, data_cls, ...)
                self.data, self.slices, self.sizes = out[0], out[1], out[2]
        else:
            # Fallback: single-object save
            self.data = out
            self.slices = {}
            self.sizes = {}

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [os.path.basename(self._processed_path)]


def find_processed_pt(path: str) -> Optional[str]:
    """Resolve a user-provided path to a processed .pt file.

    Accepts either: (1) a direct path to a .pt file, (2) a directory that
    contains a processed/data.pt, or (3) a directory that already contains
    data.pt.
    """
    if os.path.isfile(path) and path.endswith('.pt'):
        return os.path.abspath(path)
    if os.path.isdir(path):
        cand = os.path.join(path, 'processed', 'data.pt')
        if os.path.isfile(cand):
            return os.path.abspath(cand)
        cand2 = os.path.join(path, 'data.pt')
        if os.path.isfile(cand2):
            return os.path.abspath(cand2)
    return None


def load_data_by_index(processed_pt: str, idx: int):
    ds = _LoadedInMemoryDataset(processed_pt)
    if idx < 0 or idx >= len(ds):
        raise IndexError(f"Index {idx} out of range (0..{len(ds)-1})")
    return ds[idx]


def find_index_by_bgf_name(processed_pt: str, bgf_name: str) -> int:
    ds = _LoadedInMemoryDataset(processed_pt)
    # Search datasets for the bgf_name attribute on the Data objects
    for i in range(len(ds)):
        d = ds[i]
        if hasattr(d, 'bgf_name') and d.bgf_name == bgf_name:
            return i
    raise ValueError(f"No graph with bgf_name '{bgf_name}' found in {processed_pt}")


def graph_to_networkx_with_edge_features(data, one_hot_encoding: bool = True):
    """Convert a torch_geometric.data.Data to a networkx graph and
    prepare edge labels from edge_attr. Returns (G, edge_labels).
    """
    # Call to_networkx to build graph structure and node attributes only.
    # Collect edge attributes from the Data object directly to avoid
    # interoperability issues between PyG versions and attribute names.
    G = to_networkx(data, node_attrs=['primary_node_labels'], to_undirected=True)

    # Force undirected graph semantics for plotting/analysis
    try:
        G = nx.Graph(G)
    except Exception:
        # fallback: if conversion fails, keep original
        pass

    # Build edge_labels by reading edge attributes from the original Data object
    edge_labels = {}
    try:
        # data.edge_index expected shape [2, num_edges]
        ei = data.edge_index
        # prefer edge_attributes, else edge_attr
        if hasattr(data, 'edge_attributes'):
            eattr = data.edge_attributes
        elif hasattr(data, 'edge_attr'):
            eattr = data.edge_attr
        else:
            eattr = None

        if eattr is None:
            # fallback: no edge attributes available
            for u, v in G.edges():
                edge_labels[(u, v)] = ''
        else:
            # ensure tensors are on CPU numpy
            try:
                ei_np = ei.detach().cpu().numpy()
            except Exception:
                ei_np = ei.numpy()
            try:
                eattr_np = eattr.detach().cpu().numpy()
            except Exception:
                eattr_np = np.array(eattr)
            # iterate edges from edge_index and map attributes
            # edge_index may contain both directions; map accordingly
            for idx in range(ei_np.shape[1]):
                u = int(ei_np[0, idx]); v = int(ei_np[1, idx])
                val = eattr_np[idx]
                edge_labels[(u, v)] = val
    except Exception:
        # final fallback: populate empty labels
        for u, v in G.edges():
            edge_labels[(u, v)] = ''

    # Post-process edge_labels values into human-friendly strings
    import numpy as _np
    for k, attr in list(edge_labels.items()):
        try:
            a = attr
            # if a is torch tensor
            if hasattr(a, 'detach'):
                a = a.detach().cpu().numpy()
            # numpy array handling
            if hasattr(a, 'ndim'):
                if a.ndim == 0:
                    edge_labels[k] = f"{float(a):.3g}"
                elif a.ndim == 1:
                    if a.shape[0] == 1:
                        edge_labels[k] = f"{float(a[0]):.3g}"
                    elif a.shape[0] <= 4:
                        edge_labels[k] = '[' + ','.join(f"{float(x):.3g}" for x in a) + ']'
                    else:
                        edge_labels[k] = '[' + ','.join(f"{float(x):.3g}" for x in a[:6]) + '...]'
                elif a.ndim == 2:
                    flat = a.flatten()
                    if flat.shape[0] == 1:
                        edge_labels[k] = f"{float(flat[0]):.3g}"
                    elif flat.shape[0] <= 4:
                        edge_labels[k] = '[' + ','.join(f"{float(x):.3g}" for x in flat) + ']'
                    else:
                        edge_labels[k] = '[' + ','.join(f"{float(x):.3g}" for x in flat[:6]) + '...]'
                else:
                    flat = a.flatten()
                    edge_labels[k] = '[' + ','.join(f"{float(x):.3g}" for x in flat[:6]) + '...]'
            else:
                edge_labels[k] = str(a)
        except Exception:
            edge_labels[k] = str(attr)
    return G, edge_labels


def _draw_colored_edges(G, pos, edge_labels, ax, palette, edge_width=1.0, show_text_bbox=True):
    """Draw edges using `palette`.
    - Draw base light gray edges first.
    - If an edge's label parses as an int, draw that edge thick and colored by palette[int % len(palette)].
    - Otherwise draw the label text at the edge midpoint, colored by a categorical mapping.
    """
    # base faint edges
    try:
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=edge_width, ax=ax)
    except Exception:
        pass

    edge_list_colored = []
    edge_colors = []
    edge_text_items = []
    # iterate over actual graph edges to ensure consistent order and presence
    for u, v in G.edges():
        # look up label in either orientation
        lbl = edge_labels.get((u, v), edge_labels.get((v, u), None))
        if lbl is None:
            continue
        s = str(lbl)
        # Try to parse the label safely. It can be a scalar string like '1', or a list-like '[1,2]'.
        parsed_int = None
        try:
            val = ast.literal_eval(s)
            # if it's a sequence, take first element
            if isinstance(val, (list, tuple)) and len(val) > 0:
                candidate = val[0]
            else:
                candidate = val
            parsed_int = int(float(candidate))
        except Exception:
            # fallback: try to parse directly from the string as a number
            try:
                parsed_int = int(float(s))
            except Exception:
                parsed_int = None

        if parsed_int is not None:
            edge_list_colored.append((u, v))
            edge_colors.append(palette[parsed_int % len(palette)])
        else:
            try:
                midx = (pos[u][0] + pos[v][0]) / 2.0
                midy = (pos[u][1] + pos[v][1]) / 2.0
                edge_text_items.append((midx, midy, s))
            except Exception:
                continue

    if edge_list_colored:
        try:
            edge_widths = [1 for _ in edge_list_colored]
            # double edge width for if type is DELETE or relabel
            nx.draw_networkx_edges(G, pos, edgelist=edge_list_colored, edge_color=edge_colors, width=edge_width, ax=ax)
        except Exception:
            pass
        # When integer-labeled edges are present and colored, omit drawing any edge label text
        return

    if edge_text_items:
        unique_texts = sorted(list({t for (_, _, t) in edge_text_items}))
        mapping = {val: palette[i % len(palette)] for i, val in enumerate(unique_texts)}
        for midx, midy, text in edge_text_items:
            color = mapping.get(text, '#444444')
            try:
                if show_text_bbox:
                    ax.text(midx, midy, text, fontsize=8, color=color, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                else:
                    ax.text(midx, midy, text, fontsize=8, color=color, ha='center', va='center')
            except Exception:
                pass


# The main plotting utilities: plot_graph and plot_edit_path
# These were moved from plot_graph.py so the CLI script can stay minimal.

def plot_graph(data, title: Optional[str] = None, show_node_labels: bool = True, output: Optional[str] = None, color_nodes_by_label: bool = True, red_font_size: int = 10):
    # Convert to networkx and prepare labels
    G, edge_labels = graph_to_networkx_with_edge_features(data)
    standard_edge_width = 2.0
    # Match the standard node size used in the edit-path plotting helper
    standard_node_size = 200
    # Determine node positions: if node features exist and are 2-d, use them
    pos = None
    if hasattr(data, 'x') and data.x is not None:
        try:
            x = data.x.detach().cpu().numpy()
            if x.shape[1] == 2:
                pos = {i: (float(x[i, 0]), float(x[i, 1])) for i in range(x.shape[0])}
        except Exception:
            pos = None
    if pos is None:
        # Use Kamada-Kawai layout (often called "kawai/"kawaii" by typo)
        # It typically produces clearer layouts for small-to-medium graphs.
        try:
            pos = pygraphviz_layout(G, prog='neato', root=0, args=' -Gmode="KK" -Nmaxiter=10000 ')
        except Exception:
            # Fallback to spring layout if kamada_kawai fails for some graph
            pos = nx.spring_layout(G, seed=42)

    # Prepare node labels from attributes if requested.
    # We remove node ids from the inside labels (they will be shown outside).
    node_labels = None
    if show_node_labels:
        try:
            if hasattr(data, 'x') and data.x is not None:
                x = data.x
                # handle 1-D or 2-D numeric arrays -> only show the attribute values inside the node
                if x.ndim == 1:
                    node_labels = {i: f"{float(x[i]):.3g}" for i in range(x.shape[0])}
                elif x.ndim == 2 and x.shape[1] == 1:
                    node_labels = {i: f"{float(x[i,0]):.3g}" for i in range(x.shape[0])}
                elif x.ndim == 2 and x.shape[1] <= 4:
                    node_labels = {i: '[' + ','.join(f"{float(v):.3g}" for v in x[i].flatten()) + ']' for i in range(x.shape[0])}
                else:
                    try:
                        node_labels = {i: '[' + ','.join(f"{float(v):.3g}" for v in x[i].flatten()[:6]) + ']' for i in range(x.shape[0])}
                    except Exception:
                        node_labels = None
            else:
                # No node attributes -> do not draw any label inside the node
                node_labels = None
        except Exception:
            node_labels = None

    # draw nodes and edges
    # If requested, color nodes by their integer labels (0,1,2,...) using a fixed tab20 palette.
    # If node labels are not integer-like, fall back to categorical mapping.
    colored_legend = None
    palette = TAB20_PALETTE
    if color_nodes_by_label and node_labels is not None:
        node_order = list(G.nodes())
        vals = [node_labels.get(n) for n in node_order]
        # try parse as integers
        int_vals = []
        all_int = True
        for v in vals:
            try:
                iv = int(float(v))
                int_vals.append(iv)
            except Exception:
                all_int = False
                break

        if all_int and len(int_vals) > 0:
            # direct integer->color mapping (value modulo palette size)
            colors_hex = [palette[iv % len(palette)] for iv in int_vals]
            nx.draw_networkx_nodes(G, pos, nodelist=node_order, node_color=colors_hex, node_size=standard_node_size)
            # legend: show only unique integer categories present
            unique_ints = sorted(set(int_vals))
            legend_items = [(str(i), palette[i % len(palette)]) for i in unique_ints]
            colored_legend = ('categorical', legend_items)
        else:
            # fallback categorical mapping
            unique = sorted(list(dict.fromkeys(vals)))
            mapping = {val: palette[i % len(palette)] for i, val in enumerate(unique)}
            colors_hex = [mapping.get(v, '#cccccc') for v in vals]
            nx.draw_networkx_nodes(G, pos, nodelist=node_order, node_color=colors_hex, node_size=standard_node_size)
            legend_items = [(val, mapping[val]) for val in unique]
            colored_legend = ('categorical', legend_items)
    else:
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=standard_node_size)

    nx.draw_networkx_edges(G, pos, edge_color='gray')
    # draw edge colors/labels using helper (colors and optional labels)
    _draw_colored_edges(G, pos, edge_labels, plt.gca(), palette, edge_width=standard_edge_width)

    # draw labels: inside labels (bold black) show node attributes; outside ids (bold red) show node indices
    if node_labels is not None and (not color_nodes_by_label):
        # draw the label inside the node on top of the node marker (bold, black)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='black', font_weight='bold')

    # always draw node ids outside the nodes (bold, red)
    try:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        dx = (max(xs) - min(xs)) if xs else 1.0
        dy = (max(ys) - min(ys)) if ys else 1.0
        off = 0.05 * max(dx, dy, 1.0)
        pos_ids = {n: (pos[n][0] + off, pos[n][1] + off) for n in G.nodes()}
        id_labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos_ids, labels=id_labels, font_size=red_font_size, font_color='red', font_weight='bold')
    except Exception:
        pass

    # if we prepared a legend for colored nodes (or edges), add it on the right
    if colored_legend is not None:
        kind, payload = colored_legend
        fig = plt.gcf()
        if kind == 'categorical':
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=col, label=str(lbl)) for lbl, col in payload]
            plt.gca().legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    if title:
        # Add number of nodes and edges to the plot title
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        plt.title(f"{title} | Nodes: {num_nodes}, Edges: {num_edges}")
    plt.tight_layout()

    if output:
        # If output looks like a directory, create a filename; otherwise use it as file path
        out_path = output
        graph_idx = None
        if title and title.startswith("Graph index "):
            try:
                graph_idx = int(title.split("Graph index ")[1].split()[0])
            except Exception:
                graph_idx = None
        elif title and "(index " in title:
            try:
                graph_idx = int(title.split("(index ")[1].split(")")[0])
            except Exception:
                graph_idx = None
        # Compose filename with graph index if available
        if os.path.isdir(output) or output.endswith(os.path.sep):
            fname = "graph_plot"
            if graph_idx is not None:
                fname += f"_idx{graph_idx}"
            fname += ".png"
            out_path = os.path.join(output, fname)
        elif not (output.lower().endswith('.png') or output.lower().endswith('.jpg') or output.lower().endswith('.pdf')):
            # if user passed e.g. '/tmp/out' without extension, add .png and index
            out_path = output
            if graph_idx is not None:
                out_path += f"_idx{graph_idx}"
            out_path += ".png"
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot to: {out_path}")
    else:
        plt.show()


def plot_edit_path(graphs, edit_ops, output=None, show_labels=True, one_fig_per_step = False, color_nodes_by_label: bool = True, node_size: int = 200, edge_width: float = 1.0, red_font_size: int = 10):
    """
    Visualize an edit path between two graphs.
    Args:
        graphs: list of torch_geometric.data.Data objects representing the graph at each step.
        edit_ops: list of dicts or strings describing the edit operation at each step.
        output: if provided, save the figure(s) to this path (as PNG or PDF).
        highlight_colors: dict mapping operation types to colors (optional).
        show_labels: whether to show node labels.
        one_fig_per_step: plot one figure per step or all steps as subplots in one figure
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from math import ceil

    # allow caller to configure the default node marker size and the edge text/edge highlight width
    standard_node_size = node_size
    standard_edge_width = edge_width

    n_steps = len(graphs)

    # helper to parse an edit operation into a unified dict
    def _parse_edit_op(op):
        """Return dict with keys: type (str), nodes (set), edges (set of (u,v))."""
        res = {'type': 'none', 'nodes': set(), 'edges': set()}
        if op is None:
            return res
        # dict-like
        if isinstance(op, dict):
            # try common keys
            t = op.get('type') or op.get('op') or op.get('action') or 'none'
            res['type'] = str(t)
            # nodes: accept 'node', 'nodes'
            nodes = op.get('nodes') or op.get('node')
            if nodes is not None:
                if isinstance(nodes, (list, tuple, set)):
                    res['nodes'] = set(int(x) for x in nodes)
                else:
                    try:
                        res['nodes'] = {int(nodes)}
                    except Exception:
                        pass
            # edges: accept 'edge', 'edges' as pairs
            edges = op.get('edges') or op.get('edge')
            if edges is not None:
                e_set = set()
                if isinstance(edges, (list, tuple, set)):
                    for e in edges:
                        try:
                            u, v = e
                            e_set.add((int(u), int(v)))
                        except Exception:
                            continue
                else:
                    # single pair
                    try:
                        u, v = edges
                        e_set.add((int(u), int(v)))
                    except Exception:
                        pass
                res['edges'] = e_set
            return res

        # string-like: try formats 'optype', 'optype:payload' where payload can be '5' or '(1,2)'
        if isinstance(op, str):
            s = op.strip()
            if ':' in s:
                typ, payload = s.split(':', 1)
                typ = typ.strip()
                payload = payload.strip()
                res['type'] = typ
                # node id
                if payload.startswith('(') and payload.endswith(')'):
                    payload = payload[1:-1]
                if ',' in payload:
                    parts = [p.strip() for p in payload.split(',')]
                    if len(parts) == 2:
                        try:
                            u = int(parts[0]); v = int(parts[1])
                            res['edges'].add((u, v))
                        except Exception:
                            pass
                else:
                    try:
                        res['nodes'].add(int(payload))
                    except Exception:
                        pass
            else:
                res['type'] = s
            return res

        # list-like: [object, ids or id, operation type]
        if isinstance(op, list):
            # expecting 3 elements
            if len(op) == 3:
                obj = op[0]
                ids = op[1]
                typ = op[2]
                res['type'] = str(typ)
                if obj in ('node', 'nodes', 'NODE'):
                    if isinstance(ids, (list, tuple, set)):
                        res['nodes'] = set(int(x) for x in ids)
                    else:
                        try:
                            res['nodes'] = {int(ids)}
                        except Exception:
                            pass
                elif obj in ('edge', 'edges', 'EDGE'):
                    e_set = set()
                    # format is id1--id2
                    ids = ids.split('--')
                    try:
                        e_set.add((int(ids[0]), int(ids[1])))
                    except Exception:
                        pass
                    res['edges'] = e_set
                return res

        # fallback
        return res

    # New helper: format an edit op into a human-readable title like
    # "Insert Node 5", "Delete Edge 1 - 2", "Relabel Node 3" or "Initial".
    def _format_edit_op(op):
        parsed = _parse_edit_op(op)
        typ = str(parsed.get('type', 'none')).lower()

        # determine action
        action = None
        if any(k in typ for k in ('insert', 'add', 'create')):
            action = 'Insert'
        elif any(k in typ for k in ('delete', 'del', 'remove')):
            action = 'Delete'
        elif any(k in typ for k in ('sub', 'relabel', 'replace', 'subst')):
            action = 'Relabel'
        elif typ and typ != 'none':
            # fallback: capitalize first token
            action = typ.replace('_', ' ').capitalize()
        else:
            action = None

        # determine target (node(s) or edge(s))
        nodes = parsed.get('nodes', set())
        edges = parsed.get('edges', set())

        if nodes and not edges:
            # one or more nodes
            node_list = sorted(nodes)
            if len(node_list) == 1:
                target = f"Node {node_list[0]}"
            else:
                target = "Nodes " + ",".join(str(n) for n in node_list)
        elif edges:
            # show first edge or comma-separated list
            edge_list = sorted(list(edges))
            # pick representative: prefer a single pair
            rep = edge_list[0]
            try:
                u, v = rep
                target = f"Edge {u} - {v}"
            except Exception:
                target = "Edge"
        else:
            # no explicit nodes/edges provided; try infer from textual type
            if 'node' in typ:
                target = 'Node'
            elif 'edge' in typ:
                target = 'Edge'
            else:
                target = ''

        if action is None and not target:
            return 'Initial'

        if action and target:
            return f"{action} {target}"
        elif action:
            return action
        elif target:
            return target
        else:
            return 'Initial'

    # small helper that draws a single graph on the provided Axes and applies highlights
    # accepts an optional `color_mapping` which is either ('numeric', bins, palette) or ('categorical', mapping_dict, palette)
    # pos_override: if provided, use this layout dict instead of computing a new one
    def _draw_graph_on_ax(data, ax, title=None, show_node_labels=True, op=None, show_node_ids_outside=True, color_nodes_by_label=True, color_mapping=None, show_title=True, show_text_bbox=True, pos_override=None):
        G, edge_labels = graph_to_networkx_with_edge_features(data)

        # use provided layout if available, otherwise compute layout
        if pos_override is not None:
            pos = pos_override
        else:
            pos = compute_layout(data)

        # prepare node labels
        node_labels = None
        if show_node_labels:
            try:
                if hasattr(data, 'primary_node_labels') and data.primary_node_labels is not None:
                    x = data.primary_node_labels
                    # handle 1-D or 2-D numeric arrays -> only show the attribute values inside the node
                    if x.ndim == 1:
                        node_labels = {i: int(x[i]) for i in range(x.shape[0])}
                    elif x.ndim == 2 and x.shape[1] == 1:
                        node_labels = {i: f"{float(x[i,0]):.3g}" for i in range(x.shape[0])}
                    elif x.ndim == 2 and x.shape[1] <= 4:
                        node_labels = {i: '[' + ','.join(f"{float(v):.3g}" for v in x[i].flatten()) + ']' for i in range(x.shape[0])}
                    else:
                        try:
                            node_labels = {i: '[' + ','.join(f"{float(v):.3g}" for v in x[i].flatten()[:6]) + ']' for i in range(x.shape[0])}
                        except Exception:
                            node_labels = {i: str(i) for i in range(G.number_of_nodes())}
                else:
                    node_labels = {i: str(i) for i in range(G.number_of_nodes())}
            except Exception:
                node_labels = {i: str(i) for i in range(G.number_of_nodes())}



        # compute highlight sets from op
        parsed = _parse_edit_op(op)
        op_type = parsed.get('type', 'none')
        highlight_node_set = parsed.get('nodes', set())
        highlight_edge_set = parsed.get('edges', set())
        # normalize edge direction: networkx undirected edges may be (u,v) or (v,u)
        normalized_edge_set = set()
        for u, v in highlight_edge_set:
            normalized_edge_set.add((u, v))
            normalized_edge_set.add((v, u))

        # draw on the provided axis: draw defaults first, then overlay highlighted nodes/edges
        ax.clear()
        colored_legend = None


        # draw network nodes borders for operations (nodes with bigger sizes)
        hex_white = mpl.colors.to_hex(mpl.colors.to_rgb('black'))
        border_colors = [hex_white for _ in G.nodes()]
        border_factor = 4
        node_sizes = [standard_node_size for _ in G.nodes()]
        if op is not None:
            if op_type in ('DELETE', 'node_delete', 'Delete'):
                if parsed.get('nodes'):
                    n = list(parsed.get('nodes'))[0]
                    border_colors[n] = mpl.colors.to_hex(mpl.colors.to_rgb('red'))
                    node_sizes[n] = int(border_factor * standard_node_size)
            elif op_type in ('RELABEL', 'node_subst', 'Relabel'):
                if parsed.get('nodes'):
                    n = list(parsed.get('nodes'))[0]
                    border_colors[n] = mpl.colors.to_hex(mpl.colors.to_rgb('black'))
                    node_sizes[n] = border_factor * standard_node_size
            elif op_type in ('INSERT', 'node_insert', 'Insert'):
                if parsed.get('edges'):
                    source, target = list(parsed.get('edges'))[0]
                    border_colors[source] = mpl.colors.to_hex(mpl.colors.to_rgb('green'))
                    node_sizes[source] = border_factor * standard_node_size
                    border_colors[target] = mpl.colors.to_hex(mpl.colors.to_rgb('green'))
                    node_sizes[target] = border_factor * standard_node_size

        nx.draw_networkx_nodes(G, pos, node_color=border_colors, node_size=node_sizes, ax=ax)

        # draw nodes colored by label if requested
        if color_nodes_by_label and node_labels is not None:
            node_order = list(G.nodes())
            vals = [node_labels.get(n) for n in node_order]
            # prepare palette
            palette_local = TAB20_PALETTE
            # If a global color_mapping is provided, honor it (categorical mapping or numeric bins)
            if color_mapping is not None:
                kind = color_mapping[0]
                if kind == 'numeric':
                    bins = color_mapping[1]; pal = color_mapping[2]
                    colors_hex = []
                    for v in vals:
                        try:
                            fv = float(v)
                            b = 0
                            for i in range(len(bins) - 1):
                                if fv >= bins[i] and fv <= bins[i + 1]:
                                    b = i; break

                            colors_hex.append(pal[b % len(pal)])
                        except Exception:
                            colors_hex.append('#cccccc')
                    nx.draw_networkx_nodes(G, pos, nodelist=node_order, node_color=colors_hex, node_size=standard_node_size, ax=ax)
                    # prepare legend from bins
                    legend_items = [(f"{bins[i]:.3g}–{bins[i+1]:.3g}", pal[i % len(pal)]) for i in range(len(bins)-1)]
                    colored_legend = ('categorical', legend_items)
                elif kind == 'categorical':
                    mapping = color_mapping[1]; pal = color_mapping[2]
                    colors_hex = [palette_local[v] for v in vals]
                    nx.draw_networkx_nodes(G, pos, nodelist=node_order, node_color=colors_hex, node_size=standard_node_size, ax=ax)
                    legend_items = [(lbl, mapping[lbl]) for lbl in mapping.keys()]
                    colored_legend = ('categorical', legend_items)
                else:
                    # unknown color_mapping form -> fallback to local integer/categorical logic below
                    color_mapping = None

            if color_mapping is None:
                # Try integer mapping: parse each label as int(float(v)). If all succeed, map directly using palette index.
                int_vals = []
                all_int = True
                for v in vals:
                    try:
                        iv = int(float(v))
                        int_vals.append(iv)
                    except Exception:
                        all_int = False
                        break

                if all_int and len(int_vals) > 0:
                    colors_hex = [palette_local[iv % len(palette_local)] for iv in int_vals]
                    nx.draw_networkx_nodes(G, pos, nodelist=node_order, node_color=colors_hex, node_size=standard_node_size, ax=ax)
                    unique_ints = sorted(set(int_vals))
                    legend_items = [(str(i), palette_local[i % len(palette_local)]) for i in unique_ints]
                    colored_legend = ('categorical', legend_items)
                else:
                    # fallback categorical mapping using the string values
                    unique = sorted(list(dict.fromkeys(vals)))
                    mapping = {val: palette_local[i % len(palette_local)] for i, val in enumerate(unique)}
                    colors_hex = [mapping.get(v, '#cccccc') for v in vals]
                    nx.draw_networkx_nodes(G, pos, nodelist=node_order, node_color=colors_hex, node_size=standard_node_size, ax=ax)
                    legend_items = [(val, mapping[val]) for val in unique]
                    colored_legend = ('categorical', legend_items)
        else:
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=standard_node_size, ax=ax)

        # prepare palette for tab20
        palette = TAB20_PALETTE
        edge_widths = [1 for _ in G.edges()]
        edge_colors = [mpl.colors.to_hex(mpl.colors.to_rgb('lightgray')) for _ in G.edges()]
        # double edge width for if type is DELETE or relabel
        if parsed and parsed['edges']:
            for i, (u, v) in enumerate(G.edges()):
                if (u, v) in normalized_edge_set:
                    edge_widths[i] = 2 * standard_edge_width
                    edge_colors[i] = mpl.colors.to_hex(mpl.colors.to_rgb('red'))
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, ax=ax)
        # draw edge colors/labels using helper (colors and optional labels)
        if edge_labels:
            _draw_colored_edges(G, pos, edge_labels, ax, palette, edge_width=standard_edge_width, show_text_bbox=show_text_bbox)

        # draw labels: inside labels (bold black) show node attributes; outside ids (bold red) show node indices
        if node_labels is not None and (not color_nodes_by_label):
            # draw the label inside the node on top of the node marker (bold, black)
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='black', font_weight='bold', ax=ax)

        # always draw node ids outside the nodes (bold, red)
        try:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            dx = (max(xs) - min(xs)) if xs else 1.0
            dy = (max(ys) - min(ys)) if ys else 1.0
            off = 0.05 * max(dx, dy, 1.0)
            pos_ids = {n: (pos[n][0] + off, pos[n][1] + off) for n in G.nodes()}
            id_labels = {n: str(n) for n in G.nodes()}
            nx.draw_networkx_labels(G, pos_ids, labels=id_labels, font_size=red_font_size, font_color='red', font_weight='bold', ax=ax)
        except Exception:
            pass

        # return legend info (kind, payload) so caller can place a single legend on the figure if desired
        # Set the axis title to the formatted operation string (user requested)
        try:
            if show_title:
                title_text = None
                # Prefer an explicit op description if provided
                if op is not None:
                    title_text = _format_edit_op(op)
                # fall back to caller-provided title
                if (title_text is None or title_text == '') and title:
                    title_text = title
                if title_text is None or title_text == '':
                    title_text = 'Initial'
                ax.set_title(title_text)
        except Exception:
            if show_title:
                try:
                    if title:
                        ax.set_title(str(title))
                except Exception:
                    pass

        return colored_legend

    if n_steps == 0:
        raise ValueError("No graphs provided to plot_edit_path")

    # Build a global color_mapping across all graphs so colors are consistent across steps
    global_color_mapping = None
    if color_nodes_by_label:
        all_vals = []
        all_numeric = True
        numeric_list = []
        import numpy as np
        for g in graphs:
            try:
                if hasattr(g, 'x') and g.x is not None:
                    try:
                        x = g.x.detach().cpu().numpy()
                    except Exception:
                        x = np.array(g.x)
                    if hasattr(x, 'ndim') and x.ndim == 1:
                        vals_g = [f"{float(x[i]):.3g}" for i in range(x.shape[0])]
                    elif hasattr(x, 'ndim') and x.ndim == 2 and x.shape[1] == 1:
                        vals_g = [f"{float(x[i,0]):.3g}" for i in range(x.shape[0])]
                    elif hasattr(x, 'ndim') and x.ndim >= 2:
                        vals_g = ['[' + ','.join(f"{float(v):.3g}" for v in x[i].flatten()) + ']' for i in range(x.shape[0])]
                    else:
                        vals_g = []
                else:
                    vals_g = []
            except Exception:
                vals_g = []
            for v in vals_g:
                all_vals.append(v)
                try:
                    numeric_list.append(float(v))
                except Exception:
                    all_numeric = False

        palette_local = TAB20_PALETTE
        if all_vals:
            if all_numeric:
                nbins = min(len(palette_local), 8)
                if numeric_list:
                    vmin = min(numeric_list); vmax = max(numeric_list)
                else:
                    vmin = 0.0; vmax = 0.0
                if vmin == vmax:
                    bins = [vmin - 0.5, vmin + 0.5]
                else:
                    bins = list(np.linspace(vmin, vmax, nbins + 1))
                global_color_mapping = ('numeric', bins, palette_local)
            else:
                unique_all = sorted(list(dict.fromkeys(all_vals)))
                mapping = {val: palette_local[i % len(palette_local)] for i, val in enumerate(unique_all)}
                global_color_mapping = ('categorical', mapping, palette_local)

    if one_fig_per_step:
        # produce one file (or show) per step using the internal drawer so we can apply highlights
        prev_pos = None
        # Save the source (initial) graph as the first plot (no title and with '_title')
        try:
            data_src = graphs[0]
            pos_src = compute_layout(data_src)
            # Build source output paths
            if output:
                if os.path.isdir(output) or output.endswith(os.path.sep):
                    base = output.rstrip(os.path.sep)
                    os.makedirs(base, exist_ok=True)
                    out_src = os.path.join(base, 'source.png')
                    out_src_title = os.path.join(base, 'source_title.png')
                else:
                    root, ext = os.path.splitext(output)
                    if ext == '':
                        ext = '.png'
                    out_src = f"{root}_source{ext}"
                    out_src_title = f"{root}_source_title{ext}"

                # create and save without title
                try:
                    fig_s, ax_s = plt.subplots(figsize=(8, 6))
                    _draw_graph_on_ax(data_src, ax_s, title='Source graph', show_node_labels=show_labels, op=None, show_node_ids_outside=True, color_nodes_by_label=color_nodes_by_label, color_mapping=global_color_mapping, show_title=False, show_text_bbox=False, pos_override=pos_src)
                    try:
                        ax_s.set_xticks([]); ax_s.set_yticks([])
                        for s in ax_s.spines.values():
                            s.set_visible(False)
                    except Exception:
                        pass
                    fig_s.savefig(out_src, dpi=300, bbox_inches='tight')
                    print(f"Saved source graph to: {out_src}")
                    plt.close(fig_s)
                except Exception as e:
                    print(f"Failed to save source (no title): {e}")

                # create and save with title
                try:
                    fig_st, ax_st = plt.subplots(figsize=(8, 6))
                    _draw_graph_on_ax(data_src, ax_st, title='Source graph', show_node_labels=show_labels, op=None, show_node_ids_outside=True, color_nodes_by_label=color_nodes_by_label, color_mapping=global_color_mapping, show_title=True, show_text_bbox=False, pos_override=pos_src)
                    try:
                        ax_st.set_xticks([]); ax_st.set_yticks([])
                        for s in ax_st.spines.values():
                            s.set_visible(False)
                    except Exception:
                        pass
                    fig_st.savefig(out_src_title, dpi=300, bbox_inches='tight')
                    print(f"Saved source graph with title to: {out_src_title}")
                    plt.close(fig_st)
                except Exception as e:
                    print(f"Failed to save source (with title): {e}")
        except Exception:
            # if anything goes wrong, continue with step plots
            pass

        for step in range(n_steps):
            data = graphs[step]
            G_step, _ = graph_to_networkx_with_edge_features(data)

            op = edit_ops[step] if step < len(edit_ops) else None
            title = None
            if step > 0:
                title = f"Step {step}" + f": {edit_ops[step-1]}"

            # compute/derive positions: reuse prev_pos where possible
            if step == 0:
                pos_for_step = compute_layout(data)
            else:
                # copy positions for nodes that still exist
                pos_for_step = {n: prev_pos[n] for n in G_step.nodes() if prev_pos is not None and n in prev_pos}
                # place new nodes near their neighbors if possible
                missing = [n for n in G_step.nodes() if n not in pos_for_step]
                if missing:
                    # compute average pos of existing positions to use as fallback center
                    if pos_for_step:
                        xs = [p[0] for p in pos_for_step.values()]
                        ys = [p[1] for p in pos_for_step.values()]
                        center = (sum(xs)/len(xs), sum(ys)/len(ys))
                    else:
                        center = (0.0, 0.0)
                    # deterministic placement for newly appearing nodes
                    for n in missing:
                        neigh = list(G_step.neighbors(n)) if hasattr(G_step, 'neighbors') else []
                        neigh_pos = [pos_for_step[u] for u in neigh if u in pos_for_step]
                        def _det_offset(node_id, scale):
                            # simple deterministic fractional offset in [-0.5*scale, +0.5*scale]
                            return (((node_id * 0.6180339887498948) % 1.0) - 0.5) * scale

                        if neigh_pos:
                            avgx = sum(p[0] for p in neigh_pos)/len(neigh_pos)
                            avgy = sum(p[1] for p in neigh_pos)/len(neigh_pos)
                            pos_for_step[n] = (avgx + _det_offset(n, 0.06), avgy + _det_offset(n + 17, 0.06))
                        else:
                            pos_for_step[n] = (center[0] + _det_offset(n, 0.1), center[1] + _det_offset(n + 17, 0.1))

            fig, ax = plt.subplots(figsize=(8, 6))
            legend_info = _draw_graph_on_ax(data, ax, title=title, show_node_labels=show_labels, op=op, show_node_ids_outside=True, color_nodes_by_label=color_nodes_by_label, color_mapping=global_color_mapping, show_text_bbox=False, pos_override=pos_for_step)

            # We save two variants per step: one WITHOUT title and one WITH title
            # Build output paths for both variants
            if output:
                if os.path.isdir(output) or output.endswith(os.path.sep):
                    base = output.rstrip(os.path.sep)
                    os.makedirs(base, exist_ok=True)
                    out_no_title = os.path.join(base, f"step_{step}.png")
                    out_with_title = os.path.join(base, f"step_{step}_title.png")
                else:
                    root, ext = os.path.splitext(output)
                    if ext == '':
                        ext = '.png'
                    out_no_title = f"{root}{ext}"
                    out_with_title = f"{root}_title{ext}"

                # First: create and save the variant WITHOUT title
                try:
                    fig_no, ax_no = plt.subplots(figsize=(8, 6))
                    _draw_graph_on_ax(data, ax_no, title=title, show_node_labels=show_labels, op=op, show_node_ids_outside=True, color_nodes_by_label=color_nodes_by_label, color_mapping=global_color_mapping, show_title=False, show_text_bbox=False, pos_override=pos_for_step)
                    try:
                        ax_no.set_xticks([])
                        ax_no.set_yticks([])
                        for spine in ax_no.spines.values():
                            spine.set_visible(False)
                    except Exception:
                        pass
                    fig_no.savefig(out_no_title, dpi=300, bbox_inches='tight')
                    print(f"Saved edit path step to: {out_no_title}")
                    plt.close(fig_no)
                except Exception as e:
                    print(f"Failed to save single-step figure (no title): {e}")

                # Second: create and save the variant WITH title (suffix '_title')
                try:
                    fig_t, ax_t = plt.subplots(figsize=(8, 6))
                    _draw_graph_on_ax(data, ax_t, title=title, show_node_labels=show_labels, op=op, show_node_ids_outside=True, color_nodes_by_label=color_nodes_by_label, color_mapping=global_color_mapping, show_title=True, show_text_bbox=False, pos_override=pos_for_step)
                    try:
                        ax_t.set_xticks([])
                        ax_t.set_yticks([])
                        for spine in ax_t.spines.values():
                            spine.set_visible(False)
                    except Exception:
                        pass
                    fig_t.savefig(out_with_title, dpi=300, bbox_inches='tight')
                    print(f"Saved edit path step with title to: {out_with_title}")
                    plt.close(fig_t)
                except Exception as e:
                    print(f"Failed to save single-step figure (with title): {e}")
            else:
                # No output path provided: show interactive plot (no title variant)
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    _draw_graph_on_ax(data, ax, title=title, show_node_labels=show_labels, op=op, show_node_ids_outside=True, color_nodes_by_label=color_nodes_by_label, color_mapping=global_color_mapping, show_title=False, pos_override=pos_for_step)
                    plt.show()
                except Exception:
                    plt.show()
    else:
        # single figure with subplots for each step: arrange into rows with max 5 columns
        max_cols = 5
        cols = min(max_cols, n_steps)
        rows = int(ceil(n_steps / cols))
        # figure size scales with number of columns and rows
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        # flatten axes regardless of shape
        import numpy as _np
        if rows == 1 and cols == 1:
            ax_list = [axes]
        else:
            try:
                ax_list = list(_np.array(axes).flatten())
            except Exception:
                # final fallback: try to iterate
                try:
                    ax_list = [a for a in axes]
                except Exception:
                    ax_list = [axes]

        legend_info = None
        prev_pos = None
        for i in range(rows * cols):
            if i < n_steps:
                ax = ax_list[i]
                data = graphs[i]
                G_i, _ = graph_to_networkx_with_edge_features(data)
                op = edit_ops[i] if i < len(edit_ops) else None
                title = None if i < len(edit_ops) else f"Target Graph"

                # derive positions for this subplot, reusing prev_pos where possible
                if i == 0:
                    pos_for_subplot = compute_layout(data)
                else:
                    pos_for_subplot = {n: prev_pos[n] for n in G_i.nodes() if prev_pos is not None and n in prev_pos}
                    missing = [n for n in G_i.nodes() if n not in pos_for_subplot]
                    if missing:
                        if pos_for_subplot:
                            xs = [p[0] for p in pos_for_subplot.values()]
                            ys = [p[1] for p in pos_for_subplot.values()]
                            center = (sum(xs)/len(xs), sum(ys)/len(ys))
                        else:
                            center = (0.0, 0.0)
                        import random
                        for n in missing:
                            neigh = list(G_i.neighbors(n)) if hasattr(G_i, 'neighbors') else []
                            neigh_pos = [pos_for_subplot[u] for u in neigh if u in pos_for_subplot]
                            def _det_offset(node_id, scale):
                                return (((node_id * 0.6180339887498948) % 1.0) - 0.5) * scale

                            if neigh_pos:
                                avgx = sum(p[0] for p in neigh_pos)/len(neigh_pos)
                                avgy = sum(p[1] for p in neigh_pos)/len(neigh_pos)
                                pos_for_subplot[n] = (avgx + _det_offset(n, 0.06), avgy + _det_offset(n + 17, 0.06))
                            else:
                                pos_for_subplot[n] = (center[0] + _det_offset(n, 0.1), center[1] + _det_offset(n + 17, 0.1))

                li = _draw_graph_on_ax(data, ax, title=title, show_node_labels=show_labels, op=op, show_node_ids_outside=True, color_nodes_by_label=color_nodes_by_label, color_mapping=global_color_mapping, show_title=False, pos_override=pos_for_subplot)
                prev_pos = pos_for_subplot
                # hide spines/ticks on each subplot to avoid visible frames
                try:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                except Exception:
                    pass
                if legend_info is None and li is not None:
                    legend_info = li
            else:
                ax = ax_list[i]
                ax.set_visible(False)

        plt.tight_layout()
        # if we have a legend for colored nodes, place it on the figure's right
        # Note: we intentionally do NOT add the legend into the subplot figure.
        # Instead, if a legend is available we will save it as a separate image
        # named 'legend.png' next to the subplot output file after saving the figure.

        if output:
            out_path = output
            if os.path.isdir(output) or output.endswith(os.path.sep):
                out_path = os.path.join(output, 'editpath_subplots.png')
            else:
                root, ext = os.path.splitext(output)
                if ext == '':
                    out_path = root + '.png'
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved edit path subplots to: {out_path}")

            # Also save a second version of the combined subplot figure WITH titles
            try:
                # Recreate the subplot figure but request titles on each subplot
                fig_title, axes_title = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
                if rows == 1 and cols == 1:
                    ax_title_list = [axes_title]
                else:
                    try:
                        ax_title_list = list(_np.array(axes_title).flatten())
                    except Exception:
                        try:
                            ax_title_list = [a for a in axes_title]
                        except Exception:
                            ax_title_list = [axes_title]

                for i in range(rows * cols):
                    if i < n_steps:
                        ax_t = ax_title_list[i]
                        data = graphs[i]
                        op = edit_ops[i] if i < len(edit_ops) else None
                        _draw_graph_on_ax(data, ax_t, title=None, show_node_labels=show_labels, op=op, show_node_ids_outside=True, color_nodes_by_label=color_nodes_by_label, color_mapping=global_color_mapping, show_title=True)
                    else:
                        try:
                            ax_title_list[i].set_visible(False)
                        except Exception:
                            pass

                fig_title.tight_layout()
                root, ext = os.path.splitext(out_path)
                out_path_title = root + '_title' + (ext or '.png')
                fig_title.savefig(out_path_title, dpi=300, bbox_inches='tight')
                plt.close(fig_title)
                print(f"Saved edit path subplots with titles to: {out_path_title}")
            except Exception as e:
                print(f"Failed saving titled subplot figure: {e}")

            # Save separate legend image (legend.png) next to the subplot image
            if color_nodes_by_label and legend_info is not None:
                try:
                    kind, payload = legend_info
                    legend_out_dir = os.path.dirname(out_path) or '.'
                    legend_out = os.path.join(legend_out_dir, 'legend.png')
                    if kind == 'colorbar':
                        # payload is expected to be a mappable; create a small figure with the colorbar
                        fig_leg = plt.figure(figsize=(2, 3))
                        ax_leg = fig_leg.add_axes([0.05, 0.05, 0.9, 0.9])
                        fig_leg.colorbar(payload, cax=ax_leg)
                        fig_leg.savefig(legend_out, bbox_inches='tight', dpi=300)
                        plt.close(fig_leg)
                    elif kind == 'categorical':
                        import matplotlib.patches as mpatches
                        # create a legend-only figure
                        fig_leg = plt.figure(figsize=(2 + max(0, len(payload) - 1) * 0.3, 0.6 + (len(payload) // 4) * 0.25))
                        ax_leg = fig_leg.add_subplot(111)
                        ax_leg.axis('off')
                        patches = [mpatches.Patch(color=mpl.colors.to_hex(col), label=str(lbl)) for lbl, col in payload]
                        # place legend centered
                        ax_leg.legend(handles=patches, loc='center')
                        fig_leg.savefig(legend_out, bbox_inches='tight', dpi=300)
                        plt.close(fig_leg)
                    print(f"Saved legend to: {legend_out}")
                except Exception as e:
                    print(f"Failed saving legend: {e}")
            plt.close(fig)
        else:
            plt.show()

