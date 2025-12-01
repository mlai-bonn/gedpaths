"""
Utilities for edit-path BGF datasets.

Provides GEDPathsInMemoryDataset which extends BGFInMemoryDataset (from
torch_geometric_exporter) with convenience methods to fetch the sequence of
path-graphs for a given (start_id, end_id) pair.

Each Data object produced by the BGF loader contains the attributes
`edit_path_start`, `edit_path_end` and `edit_path_step` (integers). We build an
index mapping (start,end) -> ordered list of dataset indices (ordered by step).

Example usage
--------------
from python_src.converter.GEDPathsInMemory import GEDPathsInMemoryDataset

bgf_path = "Results/Paths/F2/MUTAG/MUTAG_edit_paths.bgf"
root_dir = os.path.dirname(bgf_path) or "."
ds = GEDPathsInMemoryDataset(root=root_dir, path=bgf_path)
# get the Data objects for pair (0, 1)
seq = ds.get_path_graphs(0, 1)
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, cast, Any, Optional
from torch_geometric.data import Data
from dataclasses import dataclass
import os

from .torch_geometric_exporter import BGFInMemoryDataset


@dataclass
class EditOperation:
    """Represent a single edit operation in an edit path.

    Fields mirror the C++ EditOperation tuple used in the repo where possible:
    - source: source graph id (int)
    - step: step id within the path (int)
    - target: target graph id (int)
    - operation: a small dict describing operation type/object/extra fields
    """
    source: int
    step: int
    target: int
    operation: Dict[str, Any]


class GEDPathsInMemoryDataset(BGFInMemoryDataset):
    """In-memory dataset helper for GED edit-path graphs.

    Builds an index that maps (start_id, end_id) -> ordered list of dataset
    indices corresponding to the edit path (ordered by edit_path_step). Use
    `get_path_graphs(start, end)` to obtain the Data objects for that pair.
    """

    def __init__(self, root: str, path: str, **kwargs):
        # initialize and load processed data (BGFInMemoryDataset does torch.load)
        # Accept optional `edit_path_data` kwarg pointing to a text/binary file
        self._edit_path_data: Optional[str] = kwargs.pop("edit_path_data", None)
        super().__init__(root=root, path=path, **kwargs)
        # Build the mapping from (start,end) -> list of indices, ordered by step
        self._pair_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        # mapping from (start,end) -> ordered list of EditOperation
        self._pair_to_operations: Dict[Tuple[int, int], List[EditOperation]] = defaultdict(list)
        self._build_index()
        # If an edit-path data file was provided, try to parse it
        if self._edit_path_data:
            try:
                self._load_edit_path_data(self._edit_path_data)
            except Exception:
                # best-effort: don't fail dataset creation if parsing fails
                pass

    def _build_index(self) -> None:
        """Scan all graphs in the dataset and create the mapping.

        The dataset must have `edit_path_start`, `edit_path_end` and
        `edit_path_step` set on each Data object (this is done by the BGF loader).

        This implementation tries a fast-path that reads the attributes directly
        from the collated `self.data` tensors (cheap). If that isn't possible
        it falls back to the safe but slower `self.get(i)` loop.
        """
        # If the processed file is empty or dataset not populated, nothing to do
        try:
            n = len(self)
        except Exception:
            # Best-effort: if len() fails, try to infer from slices
            if hasattr(self, "slices") and isinstance(self.slices, dict):
                any_slice = next(iter(self.slices.values()))
                n = len(any_slice)
            else:
                n = 0

        temp_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)

        # Fast path: if the collated `self.data` contains per-graph attributes
        # `edit_path_start`, `edit_path_end`, `edit_path_step` we can extract
        # them without calling `self.get(i)` which is much faster for large
        # datasets.
        fast_done = False
        try:
            if n > 0 and hasattr(self, "data") and hasattr(self.data, "edit_path_start") and hasattr(self.data, "edit_path_end") and hasattr(self.data, "edit_path_step"):
                # Attempt to convert these attributes to flat Python lists
                def _to_list(x):
                    # Handle torch tensor
                    try:
                        import torch
                        if isinstance(x, torch.Tensor):
                            return x.cpu().numpy().reshape(-1).tolist()
                    except Exception:
                        pass
                    # Handle numpy array or other sequence-like
                    try:
                        import numpy as _np
                        if isinstance(x, (_np.ndarray, list, tuple)):
                            return _np.asarray(x).reshape(-1).tolist()
                    except Exception:
                        pass
                    # If it's a scalar, replicate into length n only if slices are present
                    try:
                        if hasattr(x, '__len__'):
                            return list(x)
                    except Exception:
                        pass
                    return None

                starts = _to_list(self.data.edit_path_start)
                ends = _to_list(self.data.edit_path_end)
                steps = _to_list(self.data.edit_path_step)

                if starts is not None and ends is not None and steps is not None and len(starts) == n and len(ends) == n and len(steps) == n:
                    for i in range(n):
                        try:
                            key = (int(starts[i]), int(ends[i]))
                            step = int(steps[i])
                        except Exception:
                            continue
                        temp_map[key].append((step, i))
                    fast_done = True
        except Exception:
            fast_done = False

        if not fast_done:
            # Fallback slow but robust path: iterate items and read attributes via get(i)
            for i in range(n):
                try:
                    d: Any = self.get(i)
                except Exception:
                    # If get fails, skip this index
                    continue
                # Read expected attributes; if missing, skip
                if not hasattr(d, "edit_path_start") or not hasattr(d, "edit_path_end") or not hasattr(d, "edit_path_step"):
                    continue
                key = (int(d.edit_path_start), int(d.edit_path_end))
                step = int(d.edit_path_step)
                temp_map[key].append((step, i))

        # Sort by step and store only indices
        for key, step_idx_list in temp_map.items():
            sorted_list = sorted(step_idx_list, key=lambda si: si[0])
            self._pair_to_indices[key] = [idx for _step, idx in sorted_list]

    def get_path_graph_indices(self, start: int, end: int) -> List[int]:
        """Return the dataset indices for the path from `start` to `end`.

        The indices are ordered by `edit_path_step`.
        """
        return list(self._pair_to_indices.get((int(start), int(end)), []))

    def get_path_graphs(self, start: int, end: int) -> List[Data]:
        """Return a list of `torch_geometric.data.Data` objects representing the
        edit path graphs for the pair (start, end), ordered by edit step.
        """
        indices = self.get_path_graph_indices(start, end)
        # self.get() returns a dataset item (unknown exact runtime type); cast to Data
        return [cast(Data, self.get(i)) for i in indices]

    def get_path_operations(self, start: int, end: int) -> List[EditOperation]:
        """Return the list of EditOperation for the pair (start, end), ordered by step.

        If no edit-path data was loaded, returns an empty list.
        """
        return list(self._pair_to_operations.get((int(start), int(end)), []))

    def _load_edit_path_data(self, path: str) -> None:
        """Parse an edit-path data file into EditOperation objects.

        The repository contains C++ code that writes edit path info as a binary
        file (`ReadEditPathInfo` / `WriteEditPathInfo`). There isn't a Python
        reader in the repo; to be robust we support two simple text formats:
        1) A CSV-like text where each line encodes: source,step,target,op_type,op_object,...
        2) A whitespace-separated integer sequence per operation: source step target type object [extra...]

        This parser will attempt to autodetect a simple CSV or whitespace format.
        It will populate `_pair_to_operations` keyed by (source,target) with
        operations ordered by step.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ops_by_pair: Dict[Tuple[int, int], List[Tuple[int, EditOperation]]] = defaultdict(list)
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # try CSV (commas) first
                if "," in line:
                    parts = [p.strip() for p in line.split(",")]
                else:
                    parts = line.split()

                # Need at least source, step, target
                if len(parts) < 3:
                    continue
                try:
                    source = int(parts[0])
                    step = int(parts[1])
                    target = int(parts[2])
                except ValueError:
                    # not parseable; skip
                    continue

                # remaining fields describe operation; keep as raw tokens and also try to infer
                op_tokens = parts[3:]
                op_dict: Dict[str, Any] = {"raw": op_tokens}
                if op_tokens:
                    # try common encoding: type (INSERT/DELETE/RELABEL) and object (NODE/EDGE)
                    t = op_tokens[0].upper()
                    if t in ("INSERT", "DELETE", "RELABEL"):
                        op_dict["type"] = t
                        if len(op_tokens) > 1:
                            obj = op_tokens[1].upper()
                            op_dict["object"] = obj
                edop = EditOperation(source=source, step=step, target=target, operation=op_dict)
                ops_by_pair[(source, target)].append((step, edop))

        # sort lists by step and store
        for pair, step_ops in ops_by_pair.items():
            sorted_ops = [op for _s, op in sorted(step_ops, key=lambda si: si[0])]
            self._pair_to_operations[pair] = sorted_ops


# Optional simple CLI when the module is executed directly
if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a BGF edit-path dataset and query pairs.")
    parser.add_argument("bgf", help="Path to the edit-path BGF file")
    parser.add_argument("--pair", help="Pair as START,END (e.g. 0,1)", default=None)
    args = parser.parse_args()

    bgf_path = args.bgf
    root_dir = os.path.dirname(bgf_path) or "."
    ds = GEDPathsInMemoryDataset(root=root_dir, path=bgf_path)
    print(f"Dataset processed path: {ds.processed_paths[0]}")
    print(f"Total graphs: {len(ds)}")
    print(f"Available pairs (count={len(ds.available_pairs())}): {ds.available_pairs()[:20]}")

    if args.pair:
        try:
            s, e = args.pair.split(",")
            s_i = int(s.strip())
            e_i = int(e.strip())
        except Exception:
            raise SystemExit("--pair must be START,END with integers")
        idxs = ds.get_path_graph_indices(s_i, e_i)
        print(f"Indices for pair ({s_i},{e_i}): {idxs}")
        print(f"Number of steps: {len(idxs)}")
        if idxs:
            example = ds.get(idxs[0])
            print(f"Example Data object attributes: {{'num_nodes': example.num_nodes, 'bgf_name': getattr(example, 'bgf_name', None), 'edit_path_step': getattr(example, 'edit_path_step', None)}}")
