from __future__ import annotations
import io
import os
import struct
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


# -------- stdlib helpers --------

def _read_exact(f: io.BufferedReader, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("Unexpected EOF.")
    return b

def _read_int(f, endian: str) -> int:
    return struct.unpack(endian + "i", _read_exact(f, 4))[0]

def _read_uint(f, endian: str) -> int:
    return struct.unpack(endian + "I", _read_exact(f, 4))[0]

def _read_size_t(f, endian: str, size_t_bytes: int) -> int:
    fmt = "Q" if size_t_bytes == 8 else "I"
    return struct.unpack(endian + fmt, _read_exact(f, size_t_bytes))[0]

def _read_string(f, endian: str) -> str:
    n = _read_uint(f, endian)
    if n == 0:
        return ""
    return _read_exact(f, n).decode("utf-8", errors="strict")

def _read_np_block(f, dtype: np.dtype, count: int) -> np.ndarray:
    if count == 0:
        return np.empty((0,), dtype=dtype)
    buf = _read_exact(f, count * dtype.itemsize)
    arr = np.frombuffer(buf, dtype=dtype, count=count)
    if arr.size != count:
        raise EOFError("Short read.")
    return arr

def _read_torch_block(f, dtype: torch.dtype, count: int) -> torch.Tensor:
    if count == 0:
        return torch.empty((0,), dtype=dtype)
    buf = _read_exact(f, count * torch.tensor([], dtype=dtype).element_size())
    arr = torch.frombuffer(buf, dtype=dtype)
    if arr.numel() != count:
        raise EOFError("Short read.")
    return arr


# -------- two-pass layout support --------

@dataclass
class _GraphHeader:
    name: str
    graph_type: int
    node_number: int
    node_features: int
    node_feature_names: List[str]
    edge_number: int
    edge_features: int
    edge_feature_names: List[str]


def bgf_to_pyg_data_list(
        path: str,
        *,
        endian: str = "<",          # '<' little-endian, '>' big-endian
        size_t_bytes: int = 8,      # 8 for 64-bit writers, 4 for 32-bit
        keep_feature_names: bool = True,
        out_dtype: np.dtype = np.float32,
        undirected: bool = True,
) -> Tuple[List[Data], int, int, int, int]:
    """
    Loader for a BGF layout where:
      - For all graphs: headers + edge lists are written first (pass 1),
      - Then, for each graph in order: node features followed by edge features (pass 2).
    Uses NumPy frombuffer on raw bytes for fast bulk reads.
    """
    assert endian in ("<", ">")
    assert size_t_bytes in (4, 8)
    st_dtype  = np.dtype("u8" if size_t_bytes == 8 else "u4").newbyteorder("<" if endian == "<" else ">")
    dbl_dtype = np.dtype("f8").newbyteorder("<" if endian == "<" else ">")

    headers: List[_GraphHeader] = []

    with open(path, "rb") as f:
        # ---- header ----
        compatibility_format_version = _read_int(f, endian)  # we store later on Data
        graph_number = _read_int(f, endian)
        if graph_number < 0 or graph_number > 10**7:
            raise ValueError("Unreasonable graph count; check endianness/size_t.")

        # ---- PASS 1: read per-graph metadata + topology (edges) for ALL graphs ----
        # Print periodic progress instead of per-header messages
        print(f"PASS 1/2 - reading headers and topology for {graph_number} graphs...")
        step_h = max(1, graph_number // 10) if graph_number > 0 else 1
        for i in range(graph_number):
            name = _read_string(f, endian)
            gtype = _read_int(f, endian)

            n  = _read_size_t(f, endian, size_t_bytes)
            nf = _read_uint(f, endian)
            node_feature_names = [_read_string(f, endian) for _ in range(nf)]

            m  = _read_size_t(f, endian, size_t_bytes)
            ef = _read_uint(f, endian)
            edge_feature_names = [_read_string(f, endian) for _ in range(ef)]

            headers.append(_GraphHeader(
                name=name,
                graph_type=int(gtype),
                node_number=int(n),
                node_features=int(nf),
                node_feature_names=node_feature_names,
                edge_number=int(m),
                edge_features=int(ef),
                edge_feature_names=edge_feature_names,
            ))
            if (i + 1) % step_h == 0 or (i + 1) == graph_number:
                print(f"  PASS 1: processed {i+1}/{graph_number} headers")


        # ---- PASS 2: read per-graph node/edge features (simplified, undirected option) ----
        max_node_labels = 0
        max_edge_labels = 0
        raw_data = {'x': [], 'edge_index': [], 'edge_attr': [], 'primary_node_labels': [], 'primary_edge_labels': []}

        print(f"PASS 2/2 - reading node/edge features for {len(headers)} graphs...")
        step2 = max(1, len(headers) // 10) if len(headers) > 0 else 1

        # precompute label indices
        node_label_indices = [ -1 if h.node_feature_names is None else next((j for j, nm in enumerate(h.node_feature_names) if nm.lower() == 'label'), -1) for h in headers ]
        edge_label_indices = [ -1 if h.edge_feature_names is None else next((j for j, nm in enumerate(h.edge_feature_names) if nm.lower() == 'label'), -1) for h in headers ]

        # compute maximum number of attribute columns (excluding label column if present)
        if len(headers) == 0:
            num_node_attributes = 0
            num_edge_attributes = 0
        else:
            try:
                num_node_attributes = max(0, max((h.node_features - 1) if node_label_indices[i] >= 0 else h.node_features for i, h in enumerate(headers)))
            except Exception:
                num_node_attributes = 0
            try:
                num_edge_attributes = max(0, max((h.edge_features - 1) if edge_label_indices[i] >= 0 else h.edge_features for i, h in enumerate(headers)))
            except Exception:
                num_edge_attributes = 0

        for idx, h in enumerate(headers):
            if (idx % step2) == 0:
                print(f"  PASS 2: starting graph {idx+1}/{len(headers)}: {h.name}")

            # NODE FEATURES
            if h.node_features > 0:
                block = _read_torch_block(f, torch.float64, h.node_number * h.node_features)
                x_arr = block.numpy().reshape((h.node_number, h.node_features)).astype(out_dtype)
            else:
                x_arr = np.empty((h.node_number, 0), dtype=out_dtype)
            raw_data['x'].append(x_arr)

            # primary node labels (or None)
            pnl = None
            nli = node_label_indices[idx]
            if nli >= 0 and x_arr.size:
                try:
                    col = x_arr[:, nli]
                    pnl = np.array(col, copy=True)
                    v = int(np.nanmax(col))
                    if v > max_node_labels:
                        max_node_labels = v
                except Exception:
                    pnl = None
            raw_data['primary_node_labels'].append(pnl)

            # EDGES
            m = h.edge_number
            if m > 0:
                ei = np.empty((2, m), dtype=np.int64)
            else:
                ei = np.empty((2, 0), dtype=np.int64)
            ea = None
            for e_i in range(m):
                u = _read_size_t(f, endian, size_t_bytes)
                v = _read_size_t(f, endian, size_t_bytes)
                if u >= h.node_number or v >= h.node_number:
                    raise ValueError("Invalid edge index; check endianness/size_t.")
                ei[0, e_i] = int(u)
                ei[1, e_i] = int(v)
                if h.edge_features > 0:
                    ef = _read_torch_block(f, torch.float64, h.edge_features)
                    if ea is None:
                        ea = np.empty((m, h.edge_features), dtype=out_dtype)
                    ea[e_i, :] = ef.numpy()

            # if undirected: duplicate edges and attributes
            if undirected and ei.shape[1] > 0:
                ei_rev = ei[[1, 0], :]
                ei_all = np.concatenate([ei, ei_rev], axis=1)
                ea_all = np.vstack([ea, ea]) if ea is not None else None
            else:
                ei_all = ei
                ea_all = ea

            # primary edge labels (or None)
            pel = None
            eli = edge_label_indices[idx]
            if ea_all is not None and eli >= 0:
                try:
                    pel_col = ea_all[:, eli]
                    pel = np.array(pel_col, copy=True)
                    v = int(np.nanmax(pel_col))
                    if v > max_edge_labels:
                        max_edge_labels = v
                except Exception:
                    pel = None

            raw_data['edge_index'].append(ei_all)
            raw_data['edge_attr'].append(ea_all)
            raw_data['primary_edge_labels'].append(pel)

            if (idx + 1) % step2 == 0 or (idx + 1) == len(headers):
                print(f"  PASS 2: processed {idx+1}/{len(headers)} graphs (max_node_labels={max_node_labels}, max_edge_labels={max_edge_labels})")


        # ---- FINAL ASSEMBLY: build Data objects ----
        data_list: List[Data] = []
        for i, h in enumerate(headers):
            if (i % step_h) == 0:
                print(f"ASSEMBLY: building Data object for graph {i+1}/{len(headers)}: {h.name}")
            # Convert numpy buffers to torch tensors for downstream ops
            x_np = raw_data['x'][i]
            if x_np.size == 0:
                x = None
            else:
                x = torch.from_numpy(x_np)

            ei_np = raw_data['edge_index'][i]
            edge_index = torch.from_numpy(ei_np).to(torch.long) if ei_np is not None else torch.empty((2, 0), dtype=torch.long)

            edge_attr = None
            if raw_data['edge_attr'][i] is not None:
                ea_np = raw_data['edge_attr'][i]
                if ea_np.size == 0:
                    edge_attr = None
                else:
                    edge_attr = torch.from_numpy(ea_np)

            # one-hot encode the column in x resp. edge_attr if the feature index name is "label"
            if h.node_feature_names is not None and x is not None:
                # iterate over copy of names to avoid mutation issues
                for col_idx, fname in enumerate(list(h.node_feature_names)):
                    if fname.lower() == "label":
                        try:
                            labels = torch.as_tensor(x[:, col_idx], dtype=torch.long)
                            one_hot = torch.nn.functional.one_hot(labels, num_classes=(max_node_labels + 1)).to(x.dtype)
                            x = torch.cat([x[:, :col_idx], one_hot, x[:, col_idx + 1:]], dim=1)
                        except Exception:
                            pass
            if h.edge_feature_names is not None and edge_attr is not None:
                for col_idx, fname in enumerate(list(h.edge_feature_names)):
                    if fname.lower() == "label":
                        try:
                            labels = torch.as_tensor(edge_attr[:, col_idx], dtype=torch.long)
                            one_hot = torch.nn.functional.one_hot(labels, num_classes=(max_edge_labels + 1)).to(edge_attr.dtype)
                            edge_attr = torch.cat([edge_attr[:, :col_idx], one_hot, edge_attr[:, col_idx + 1:]], dim=1)
                        except Exception:
                            pass

            d = Data(x=x, edge_index=edge_index, edge_attributes=edge_attr)
            # add metadata fields
            d.y = -1
            d.bgf_name = h.name
            d.node_attributes = torch.Tensor()
            pnl = raw_data['primary_node_labels'][i]
            pel = raw_data['primary_edge_labels'][i]
            d.primary_node_labels = torch.from_numpy(pnl).type(torch.long) if pnl is not None else torch.Tensor()
            d.primary_edge_labels = torch.from_numpy(pel).type(torch.long) if pel is not None else torch.Tensor()
            d.edge_attributes = torch.Tensor()
            # add zero labels
            # split the name into start graph end graph and edit step
            bgf_name_parts = h.name.split("_")
            if len(bgf_name_parts) > 3:
             d.edit_path_start = int(bgf_name_parts[-3])
             d.edit_path_end = int(bgf_name_parts[-2])
             d.edit_path_step = int(bgf_name_parts[-1])
            if keep_feature_names:
             d.node_feature_names = h.node_feature_names
             d.edge_feature_names = h.edge_feature_names
            data_list.append(d)
    num_node_labels = max_node_labels + 1
    num_edge_labels = max_edge_labels + 1

    return data_list, num_node_labels, num_node_attributes, num_edge_labels, num_edge_attributes


# -------- optional: InMemoryDataset wrapper --------

class BGFInMemoryDataset(InMemoryDataset):
    """
    Wrapper around the two-pass loader.
    """
    def __init__(
            self,
            root: str,
            path: str,
            *,
            endian: str = "<",
            size_t_bytes: int = 8,
            keep_feature_names: bool = True,
            out_dtype: np.dtype = np.float32,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self._bgf_path = path
        self._endian = endian
        self._size_t_bytes = size_t_bytes
        self._keep_feature_names = keep_feature_names
        self._out_dtype = out_dtype
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load processed file in a way that is compatible with multiple PyG versions.
        # Newer PyG versions save a tuple like (data, slices, sizes, data_cls).
        # Older versions saved (data, slices).
        out = torch.load(self.processed_paths[0], weights_only=False)
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
            # fallback: single-object save
            self.data = out
            self.slices = {}
            self.sizes = {}

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(self._bgf_path)]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        if not os.path.isfile(self._bgf_path):
            raise FileNotFoundError(self._bgf_path)

    def process(self):
        data_list, max_node_label, max_node_attr, max_edge_label, max_edge_attr = bgf_to_pyg_data_list(
            self._bgf_path,
            endian=self._endian,
            size_t_bytes=self._size_t_bytes,
            keep_feature_names=self._keep_feature_names,
            out_dtype=self._out_dtype,
        )


        os.makedirs(self.processed_dir, exist_ok=True)
        # Save in PyG-compatible format: (data, slices, sizes, data_cls)
        data, slices = self.collate(data_list)
        sizes = {
            'num_node_labels': int(max_node_label),
            'num_node_attributes': int(max_node_attr),
            'num_edge_labels': int(max_edge_label),
            'num_edge_attributes': int(max_edge_attr),
        }
        torch.save((data, slices, sizes, type(data)), self.processed_paths[0])
