#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - optional, handled via fallback paths
    np = None  # type: ignore[assignment]

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for this script.") from exc

from torch_geometric.datasets import GEDDataset, ZINC

GED_MIRROR_DEFAULT = "https://drive.google.com/drive/folders/1MOOUxxC_76Jseuc-JWaJ6B6LfU6-wNfR"
GED_MIRROR_MODES = ("auto", "force", "off")


@dataclass(frozen=True)
class DatasetSpec:
    folder_name: str
    source: str
    source_name: str
    max_nodes: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate targets.txt for datasets in Results/Mappings/Precomputed "
            "from official dataset sources."
        )
    )
    parser.add_argument(
        "--precomputed-root",
        default="Results/Mappings/Precomputed",
        help="Root folder containing per-dataset precomputed mapping folders.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset folders to process. Default: all folders under precomputed root.",
    )
    parser.add_argument(
        "--cache-root",
        default="Data/RawTargets",
        help="Local cache root for downloaded PyG/OGB datasets.",
    )
    parser.add_argument(
        "--ged-mirror-folder-url",
        default=GED_MIRROR_DEFAULT,
        help="Google Drive folder URL for GED dataset archives/pickles mirror.",
    )
    parser.add_argument(
        "--ged-mirror-mode",
        choices=GED_MIRROR_MODES,
        default="auto",
        help="Mirror usage mode for GED datasets: auto | force | off.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing targets.txt files.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail dataset on alignment/loader errors (default: true).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-dataset diagnostics.",
    )
    return parser.parse_args()


def discover_dataset_folders(precomputed_root: Path) -> list[str]:
    if not precomputed_root.exists() or not precomputed_root.is_dir():
        raise FileNotFoundError(f"Precomputed root does not exist: {precomputed_root}")
    return sorted(p.name for p in precomputed_root.iterdir() if p.is_dir())


def parse_dataset_spec(folder_name: str) -> DatasetSpec:
    if folder_name == "AIDS700nef":
        return DatasetSpec(folder_name, source="ged", source_name="AIDS700nef")
    if folder_name == "LINUX":
        return DatasetSpec(folder_name, source="ged", source_name="LINUX")
    if folder_name == "ALKANE":
        return DatasetSpec(folder_name, source="ged", source_name="ALKANE")
    if folder_name == "IMDBMulti":
        return DatasetSpec(folder_name, source="ged", source_name="IMDBMulti")

    imdb_match = re.fullmatch(r"IMDB-(\d+)", folder_name)
    if imdb_match:
        return DatasetSpec(
            folder_name,
            source="ged",
            source_name="IMDBMulti",
            max_nodes=int(imdb_match.group(1)),
        )

    molhiv_match = re.fullmatch(r"molhiv-(\d+)", folder_name)
    if molhiv_match:
        return DatasetSpec(
            folder_name,
            source="ogb",
            source_name="ogbg-molhiv",
            max_nodes=int(molhiv_match.group(1)),
        )

    code2_match = re.fullmatch(r"code2-(\d+)", folder_name)
    if code2_match:
        return DatasetSpec(
            folder_name,
            source="ogb",
            source_name="ogbg-code2",
            max_nodes=int(code2_match.group(1)),
        )

    zinc_match = re.fullmatch(r"zinc-(\d+)", folder_name)
    if zinc_match:
        return DatasetSpec(
            folder_name,
            source="zinc_auto",
            source_name="ZINC",
            max_nodes=int(zinc_match.group(1)),
        )

    raise ValueError(f"Unsupported dataset folder name: {folder_name}")


def _to_int_if_tensor(value: Any) -> int | None:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return int(value.reshape(-1)[0].item())
    return None


def _graph_index(data: Any, fallback: int) -> int:
    idx = getattr(data, "i", None)
    idx_tensor = _to_int_if_tensor(idx)
    if idx_tensor is not None:
        return idx_tensor
    if isinstance(idx, (int, np.integer if np is not None else int)):
        return int(idx)
    return fallback


def _ged_extract_exts(source_name: str) -> tuple[str, ...]:
    dataset = GEDDataset.datasets.get(source_name)
    if not dataset:
        return (".zip", ".tar", ".tar.gz", ".tgz")
    extract_fn = dataset.get("extract")
    fn_name = getattr(extract_fn, "__name__", "")
    if fn_name == "extract_zip":
        return (".zip",)
    return (".tar", ".tar.gz", ".tgz")


def _is_valid_ged_base(base: Path) -> bool:
    train_dir = base / "train"
    test_dir = base / "test"
    pickle_path = base / "ged.pickle"
    if not train_dir.is_dir() or not test_dir.is_dir() or not pickle_path.is_file():
        return False
    if not any(train_dir.glob("*.gexf")):
        return False
    if not any(test_dir.glob("*.gexf")):
        return False
    return True


def _pick_best_path(paths: list[Path], source_name: str) -> Path | None:
    if not paths:
        return None

    source_lower = source_name.lower()

    def score(p: Path) -> tuple[int, int, str]:
        name = p.name.lower()
        parent = str(p.parent).lower()
        s = 0
        if source_lower == name:
            s += 3
        if source_lower in name:
            s += 2
        if source_lower in parent:
            s += 1
        return (s, len(str(p)), str(p))

    return sorted(paths, key=score, reverse=True)[0]


def _find_ged_base_candidate(raw_dir: Path, source_name: str) -> Path | None:
    candidates: list[Path] = []
    direct = raw_dir / source_name
    if direct.is_dir():
        candidates.append(direct)

    for path in raw_dir.rglob("*"):
        if not path.is_dir():
            continue
        if (path / "train").is_dir() and (path / "test").is_dir():
            candidates.append(path)

    candidates = list(dict.fromkeys(candidates))
    return _pick_best_path(candidates, source_name)


def _collect_files(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file()]


def _find_best_archive(files: list[Path], source_name: str, exts: tuple[str, ...]) -> Path | None:
    exts_lower = tuple(e.lower() for e in exts)
    matching = [p for p in files if p.name.lower().endswith(exts_lower)]
    if not matching:
        return None
    source_specific = [p for p in matching if source_name.lower() in p.name.lower()]
    chosen = source_specific if source_specific else matching
    return _pick_best_path(chosen, source_name)


def _find_best_pickle(files: list[Path], source_name: str) -> Path | None:
    matching = [p for p in files if p.name.lower().endswith((".pickle", ".pkl"))]
    if not matching:
        return None

    source_specific = [p for p in matching if source_name.lower() in p.name.lower()]
    if source_specific:
        return _pick_best_path(source_specific, source_name)

    ged_named = [p for p in matching if "ged" in p.name.lower()]
    if ged_named:
        return _pick_best_path(ged_named, source_name)

    if len(matching) == 1:
        return matching[0]
    return _pick_best_path(matching, source_name)


def _extract_archive(archive: Path, raw_dir: Path) -> None:
    suffixes = tuple(s.lower() for s in archive.suffixes)
    if archive.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(raw_dir)
        return

    if archive.suffix.lower() == ".tar" or suffixes[-2:] == (".tar", ".gz") or archive.suffix.lower() == ".tgz":
        import tarfile

        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(raw_dir)
        return

    raise RuntimeError(f"Unsupported archive format: {archive}")


def _prepare_ged_raw_from_mirror(
    root: Path,
    source_name: str,
    mirror_folder_url: str,
    verbose: bool,
) -> None:
    try:
        import gdown  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "gdown is required for GED mirror mode. Install with `pip install gdown`."
        ) from exc

    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"ged_mirror_{source_name}_") as tmp_str:
        tmp_dir = Path(tmp_str)
        downloaded_dir = tmp_dir / "mirror"
        downloaded_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"[{source_name}] downloading mirror folder: {mirror_folder_url}")

        try:
            gdown.download_folder(url=mirror_folder_url, output=str(downloaded_dir), quiet=not verbose)
        except TypeError:
            gdown.download_folder(mirror_folder_url, output=str(downloaded_dir), quiet=not verbose)

        files = _collect_files(downloaded_dir)
        if not files:
            raise RuntimeError("Mirror folder download returned no files.")

        archive = _find_best_archive(files, source_name, _ged_extract_exts(source_name))
        if archive is None:
            raise RuntimeError(
                f"Could not find a matching archive in mirror folder for {source_name}."
            )
        if verbose:
            print(f"[{source_name}] using archive: {archive.name}")

        _extract_archive(archive, raw_dir)

        base = _find_ged_base_candidate(raw_dir, source_name)
        if base is None:
            raise RuntimeError(
                f"Archive extracted, but could not find train/test directories for {source_name}."
            )

        target_base = raw_dir / source_name
        if base != target_base:
            if not target_base.exists():
                shutil.move(str(base), str(target_base))
            else:
                for part in ("train", "test"):
                    src = base / part
                    dst = target_base / part
                    if src.is_dir() and not dst.exists():
                        shutil.move(str(src), str(dst))

        target_base.mkdir(parents=True, exist_ok=True)

        pickle_path = target_base / "ged.pickle"
        if not pickle_path.exists():
            pick = _find_best_pickle(files + _collect_files(raw_dir), source_name)
            if pick is None:
                raise RuntimeError(
                    f"Could not find GED pickle in mirror folder for {source_name}."
                )
            shutil.copy2(pick, pickle_path)



def ensure_ged_raw_ready(
    cache_root: Path,
    source_name: str,
    mirror_folder_url: str,
    mirror_mode: str,
    verbose: bool,
) -> Path:
    root = cache_root / f"{source_name}"
    raw_dir = root / "raw"
    target_base = raw_dir / source_name

    if _is_valid_ged_base(target_base):
        if verbose:
            print(f"[{source_name}] reusing local GED raw cache: {target_base}")
        return root

    if mirror_mode == "off":
        if verbose:
            print(f"[{source_name}] mirror disabled, using native GEDDataset download")
        return root

    try:
        _prepare_ged_raw_from_mirror(
            root=root,
            source_name=source_name,
            mirror_folder_url=mirror_folder_url,
            verbose=verbose,
        )
        if _is_valid_ged_base(target_base):
            if verbose:
                print(f"[{source_name}] prepared GED raw cache from mirror")
            return root
        raise RuntimeError("Mirror download finished but raw layout is still invalid.")
    except Exception as exc:
        if mirror_mode == "force":
            raise RuntimeError(
                f"Mirror mode is force and preparation failed for {source_name}: {exc}"
            ) from exc
        if verbose:
            print(f"[{source_name}] mirror preparation failed, fallback to native: {exc}")
        return root


def load_ged_graphs(
    cache_root: Path,
    source_name: str,
    mirror_folder_url: str,
    mirror_mode: str,
    verbose: bool,
) -> list[Any]:
    root = ensure_ged_raw_ready(
        cache_root=cache_root,
        source_name=source_name,
        mirror_folder_url=mirror_folder_url,
        mirror_mode=mirror_mode,
        verbose=verbose,
    )

    try:
        ds = GEDDataset(root=str(root), name=source_name)
        return [ds[i] for i in range(len(ds))]
    except TypeError:
        pass

    split_graphs: list[Any] = []
    for is_train in (True, False):
        ds = GEDDataset(root=str(root), name=source_name, train=is_train)
        split_graphs.extend(ds[i] for i in range(len(ds)))

    by_idx: dict[int, Any] = {}
    overflow: list[Any] = []
    for pos, g in enumerate(split_graphs):
        idx = _graph_index(g, pos)
        if idx in by_idx:
            overflow.append(g)
        else:
            by_idx[idx] = g

    ordered = [by_idx[k] for k in sorted(by_idx)]
    ordered.extend(overflow)
    return ordered


def load_ogb_graphs(cache_root: Path, source_name: str) -> list[Any]:
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "OGB is required to load molhiv/code2 datasets. "
            "Install dependency `ogb` first."
        ) from exc

    ds = PygGraphPropPredDataset(name=source_name, root=str(cache_root / "ogb"))
    return [ds[i] for i in range(len(ds))]


def load_zinc_graphs(cache_root: Path, subset: bool) -> list[Any]:
    root = str(cache_root / ("zinc_subset" if subset else "zinc_full"))
    graphs: list[Any] = []
    for split in ("train", "val", "test"):
        ds = ZINC(root=root, subset=subset, split=split)
        graphs.extend(ds[i] for i in range(len(ds)))
    return graphs


def graph_num_nodes(data: Any) -> int:
    n = getattr(data, "num_nodes", None)
    n_from_tensor = _to_int_if_tensor(n)
    if n_from_tensor is not None:
        return n_from_tensor
    if isinstance(n, int):
        return n

    x = getattr(data, "x", None)
    if torch.is_tensor(x):
        return int(x.size(0))

    edge_index = getattr(data, "edge_index", None)
    if torch.is_tensor(edge_index) and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1
    return 0


def apply_node_filter(graphs: Sequence[Any], max_nodes: int | None) -> list[Any]:
    if max_nodes is None:
        return list(graphs)
    return [g for g in graphs if graph_num_nodes(g) <= max_nodes]


def to_python_value(value: Any) -> Any:
    if torch.is_tensor(value):
        arr = value.detach().cpu()
        if arr.numel() == 1:
            return arr.item()
        return arr.reshape(-1).tolist()

    if np is not None and isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.reshape(-1).tolist()

    if isinstance(value, (list, tuple)):
        return [to_python_value(v) for v in value]
    return value


def flatten_sequence(value: Any) -> list[Any]:
    py = to_python_value(value)
    if isinstance(py, list):
        out: list[Any] = []
        for item in py:
            if isinstance(item, list):
                out.extend(flatten_sequence(item))
            else:
                out.append(item)
        return out
    return [py]


def scalar_line(data: Any) -> str:
    y = getattr(data, "y", None)
    if y is None:
        raise ValueError("Missing `y` target attribute.")
    py = to_python_value(y)
    if isinstance(py, list):
        if len(py) != 1:
            raise ValueError(f"Expected scalar target but got list of length {len(py)}.")
        py = py[0]
    return str(py)


def code2_line(data: Any) -> str:
    y = getattr(data, "y", None)
    if y is None:
        raise ValueError("Missing `y` target attribute.")
    tokens = flatten_sequence(y)
    if not tokens:
        raise ValueError("Empty sequence target for code2 graph.")
    return " ".join(str(t) for t in tokens)


def expected_graph_count_from_graphs_txt(graphs_txt: Path) -> int:
    if not graphs_txt.exists():
        raise FileNotFoundError(f"Missing graphs.txt: {graphs_txt}")
    with graphs_txt.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                return int(stripped.split()[0])
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse graph count from first non-empty line in {graphs_txt}: {stripped}"
                ) from exc
    raise ValueError(f"graphs.txt is empty: {graphs_txt}")


def write_targets_atomic(target_path: Path, lines: Iterable[str], overwrite: bool) -> None:
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} exists. Pass --overwrite to replace it.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def _dataset_lines(
    spec: DatasetSpec,
    cache_root: Path,
    expected_count: int,
    strict: bool,
    verbose: bool,
    ged_mirror_folder_url: str,
    ged_mirror_mode: str,
) -> list[str]:
    if spec.source == "ged":
        loaded = load_ged_graphs(
            cache_root=cache_root,
            source_name=spec.source_name,
            mirror_folder_url=ged_mirror_folder_url,
            mirror_mode=ged_mirror_mode,
            verbose=verbose,
        )
        filtered = apply_node_filter(loaded, spec.max_nodes)
    elif spec.source == "ogb":
        loaded = load_ogb_graphs(cache_root, spec.source_name)
        filtered = apply_node_filter(loaded, spec.max_nodes)
    elif spec.source == "zinc_auto":
        # Auto-select subset/full variant by matching local filtered count.
        loaded_subset = load_zinc_graphs(cache_root, subset=True)
        filtered_subset = apply_node_filter(loaded_subset, spec.max_nodes)

        loaded_full = load_zinc_graphs(cache_root, subset=False)
        filtered_full = apply_node_filter(loaded_full, spec.max_nodes)

        if len(filtered_subset) == expected_count:
            loaded = loaded_subset
            filtered = filtered_subset
            if verbose:
                print(f"[{spec.folder_name}] selected ZINC subset=True")
        elif len(filtered_full) == expected_count:
            loaded = loaded_full
            filtered = filtered_full
            if verbose:
                print(f"[{spec.folder_name}] selected ZINC subset=False")
        else:
            msg = (
                f"[{spec.folder_name}] Could not auto-match ZINC variant by count. "
                f"expected={expected_count}, subset_filtered={len(filtered_subset)}, "
                f"full_filtered={len(filtered_full)}"
            )
            if strict:
                raise RuntimeError(msg)
            print(f"WARNING: {msg}")
            return []
    else:
        raise RuntimeError(f"Unhandled source type: {spec.source}")

    if verbose:
        print(
            f"[{spec.folder_name}] loaded={len(loaded)} filtered={len(filtered)} "
            f"max_nodes={spec.max_nodes}"
        )

    if len(filtered) != expected_count:
        msg = (
            f"[{spec.folder_name}] filtered graph count mismatch. "
            f"expected={expected_count}, got={len(filtered)}"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
        return []

    if spec.folder_name.startswith("code2-"):
        return [code2_line(g) for g in filtered]
    return [scalar_line(g) for g in filtered]


def main() -> int:
    args = parse_args()
    precomputed_root = Path(args.precomputed_root)
    cache_root = Path(args.cache_root)

    if args.datasets:
        dataset_folders = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        dataset_folders = discover_dataset_folders(precomputed_root)

    failures: list[str] = []
    processed = 0

    for folder in dataset_folders:
        dataset_dir = precomputed_root / folder
        graphs_txt = dataset_dir / "graphs.txt"
        targets_txt = dataset_dir / "targets.txt"

        try:
            spec = parse_dataset_spec(folder)
            expected_count = expected_graph_count_from_graphs_txt(graphs_txt)
            lines = _dataset_lines(
                spec=spec,
                cache_root=cache_root,
                expected_count=expected_count,
                strict=args.strict,
                verbose=args.verbose,
                ged_mirror_folder_url=args.ged_mirror_folder_url,
                ged_mirror_mode=args.ged_mirror_mode,
            )
            if not lines and not args.strict:
                print(f"[{folder}] skipped (non-strict mode due to warnings).")
                continue

            write_targets_atomic(targets_txt, lines, overwrite=args.overwrite)
            processed += 1
            print(f"[{folder}] wrote {len(lines)} targets to {targets_txt}")
        except Exception as exc:
            failures.append(f"{folder}: {exc}")
            print(f"[{folder}] ERROR: {exc}", file=sys.stderr)
            if args.strict:
                break

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"\nDone. Processed {processed} dataset(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
