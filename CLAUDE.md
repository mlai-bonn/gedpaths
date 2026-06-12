# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GEDPaths computes graph edit distance (GED) mappings between graph pairs (via GEDLIB) and turns them into *edit paths* — sequences of intermediate graphs — which are exported to PyTorch Geometric format for GNN experiments. The heavy computation is C++; analysis, conversion, and visualization are Python.

## External Dependencies (required to build)

- The sibling repository `../libGraph` must be cloned next to this repo (CMake hardcodes `LIBGRAPH_ROOT ../libGraph`), with GEDLIB cloned into `libGraph/external/gedlib`.
- **GUROBI 12.0.3** is required (set `GUROBI_HOME`); the exact MIP-based GED methods (F1, F2, COMPACT_MIP) only exist when GEDLIB is compiled with GUROBI. See INSTALLATION.md.

## Build and Run

```bash
# Build C++ (from repo root)
mkdir build && cd build && cmake .. && make -j 6

# Full end-to-end pipeline for one or more datasets
./experiment.sh -db MUTAG                 # also: -db MUTAG,NCI1 -method F2 -recompile [threads]
./experiment.sh -db MUTAG -only_evaluation  # skip mapping/path computation
./experiment.sh -db MUTAG -only_python      # skip all C++ steps

# Python environment
bash python_src/install.sh   # creates venv/ and installs python_src/requirements.txt
```

Executables built by CMake (run from `build/`): `CreateMappings`, `CreatePaths`, `AnalyzePaths`, `AnalyzeMappings`, `ConvertPrecomputed`, `MappingStatistics`, `Test`.

## Pipeline Architecture

`experiment.sh` chains the whole flow per dataset; each stage reads the previous stage's output:

1. **`python_src/data_loader.py -db <DB>`** — downloads TU Dortmund datasets into `Data/Graphs/`.
2. **`CreateMappings`** (`create_edit_mappings.cpp` / `src/create_edit_mappings.h`) — computes GED node mappings for random graph pairs. Output: `Results/Mappings/<METHOD>/<DB>/<DB>_ged_mapping.{bin,csv}` + `graph_ids.txt`. Key args: `-method` (e.g. F2, REFINE), `-cost`, `-num_pairs`, `-t` threads, `-method_options` (e.g. `time-limit 180`).
3. **`AnalyzeMappings`** — validates/statistics on mappings.
4. **`CreatePaths`** (`create_edit_paths.cpp`) — builds edit path graphs from mappings, with composable `-path_strategy` options: `Random`, `InsertEdges`, `DeleteEdges`, optionally followed by `DeleteIsolatedNodes`. The shorthand strategy names used in result folders and Python scripts are `Rnd`, `Rnd_d-IsoN`, `i-E_d-IsoN`, `d-E_d-IsoN`. Output: `Results/Paths_<STRATEGY>/<METHOD>/<DB>/` (`.bin`/`.bgf` + `.csv`).
5. **`AnalyzePaths`** — per-strategy path statistics.
6. **`python_src/converter/bgf_to_pt.py`** — converts `.bgf` path graphs to PyTorch Geometric `.pt` datasets (`BGFInMemoryDataset` in `converter/GEDPathsInMemory.py`).
7. **`python_src/visualization/plot_edit_path_stats.py`**, **`python_src/wl_analysis.py`** — plots and Weisfeiler-Leman analysis.

Python scripts expect the repo root on `PYTHONPATH` (experiment.sh exports it).

## Code Layout Notes

- C++ entry points are the `.cpp` files at repo root; their implementations live in headers under `src/` (header-only style, e.g. `src/create_edit_paths.h`). `src/include.h` is the shared include.
- GED method names/options (and their GEDLIB defaults) are documented in `METHODS.md` (in German). MIP methods accept GUROBI options like `--time-limit`.
- `Data/`, `Results/`, `build/`, `_archive/` are generated/ignored — don't commit their contents. `_archive/` holds parked legacy data (old `Results_New/`, stray `data/`/`dataset/` download caches) and is not used by any code.
- Binary formats: `.bin` (mappings/paths, internal) and `.bgf` (graph format read by the Python converter; reader assumes little-endian, configurable `size_t` width of 8 or 4 bytes).
