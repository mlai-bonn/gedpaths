# GEDPaths
Repo that builds up on the libgraph and uses the GEDLIB to create edit paths between two pairs of graphs from different sources.

## Installation
See [INSTALLATION.md](INSTALLATION.md) for all dependencies and detailed setup instructions.


## Run experiments

Use the provided experiment.sh script to run all experiments:
```bash
chmod u+x experiment.sh
./experiment.sh -db MUTAG
```

## Usage



### 1. Compute Mappings
- **Download a dataset** from [TUDortmund](https://chrsmrrs.github.io/datasets/) or use your own graphs in the same format in the `Data/Graphs/` folder.

#### Build the project
```bash
mkdir build
cd build
cmake ..
make -j 6
```

#### Run the mapping computation
```bash
./create_edit_mappings \
  -db MUTAG \
  -raw ../Data/Graphs/ \
  -processed ../Data/ProcessedGraphs/ \
  -mappings ../Results/Mappings/ \
  -t 30 \
  -method F2 \
  -cost CONSTANT \
  -seed 42 \
  -num_graphs 5000
```


**Main arguments:**
  - `-db <database name>`: Name of the dataset (e.g., MUTAG)
  - `-raw <raw data path>`: Path to raw graph data
  - `-processed <processed data path>`: Path to store processed graphs
  - `-mappings <output path>`: Path to store mappings (now in `Results/Mappings/`)
  - `-t <threads>`: Number of threads
  - `-method <method>`: GED method (e.g., REFINE, F2)
  - `-cost <cost>`: Edit cost type (e.g., CONSTANT)
  - `-seed <seed>`: Random seed
  - `-num_graphs <N>`: Number of graph pairs (optional)

**Output files:**
- After running, you will find the following files in `../Results/Mappings/<METHOD>/<DB>/`:
    - `<DB>_ged_mapping.bin`: Binary file containing the computed graph edit distance mappings (used for further processing).
    - `<DB>_ged_mapping.csv`: CSV file with meta information in a human-readable format (for inspection, analysis, or use in other tools).
    - `graph_ids.txt`: The list of graph pairs for which mappings were computed.


### 2. Compute Edit Paths

#### Build the project (if not already built)
```bash
mkdir build
cd build
cmake ..
make -j 6
```

#### Run the edit path computation
```bash
./create_edit_paths \
  -db MUTAG \
  -processed ../Data/ProcessedGraphs/ \
  -mappings ../Results/Mappings/REFINE/MUTAG/ \
  -num_mappings 1000 \
  -t 1
```

**Output files:**
- After running, you will find the following files in `../Results/Paths/<METHOD>/<DB>/`:
    - `<DB>_edit_paths.bin`: Binary file containing the computed edit paths (used for further processing).
    - `<DB>_edit_paths.csv`: CSV file with edit path information in a human-readable format (for inspection, analysis, or use in other tools).

**Main arguments:**
  - `-db <database name>`: Name of the dataset
  - `-processed <processed data path>`: Path to processed graphs
  - `-mappings <mappings path>`: Path to mappings (now in `Results/Mappings/REFINE/<DB>/`)
  - `-t <threads>`: Number of threads

### 3. Export to PyTorch Geometric Format
(Instructions for this step can be added here if needed.)

---

## For the exact solvers (e.g., F1, F2)
You need GUROBI 12.0.3 installed and properly configured. See [GUROBI 12.0.3 Installation](INSTALLATION.md#install-gurobi-1203-linux).

---
