#!/bin/bash
# This script is an end to end example of running all steps of an experiment.
# Usage: ./experiment.sh [-db DATASET] [-env VENV_PATH] [-recompile]

set -euo pipefail

# Defaults
DB_NAME="MUTAG"
VENV_PATH="venv"
RECOMPILE="no"
RECOMPILE_THREADS=""
ONLY_EVAL="no"
ONLY_PYTHON="no"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -db <dataset>        Dataset name (default: ${DB_NAME})
  -env <venv_path>     Path to virtual environment (default: ${VENV_PATH})
  -recompile [threads] Recompile the C++ code (optional threads). If threads omitted, defaults to half of available CPUs.
  -only_evaluation     Only run evaluation/analysis (don't compute mappings or paths)
  -only_python         Only run Python evaluation/visualization steps (skip C++ build and executables)
  -h, --help           Show this help message
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -db|--db)
      if [[ -z "${2:-}" || "${2:0:1}" == "-" ]]; then
        echo "Error: -db requires an argument"
        usage
        exit 1
      fi
      DB_NAME="$2"
      shift 2
      ;;
    -env|--env)
      if [[ -z "${2:-}" || "${2:0:1}" == "-" ]]; then
        echo "Error: -env requires an argument"
        usage
        exit 1
      fi
      VENV_PATH="$2"
      shift 2
      ;;
    -recompile|--recompile)
      RECOMPILE="recompile"
      # optional numeric argument: number of threads
      if [[ -n "${2:-}" && "${2:0:1}" != "-" ]]; then
        if ! [[ "${2}" =~ ^[0-9]+$ ]]; then
          echo "Error: -recompile expects an integer number of threads, got: ${2}"
          exit 1
        fi
        RECOMPILE_THREADS="$2"
        shift 2
      else
        shift
      fi
      ;;
    -only_evaluation|--only-evaluation)
      ONLY_EVAL="yes"
      shift
      ;;
    -only_python|--only-python)
      ONLY_PYTHON="yes"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

echo "Using dataset: ${DB_NAME}"
echo "Using virtual environment path: ${VENV_PATH}"

# Create or activate virtual environment
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  read -r -p "Virtual environment not found at ${VENV_PATH}. Create it? (y/n) " response
  if [[ "${response}" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Creating virtual environment using python_src/install.sh..."
    # Run the provided installer script which creates a venv in python_src/venv
    if [[ -f "python_src/install.sh" ]]; then
      # Ensure the installer is executable before running it
      chmod u+x python_src/install.sh
      bash python_src/install.sh
    else
      echo "Error: python_src/install.sh not found. Cannot create virtual environment."
      exit 1
    fi

    PY_SRC_VENV="$(pwd)/venv"
    if [[ ! -d "${PY_SRC_VENV}" ]]; then
      echo "Error: install script did not create ${PY_SRC_VENV}."
      exit 1
    fi

    # If the requested VENV_PATH differs from python_src/venv, create a symlink
    if [[ "${VENV_PATH}" != "venv" && "${VENV_PATH}" != "${PY_SRC_VENV}" ]]; then
      if [[ -e "${VENV_PATH}" ]]; then
        echo "Note: requested VENV_PATH ${VENV_PATH} already exists and will be used."
      else
        ln -s "${PY_SRC_VENV}" "${VENV_PATH}"
        echo "Created symlink ${VENV_PATH} -> ${PY_SRC_VENV}"
      fi
    fi

    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
  else
    echo "Please create a virtual environment at ${VENV_PATH} and install the required packages by running python_src/install.sh"
    exit 1
  fi
else
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

# Recompile C++ code if requested or if build dir missing (skip when ONLY_PYTHON)
if [[ "${ONLY_PYTHON}" != "yes" && ( "${RECOMPILE}" == "recompile" || ! -d "build" ) ]]; then
  echo "Recompiling the C++ code..."
  rm -rf build
  mkdir build
  # determine number of threads to use for make
  NUM_CPUS=$(nproc 2>/dev/null || echo 1)
  if [[ -n "${RECOMPILE_THREADS:-}" ]]; then
    THREADS="${RECOMPILE_THREADS}"
  else
    THREADS=$((NUM_CPUS/2))
    if [[ "${THREADS}" -lt 1 ]]; then
      THREADS=1
    fi
  fi
  echo "Using ${THREADS} build threads (nproc=${NUM_CPUS})"
  # run cmake/make inside the build dir but stay in repo root
  (cd build && cmake .. && make -j "${THREADS}")
fi

# Run pipeline

# download all datasets if not already present
python python_src/data_loader.py -db "${DB_NAME}"

# create the mappings for 5000 random pairs of graphs
if [[ "${ONLY_PYTHON}" == "yes" ]]; then
  echo "only_python: skipping CreateMappings"
elif [[ "${ONLY_EVAL}" == "yes" ]]; then
  echo "only_evaluation: skipping CreateMappings"
else
  if [[ -x "build/CreateMappings" ]]; then
    cd build || exit 1
    ./CreateMappings -db "${DB_NAME}" -num_pairs 5000 -method F2 -method_options threads 30 time-limit 10
    cd .. || exit 1
  else
    echo "Error: build/CreateMappings not found or not executable. Did the build succeed?"
    exit 1
  fi
fi

# analyze the created mappings
if [[ "${ONLY_PYTHON}" == "yes" ]]; then
  echo "only_python: skipping AnalyzeMappings"
else
  if [[ -x "build/AnalyzeMappings" ]]; then
    echo "Analyzing mappings..."
    cd build || exit 1
    ./AnalyzeMappings -db "${DB_NAME}" -method F2
    cd .. || exit 1
  else
    echo "Error: build/AnalyzeMappings not found or not executable. Did the build succeed?"
    exit 1
  fi
fi



# create the paths with different strategies
if [[ "${ONLY_PYTHON}" == "yes" ]]; then
  echo "only_python: skipping CreatePaths"
elif [[ "${ONLY_EVAL}" == "yes" ]]; then
  echo "only_evaluation: skipping CreatePaths"
else
  echo "Creating path graphs with different strategies..."
  if [[ -x "build/CreatePaths" ]]; then
    cd build || exit 1
    ./CreatePaths -db "${DB_NAME}" -method F2 -path_strategy Random
    ./CreatePaths -db "${DB_NAME}" -method F2 -path_strategy Random DeleteIsolatedNodes
    ./CreatePaths -db "${DB_NAME}" -method F2 -path_strategy InsertEdges DeleteIsolatedNodes
    ./CreatePaths -db "${DB_NAME}" -method F2 -path_strategy DeleteEdges DeleteIsolatedNodes
    cd .. || exit 1
  else
    echo "Error: build/CreatePaths not found or not executable. Did the build succeed?"
    exit 1
  fi
fi

# analyze the edit path graphs
if [[ "${ONLY_PYTHON}" == "yes" ]]; then
  echo "only_python: skipping AnalyzePaths"
else
  if [[ -x "build/AnalyzePaths" ]]; then
    echo "Analyzing path graphs..."
    cd build || exit 1
    ./AnalyzePaths -db "${DB_NAME}" -method F2 -path_strategy Rnd
    ./AnalyzePaths -db "${DB_NAME}" -method F2 -path_strategy Rnd_d-IsoN
    ./AnalyzePaths -db "${DB_NAME}" -method F2 -path_strategy i-E_d-IsoN
    ./AnalyzePaths -db "${DB_NAME}" -method F2 -path_strategy d-E_d-IsoN
    cd .. || exit 1
  else
    echo "Error: build/AnalyzePaths not found or not executable. Did the build succeed?"
    exit 1
  fi
fi


# Ensure repository root is on PYTHONPATH so scripts under python_src can import the package
# Use a safe expansion so `set -u` (nounset) does not fail when PYTHONPATH is unset
export PYTHONPATH="$(pwd)${PYTHONPATH:+:}${PYTHONPATH:-}"

# convert the generated path graphs to pytorch-geometric format
echo "Converting path graphs to pytorch-geometric format..."
python python_src/converter/bgf_to_pt.py --db "${DB_NAME}" --method F2 --path_strategy Rnd
python python_src/converter/bgf_to_pt.py --db "${DB_NAME}" --method F2 --path_strategy Rnd_d-IsoN
python python_src/converter/bgf_to_pt.py --db "${DB_NAME}" --method F2 --path_strategy i-E_d-IsoN
python python_src/converter/bgf_to_pt.py --db "${DB_NAME}" --method F2 --path_strategy d-E_d-IsoN



# plot the statistics with python
echo "Plotting statistics..."
python python_src/visualization/plot_edit_path_stats.py --db "${DB_NAME}" --method F2 --path_strategy Rnd
python python_src/visualization/plot_edit_path_stats.py --db "${DB_NAME}" --method F2 --path_strategy Rnd_d-IsoN
python python_src/visualization/plot_edit_path_stats.py --db "${DB_NAME}" --method F2 --path_strategy i-E_d-IsoN
python python_src/visualization/plot_edit_path_stats.py --db "${DB_NAME}" --method F2 --path_strategy d-E_d-IsoN

# do wl analysis of the dataset graphs
echo "Performing WL analysis of dataset graphs..."
python python_src/wl_analysis.py -db "${DB_NAME}" -s Rnd
python python_src/wl_analysis.py -db "${DB_NAME}" -s Rnd_d-IsoN
python python_src/wl_analysis.py -db "${DB_NAME}" -s i-E_d-IsoN
python python_src/wl_analysis.py -db "${DB_NAME}" -s d-E_d-IsoN


# plot a an example edit path
echo "Plotting example edit paths..."
python python_src/visualization/plot_edit_path.py --db "${DB_NAME}" --method F2 --path_strategy Rnd --start 3 --end 77
python python_src/visualization/plot_edit_path.py --db "${DB_NAME}" --method F2 --path_strategy Rnd_d-IsoN --start 3 --end 77
python python_src/visualization/plot_edit_path.py --db "${DB_NAME}" --method F2 --path_strategy i-E_d-IsoN --start 3 --end 77
python python_src/visualization/plot_edit_path.py --db "${DB_NAME}" --method F2 --path_strategy d-E_d-IsoN --start 3 --end 77



