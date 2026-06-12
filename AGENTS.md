  # Repository Guidelines

  ## Project Structure & Module Organization
  - Core C++ entry points live at the repository root (`create_edit_mappings.cpp`, `create_edit_paths.cpp`, `analyze_*.cpp`, `convert_precomputed.cpp`).
  - Shared C++ logic is under `src/` (header-heavy implementation files).
  - Python utilities for conversion, analysis, and plotting are in `python_src/` (`converter/`, `visualization/`).
  - Input/output data is organized as `Data/` (raw + processed) and `Results/` (mappings, paths, reports).
  - Specs and planning notes are in `specs/`; helper scripts are `experiment.sh`, `experiment_extended.sh`, and `test.sh`.

  ## Build, Test, and Development Commands
  - Configure and build C++:
    ```bash
    mkdir -p build && cd build && cmake .. && make -j6

  Builds executables such as CreateMappings, CreatePaths, AnalyzePaths, AnalyzeMappings, ConvertPrecomputed.

  - Run the end-to-end pipeline:

    ./experiment.sh -db MUTAG -recompile
  - Run extended multi-dataset experiments:

    ./experiment_extended.sh -db IMDB-16 -method Precomputed
  - Set up Python environment:

    bash python_src/install.sh

  ## Coding Style & Naming Conventions

  - C++: use C++20, 4-space indentation, and PascalCase for executable targets/functions already using that style; keep filenames in snake_case (for example, create_edit_paths.h).
  - Python: follow PEP 8 with snake_case for modules/functions.
  - Shell scripts should keep strict mode (set -euo pipefail) and long-option help text.
  - No formatter is enforced yet; match surrounding style and keep includes/imports minimal.

  ## Testing Guidelines

  - There is no full unit-test framework configured (no ctest/pytest suite).
  - Run all Python tests/checks with the repository virtual environment (`venv/bin/python` or after `source venv/bin/activate`).
  - Use smoke tests after changes:
      - Build and run build/Test (C++ load-path sanity check).
      - Run `venv/bin/python python_src/converter/test_load_pt.py` for conversion-path validation.
      - Run `venv/bin/python python_src/precomputed_targets.py --help` for precomputed target generation CLI sanity check.
      - Run a small dataset pipeline (./experiment.sh -db MUTAG -only_python) when touching Python/plotting.

  ## Commit & Pull Request Guidelines

  - Follow existing history style: concise imperative subject lines (for example, Enable compiler optimizations by default).
  - Optional automation prefix is acceptable when relevant (for example, auto-claude: ...).
  - PRs should include:
      - What changed and why.
      - Reproduction/validation commands executed.
      - Linked issue/spec (specs/...) when applicable.
      - Updated docs when CLI flags, paths, or outputs change.
