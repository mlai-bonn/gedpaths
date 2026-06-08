import argparse
import os
import sys
from os.path import dirname


CANONICAL_PATH_STRATEGIES = [
    "Rnd",
    "Rnd_d-IsoN",
    "i-E_d-IsoN",
    "d-E_d-IsoN",
]
DEFAULT_STRATEGY = "i-E_d-IsoN"
DEFAULT_METHOD = "F2"
DEFAULT_DATABASE = "MUTAG"
DEFAULT_RESULTS_ROOT = "Results"


def get_all_datasets_for_strategy(results_root: str, method: str, strategy: str) -> list[str]:
    method_path = os.path.join(results_root, f"Paths_{strategy}", method)
    if not os.path.exists(method_path):
        raise FileNotFoundError(f"Edit path directory does not exist: {method_path}")
    if not os.path.isdir(method_path):
        raise NotADirectoryError(f"Edit path path is not a directory: {method_path}")

    datasets = [
        entry
        for entry in os.listdir(method_path)
        if os.path.isdir(os.path.join(method_path, entry))
    ]
    return sorted(datasets)


def convert_single_config(
    strategy: str,
    method: str = DEFAULT_METHOD,
    database: str = DEFAULT_DATABASE,
    bgf_path: str = "",
    results_root: str = DEFAULT_RESULTS_ROOT,
) -> int:
    """Convert a BGF file to a torch_geometric InMemoryDataset using the given strategy name.

    The script expects the BGF file to be at: Results/Paths_{strategy}/{method}/{database}/{database}_edit_paths.bgf
    """
    # Ensure the project root is importable as a package root so `python_src` can be found
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # import the dataset wrapper lazily (avoids importing heavy deps on --help)
    from python_src.converter.bgf_to_torch_geometric import BGFInMemoryDataset

    if bgf_path == "":
        bgf_path = os.path.join(
            results_root,
            f"Paths_{strategy}",
            method,
            database,
            f"{database}_edit_paths.bgf",
        )
        # Use the directory containing the bgf as the dataset root so the processed file
        # will be written to <root>/processed/data.pt
        root_dir = dirname(bgf_path) or "."
        processed_path = os.path.join(root_dir, "processed", "data.pt")

        # Check whether the converted .pt file already exists to avoid unnecessary processing
        if os.path.exists(processed_path):
            print(f"Processed dataset already exists at: {processed_path}")
            print("Skipping conversion.")
            return 0

        if not os.path.exists(bgf_path):
            print(f"Warning: bgf file not found at {bgf_path}. Skipping.")
            return 1

        print(f"Using strategy: {strategy}")
        print(f"Looking for BGF at: {bgf_path}")
    else:
        # add database name to bgf_path if not already present
        root_dir = os.path.join(bgf_path, database)
        os.makedirs(root_dir, exist_ok=True)
        bgf_path = os.path.join(bgf_path, f"{database}.bgf")

        if not os.path.exists(bgf_path):
            print(f"Warning: bgf file not found at {bgf_path}. Skipping.")
            return 1

        print(f"Using explicit BGF path: {bgf_path}")

    try:
        ds = BGFInMemoryDataset(root=root_dir, path=bgf_path)
    except Exception as exc:
        print(f"Error converting {bgf_path}: {exc}")
        return 1

    print(f"Processed dataset stored at: {ds.processed_paths[0]}")
    try:
        print(f"Dataset length (graphs): {len(ds)}")
    except Exception:
        # Fallback: if len() isn't available, try to infer from slices
        if hasattr(ds, "slices") and isinstance(ds.slices, dict):
            # slices typically contains 'x' or 'edge_index' keys mapping to tensors
            any_slice = next(iter(ds.slices.values()))
            # number of examples equals length of the first slice dimension
            print(f"Dataset length (graphs, inferred): {len(any_slice)}")
        else:
            print("Dataset length: unknown")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert BGF edit-path datasets to torch_geometric .pt files"
    )
    parser.add_argument(
        "-s",
        "--path_strategy",
        dest="strategy",
        default=DEFAULT_STRATEGY,
        help=(
            "Generating path strategy name used inside Results/Paths_{strategy}/. "
            f"Default: '{DEFAULT_STRATEGY}'."
        ),
    )
    parser.add_argument(
        "-all_path_strategies",
        action="store_true",
        help="Use Rnd, Rnd_d-IsoN, i-E_d-IsoN, d-E_d-IsoN",
    )
    parser.add_argument(
        "-d",
        "--db",
        dest="database",
        default=DEFAULT_DATABASE,
        help=(
            "Database name used inside Results/Paths_{strategy}/{method}/{database}/. "
            f"Default: '{DEFAULT_DATABASE}'."
        ),
    )
    parser.add_argument(
        "-all_db",
        action="store_true",
        help="Use all datasets found under Results/Paths_<path_strategy>/<method>/",
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        default=DEFAULT_METHOD,
        help=(
            "Method name used inside Results/Paths_{strategy}/{method}/{database}/. "
            f"Default: '{DEFAULT_METHOD}'."
        ),
    )
    parser.add_argument(
        "--results-root",
        dest="results_root",
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Results root containing Paths_<path_strategy>/ directories "
            f"(default: {DEFAULT_RESULTS_ROOT})"
        ),
    )
    parser.add_argument(
        "--bgf-path",
        dest="bgf_path",
        default="",
        help=(
            "Optional explicit path to the .bgf file directory "
            "(single-run only; overrides strategy/method/database)."
        ),
    )
    args = parser.parse_args()

    db_flags = {"-d", "--db", "-db"}
    strategy_flags = {"-s", "--path_strategy"}
    db_was_set = any(arg in db_flags for arg in sys.argv[1:])
    strategy_was_set = any(arg in strategy_flags for arg in sys.argv[1:])

    if args.all_db and db_was_set:
        parser.error("-d/--db cannot be combined with -all_db.")
    if args.all_path_strategies and strategy_was_set:
        parser.error("-s/--path_strategy cannot be combined with -all_path_strategies.")
    if args.bgf_path and (args.all_db or args.all_path_strategies):
        parser.error("--bgf-path cannot be combined with -all_db or -all_path_strategies.")

    strategies = CANONICAL_PATH_STRATEGIES if args.all_path_strategies else [args.strategy]

    attempted_runs = 0
    successes = 0
    failures = []

    for strategy in strategies:
        try:
            databases = (
                get_all_datasets_for_strategy(args.results_root, args.method, strategy)
                if args.all_db
                else [args.database]
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            print(f"Error: {exc}")
            return 1

        if not databases:
            print(
                f"No dataset directories found for strategy '{strategy}' "
                f"under {os.path.join(args.results_root, f'Paths_{strategy}', args.method)}"
            )
            return 1

        for database in databases:
            attempted_runs += 1
            print(f"\n== Converting: {args.method} / {strategy} / {database} ==")
            rc = convert_single_config(
                strategy=strategy,
                method=args.method,
                database=database,
                bgf_path=args.bgf_path,
                results_root=args.results_root,
            )
            if rc == 0:
                successes += 1
            else:
                failures.append((strategy, database, rc))

    if attempted_runs == 0:
        print("Error: no datasets or strategies were processed.")
        return 1

    print(f"\nConversion summary for method {args.method}:")
    print(f"  Successful runs: {successes}/{attempted_runs}")
    if failures:
        print("  Failed runs:")
        for strategy, database, rc in failures:
            print(f"    {strategy} / {database} (exit code {rc})")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
