from os.path import dirname
import argparse


def main(strategy: str, method: str = "F2", database: str = "MUTAG", bgf_path: str = ""):
    """Convert a BGF file to a torch_geometric InMemoryDataset using the given strategy name.

    The script expects the BGF file to be at: Results/Paths_{strategy}/{method}/{database}/{database}_edit_paths.bgf
    """
    # Ensure the project root is importable as a package root so `python_src` can be found
    import os
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # import the dataset wrapper lazily (avoids importing heavy deps on --help)
    from python_src.converter.bgf_to_torch_geometric import BGFInMemoryDataset


    if bgf_path == "":
        bgf_path = f"Results/Paths_{strategy}/{method}/{database}/{database}_edit_paths.bgf"
        # Use the directory containing the bgf as the dataset root so the processed file
        # will be written to <root>/processed/data.pt
        root_dir = dirname(bgf_path) or "."
        print(f"Using strategy: {strategy}")
        print(f"Looking for BGF at: {bgf_path}")
    else:
        # add database name to bgf_path if not already present
        root_dir = bgf_path + "/" + database
        # if root_dir does not exist, create it
        os.makedirs(root_dir, exist_ok=True)
        bgf_path += f"{database}.bgf"

        print(f"Using explicit BGF path: {bgf_path}")



    ds = BGFInMemoryDataset(root=root_dir, path=bgf_path)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a BGF file to torch_geometric .pt using a generating-path strategy name"
    )
    parser.add_argument(
        "-s",
        "--path_strategy",
        dest="strategy",
        default="i-E_d-IsoN",
        help=(
            "Generating path strategy name used inside Results/Paths_{strategy}/. "
            "Default: 'Rnd_d-IsoN'."
        ),
    )
    parser.add_argument(
        "-d",
        "--db",
        dest="database",
        default="MUTAG",
        help=(
            "Database name used inside Results/Paths_{strategy}/F2/{database}/. "
            "Default: 'MUTAG'."
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        default="F2",
        help=(
            "Method name used inside Results/Paths_{strategy}/{method}/{database}/. "
            "Default: 'F2'."
        ),
    )
    parser.add_argument(
        "--bgf-path",
        dest="bgf_path",
        default="",
        help=(
            "Optional explicit path to the .bgf file (overrides strategy/method/database)."
        ),
    )
    args = parser.parse_args()
    main(args.strategy, args.method, args.database, args.bgf_path)
