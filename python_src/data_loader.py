#!/usr/bin/env python3
"""
Simple dataloader for https://chrsmrrs.github.io/datasets/docs/datasets/

Usage examples:
  python data_loader.py --list
  python data_loader.py -db MUTAG
  python data_loader.py --database MUTAG --dest Data/Graphs

This version downloads the .zip into a temporary folder and then extracts it into
the target graphs directory (default: Data/Graphs). The zip is removed after extraction.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import zipfile
import shutil
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

DATASETS_PAGE = "https://chrsmrrs.github.io/datasets/docs/datasets/"
DEFAULT_DEST = os.path.join("Data", "Graphs")


def fetch_page(url: str) -> str:
    req = Request(url, headers={"User-Agent": "python-data-loader/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except HTTPError as e:
        raise RuntimeError(f"HTTP error while fetching {url}: {e.code} {e.reason}")
    except URLError as e:
        raise RuntimeError(f"URL error while fetching {url}: {e}")


def extract_zip_links(html: str, base_url: str) -> list[str]:
    # Find href="...zip" or href='...zip'
    matches = re.findall(r"href=[\'\"]([^\'\"]+\.zip)\b", html, flags=re.IGNORECASE)
    urls = []
    for m in matches:
        full = urljoin(base_url, m)
        urls.append(full)
    # Deduplicate preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    return os.path.basename(path)


def list_available(zips: list[str]) -> None:
    if not zips:
        print("No .zip files found on the datasets page.")
        return
    print("Found the following .zip files on the datasets page:")
    for u in zips:
        print(" -", filename_from_url(u))


def download_with_progress(url: str, dest_path: str) -> None:
    req = Request(url, headers={"User-Agent": "python-data-loader/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            total = resp.getheader("Content-Length")
            if total is not None:
                try:
                    total = int(total)
                except ValueError:
                    total = None

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            chunk_size = 8192
            downloaded = 0
            with open(dest_path, "wb") as outf:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    outf.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 / total
                        print(f"\rDownloading {os.path.basename(dest_path)}: {downloaded}/{total} bytes ({pct:.1f}%)", end="", flush=True)
                if total:
                    print()
    except HTTPError as e:
        raise RuntimeError(f"HTTP error while downloading {url}: {e.code} {e.reason}")
    except URLError as e:
        raise RuntimeError(f"URL error while downloading {url}: {e}")


def find_exact_zip(zips: list[str], requested: str) -> str | None:
    # requested may be 'X' or 'X.zip' and should match filename exactly
    want = requested if requested.lower().endswith('.zip') else f"{requested}.zip"
    for u in zips:
        if filename_from_url(u) == want:
            return u
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a specific dataset .zip from chrsmrrs datasets listing")
    group = parser.add_mutually_exclusive_group(required=False)
    # Accept one or more dataset names
    group.add_argument("-db", "-dataset", "-data", "-database", dest="datasets", nargs='+', help="Name(s) of the dataset(s) to download (X or X.zip). Accepts multiple names separated by space.")
    parser.add_argument("--list", action="store_true", help="List available .zip files on the datasets page and exit")
    parser.add_argument("--dest", default=DEFAULT_DEST, help=f"Extraction directory. The zip is downloaded to a temporary folder and extracted into <dest>/<dataset> (default: {DEFAULT_DEST})")
    return parser.parse_args()


def _format_mb(bytes_val: int | None) -> str:
    if bytes_val is None:
        return "unknown"
    return f"{bytes_val / (1024*1024):.2f} MB"


def get_remote_size_bytes(url: str) -> int | None:
    """Try to obtain Content-Length via a HEAD request, fall back to a short GET if needed."""
    try:
        req = Request(url, headers={"User-Agent": "python-data-loader/1.0"}, method='HEAD')
        with urlopen(req, timeout=15) as resp:
            cl = resp.getheader('Content-Length')
            if cl:
                try:
                    return int(cl)
                except ValueError:
                    return None
    except Exception:
        # Fall back to GET but don't read the body
        try:
            req = Request(url, headers={"User-Agent": "python-data-loader/1.0"})
            with urlopen(req, timeout=15) as resp:
                cl = resp.getheader('Content-Length')
                if cl:
                    try:
                        return int(cl)
                    except ValueError:
                        return None
        except Exception:
            return None
    return None


def _move_contents(src_dir: str, dest_dir: str) -> None:
    """Move all contents of src_dir into dest_dir. Overwrite existing files/directories."""
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest):
            if os.path.isdir(dest) and not os.path.islink(dest):
                shutil.rmtree(dest)
            else:
                os.remove(dest)
        shutil.move(src, dest)


def main() -> int:
    args = parse_args()

    try:
        html = fetch_page(DATASETS_PAGE)
    except RuntimeError as e:
        print("Error:", e, file=sys.stderr)
        return 2

    zips = extract_zip_links(html, DATASETS_PAGE)

    if args.list or not args.datasets:
        list_available(zips)
        if not args.datasets:
            # If dataset not provided, listing only
            return 0

    if not args.datasets:
        print("No dataset specified. Use -db NAME or --list to see available datasets.")
        return 1

    extract_root = args.dest
    os.makedirs(extract_root, exist_ok=True)

    any_failures = False
    for requested in args.datasets:
        # Skip datasets that are already present locally (e.g. previously downloaded
        # or generated by python_src/synthetic_data_generator.py)
        local_name = requested[:-4] if requested.lower().endswith('.zip') else requested
        local_dir = os.path.join(extract_root, local_name)
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            print(f"Dataset '{local_name}' already present at {local_dir}, skipping download.")
            continue

        found = find_exact_zip(zips, requested)
        if not found:
            print(f"Requested dataset '{requested}' not found on the page.")
            # show similar names if available
            names = [filename_from_url(u) for u in zips]
            suggestions = [n for n in names if n.lower().startswith(requested.lower()) or requested.lower().startswith(n.lower().rstrip('.zip'))]
            if suggestions:
                print("Did you mean one of:")
                for s in suggestions:
                    print(" -", s)
            else:
                print("Use --list to see all available .zip files.")
            any_failures = True
            continue

        dataset_filename = filename_from_url(found)
        dataset_name = os.path.splitext(dataset_filename)[0]
        extract_dir = str(os.path.join(extract_root, dataset_name))

        size_bytes = get_remote_size_bytes(found)
        print(f"Preparing to download {dataset_filename} ({_format_mb(size_bytes)}) to temporary folder and extract to {extract_dir} ...")

        try:
            with tempfile.TemporaryDirectory(prefix="data_loader_") as td:
                temp_zip_path = os.path.join(td, dataset_filename)
                # Download into temp file
                download_with_progress(found, temp_zip_path)

                # Extract zip into a temporary extraction directory inside td
                ex_dir = str(os.path.join(td, "extracted"))
                os.makedirs(ex_dir, exist_ok=True)
                try:
                    with zipfile.ZipFile(temp_zip_path, 'r') as zf:
                        zf.extractall(path=ex_dir)
                except zipfile.BadZipFile as e:
                    print(f"Downloaded file is not a valid zip: {e}", file=sys.stderr)
                    any_failures = True
                    continue

                # After extraction, flatten nested top-level folders that repeat the dataset name.
                # Sometimes archives contain MUTAG/MUTAG/<files>. We want to extract so files end up in extract_dir
                os.makedirs(extract_dir, exist_ok=True)
                # Start at the extracted root and descend while there's exactly one directory named like the dataset
                current = str(ex_dir)
                while True:
                    items = [str(e) for e in os.listdir(current) if e not in ('.', '..')]
                    if len(items) == 1:
                        single = items[0]
                        single_path = os.path.join(current, single)
                        if os.path.isdir(single_path) and (single == dataset_name or single == dataset_filename.replace('.zip', '')):
                            # descend into the single directory and continue
                            current = single_path
                            continue
                    break

                # Move the final contents of `current` into the extract_dir
                _move_contents(str(current), str(extract_dir))

                # temp dir and file are cleaned up automatically
        except RuntimeError as e:
            print("Download failed:", e, file=sys.stderr)
            any_failures = True
            continue
        except OSError as e:
            print("Filesystem error:", e, file=sys.stderr)
            any_failures = True
            continue

        print(f"Extraction finished for {dataset_name}.")

    return 1 if any_failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
