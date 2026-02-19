#!/usr/bin/env python3
"""
Visualization for edit path statistics CSVs and per-path positions CSVs.

Usage:
  python3 plot_edit_path_stats.py /path/to/Results/Paths/F2/MUTAG/Evaluation --save --show

This script will:
 - read all .csv files in the provided directory
 - for ordinary statistic CSVs (single-column with header 'value') it creates a line plot and a histogram (existing behavior)
 - for per-path positions CSVs (files that end with '_Positions.csv' or contain 'Positions' in the filename), it will:
    - parse each row's comma-separated integer positions
    - create a counts-per-position summary CSV: <basename>_counts.csv with columns 'position,count'
    - plot counts per position (line) and save as <basename>_counts.png
    - create a heatmap (paths x positions) showing presence (1) or absence (0) and save as <basename>_heatmap.png

Dependencies: pandas, matplotlib, seaborn (seaborn optional but improves heatmap)
Install: pip3 install pandas matplotlib seaborn
"""

import argparse
import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib
import math

# module-level path for Python outputs; set in main()
python_output_dir = None

def _should_write(path: str) -> bool:
    """Return True if we should create/overwrite an output file."""
    return not os.path.exists(path)

def py_out(filename: str) -> str:
    """Return an output path under Evaluation_Python (created in main())."""
    base = python_output_dir or os.getcwd()
    return os.path.join(base, filename)


try:
    sns = importlib.import_module('seaborn')
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False


def plot_csv_file(csv_path: str, save: bool = True, show: bool = False):
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    dirpath = os.path.dirname(csv_path)

    # Positions files detection
    if 'Positions' in basename:
        ret = _process_positions_file(csv_path, dirpath, basename, save=save, show=show)
        # ret is either None or (counts_df, mat)
        if ret is not None:
            counts_df, mat = ret
            return (basename, counts_df, mat)
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return

    if 'value' not in df.columns:
        print(f"CSV {csv_path} does not contain a 'value' column. Columns: {list(df.columns)}")
        return

    values = df['value'].dropna().astype(float)
    if values.empty:
        print(f"CSV {csv_path} has no numeric values to plot.")
        return

    plt.figure(figsize=(12, 5))

    # Line plot
    plt.subplot(1, 2, 1)
    plt.plot(values.values, marker='o', linestyle='-')
    plt.title(f"{basename} - line")
    plt.xlabel('index')
    plt.ylabel('value')

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(values.values, bins=min(50, max(5, len(values)//2)))
    plt.title(f"{basename} - histogram")
    plt.xlabel('value')
    plt.ylabel('count')

    plt.tight_layout()

    outpath = os.path.join(dirpath, basename + '.png')
    if save:
        try:
            outpath = py_out(basename + '.png')
            if _should_write(outpath):
                plt.savefig(outpath)
                print(f"Saved plot to {outpath}")
                # also write a data-embedded TeX (line + histogram)
                tex_path = outpath.replace('.png', '.tex')
                if _should_write(tex_path):
                    write_tex_line_hist(tex_path, values.values, title=f"Plot: {basename}")
            else:
                print(f"Skipping existing plot: {outpath}")
        except Exception as e:
            print(f"Failed to save plot for {csv_path}: {e}")

    if show:
        plt.show()
    else:
        plt.close()


def _parse_positions_column(series: pd.Series):
    """Parse a pandas Series of strings (or NaN) where each entry is comma-separated integers.
    Returns a list of lists of ints.
    """
    positions = []
    for val in series.fillna(''):
        if isinstance(val, (int, float)):
            # single numeric value (unlikely), treat as single position if integer
            try:
                iv = int(val)
                positions.append([iv])
            except Exception:
                positions.append([])
            continue
        s = str(val).strip()
        if s == '':
            positions.append([])
            continue
        parts = [p.strip() for p in s.split(',') if p.strip() != '']
        row = []
        for p in parts:
            try:
                row.append(int(p))
            except Exception:
                # ignore non-integer token but warn
                print(f"Warning: ignoring non-integer token '{p}' in positions CSV")
        positions.append(row)
    return positions


def _process_positions_file(csv_path: str, dirpath: str, basename: str, save: bool = True, show: bool = False):
    # Read positions CSV robustly: prefer reading raw file lines (each line is one path's positions).
    try:
        with open(csv_path, 'r') as f:
            lines = [l.rstrip('\n') for l in f]
    except Exception as e:
        print(f"Failed to open positions CSV {csv_path}: {e}")
        # fallback: try pandas default reader
        try:
            df = pd.read_csv(csv_path)
        except Exception as e2:
            print(f"Failed to read positions CSV {csv_path} with pandas fallback: {e2}")
            return None
        colname = df.columns[0]
        series = df[colname].astype(str)
    else:
        # remove possible header line 'positions'
        if lines and lines[0].strip().lower() == 'positions':
            lines = lines[1:]
        series = pd.Series([l for l in lines])

    positions = _parse_positions_column(series)
    if not positions:
        print(f"No position data found in {csv_path}")
        return None

    # determine max position to size arrays
    max_pos = -1
    for row in positions:
        if row:
            max_row = max(row)
            if max_row > max_pos:
                max_pos = max_row
    if max_pos < 0:
        print(f"Positions CSV {csv_path} has no numeric positions")
        return None

    n_paths = len(positions)
    width = max_pos + 1
    # build presence matrix
    mat = np.zeros((n_paths, width), dtype=int)
    for i, row in enumerate(positions):
        for p in row:
            if 0 <= p < width:
                mat[i, p] = 1

    # counts per position
    counts = np.sum(mat, axis=0)
    counts_df = pd.DataFrame({'position': np.arange(width), 'count': counts})
    counts_csv = py_out(basename + '_counts.csv')
    if _should_write(counts_csv):
        counts_df.to_csv(counts_csv, index=False)
        print(f"Wrote position counts CSV: {counts_csv}")
    else:
        print(f"Skipping existing counts CSV: {counts_csv}")

    # plot counts line
    plt.figure(figsize=(10,4))
    plt.plot(counts_df['position'], counts_df['count'], marker='o')
    plt.title(f"{basename} - counts per position")
    plt.xlabel('position')
    plt.ylabel('count')
    plt.grid(True)
    out_counts_png = py_out(basename + '_counts.png')
    if save:
        if _should_write(out_counts_png):
            plt.savefig(out_counts_png)
            print(f"Saved counts plot: {out_counts_png}")
            tex_path = out_counts_png.replace('.png', '.tex')
            if _should_write(tex_path):
                write_tex_positions_counts(tex_path, counts_df['position'].values, counts_df['count'].values, title=f"Counts per position for {basename}")
        else:
            print(f"Skipping existing counts plot: {out_counts_png}")

    # heatmap (may be large) - compute integer figure size to avoid float warnings
    heatmap_w = min(20, max(6, width // 2))
    heatmap_h = min(20, max(4, max(1, n_paths // 10)))
    plt.figure(figsize=(int(heatmap_w), int(heatmap_h)))
    if _HAS_SEABORN:
        sns.heatmap(mat, cbar=True)
    else:
        plt.imshow(mat, aspect='auto', interpolation='nearest', cmap='Greys')
        plt.colorbar()
    plt.title(f"{basename} - heatmap (paths x positions)")
    plt.xlabel('position')
    plt.ylabel('path index')
    out_heatmap = py_out(basename + '_heatmap.png')
    if save:
        if _should_write(out_heatmap):
            plt.savefig(out_heatmap)
            print(f"Saved heatmap: {out_heatmap}")
            tex_path = out_heatmap.replace('.png', '.tex')
            if _should_write(tex_path):
                write_tex_positions_heatmap(tex_path, mat, title=f"Heatmap for {basename}")
        else:
            print(f"Skipping existing heatmap: {out_heatmap}")
    if show:
        plt.show()
    else:
        plt.close()

    # return the counts DataFrame and the presence matrix (paths x positions)
    return (counts_df, mat)


def bucket_combined_counts_df(combined_df: pd.DataFrame, n_buckets: int = 10) -> pd.DataFrame:
    """Given a combined per-position DataFrame with a 'position' column and operation columns,
    aggregate counts into n_buckets across the full position range.
    Returns a DataFrame with columns: 'bucket' (0..n_buckets-1) and one column per operation with summed counts.
    """
    # determine global max position
    max_pos = int(combined_df['position'].max())
    total_positions = max_pos + 1
    bucket_size = math.ceil(total_positions / n_buckets)

    buckets = list(range(n_buckets))
    bucket_df = pd.DataFrame({'bucket': buckets})
    # for each operation column, sum counts falling into each bucket
    op_cols = [c for c in combined_df.columns if c != 'position']
    for op in op_cols:
        sums = []
        for b in buckets:
            start = b * bucket_size
            end = min((b + 1) * bucket_size - 1, max_pos)
            mask = (combined_df['position'] >= start) & (combined_df['position'] <= end)
            s = int(combined_df.loc[mask, op].sum())
            sums.append(s)
        bucket_df[op] = sums
    return bucket_df


def plot_buckets_stacked(bucket_df: pd.DataFrame, out_png: str, normalize: bool = False, save: bool = True, show: bool = False):
    """Plot a stacked bar chart from bucket_df (bucket column + operation columns).
    If normalize=True, convert counts to proportions per bucket.
    """
    df = bucket_df.copy()
    op_cols = [c for c in df.columns if c != 'bucket']
    if normalize:
        # compute row-wise sum and divide
        row_sum = df[op_cols].sum(axis=1).replace(0, 1)
        for c in op_cols:
            df[c] = df[c] / row_sum

    # stacked bar
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(df))
    x = df['bucket'].values
    for c in op_cols:
        ax.bar(x, df[c].values, bottom=bottom, label=c)
        bottom = bottom + df[c].values
    ax.set_xlabel('bucket (path segment)')
    ax.set_ylabel('proportion' if normalize else 'count')
    ax.set_title(('Normalized ' if normalize else '') + 'Operation distribution across path buckets')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save:
        if _should_write(out_png):
            plt.savefig(out_png)
            print(f"Saved buckets plot: {out_png}")
            # write data-embedded TeX file for buckets plot
            tex_path = out_png.replace('.png', '.tex')
            if _should_write(tex_path):
                write_tex_stacked_from_df(tex_path, bucket_df, 'bucket', op_cols, 'bucket (path segment)', 'proportion' if normalize else 'count', ('Normalized ' if normalize else '') + 'Operation distribution across path buckets')
        else:
            print(f"Skipping existing buckets plot: {out_png}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_positions_bars(combined_df: pd.DataFrame, out_png: str, normalize: bool = False, save: bool = True, show: bool = False):
    """Create a stacked bar plot across positions (x-axis = position) using the operation columns.

    This produces a stacked bar for each position where each stack component is one operation (column).
    If `normalize=True` each position's stacks are converted to proportions (summing to 1) to highlight
    relative distribution per position.
    """
    df = combined_df.copy()
    if 'position' not in df.columns:
        print("Combined DataFrame does not contain a 'position' column; cannot make positions bar plot.")
        return

    op_cols = [c for c in df.columns if c != 'position']
    if not op_cols:
        print('No operation columns to plot in positions bar plot.')
        return

    x = df['position'].values
    arr = df[op_cols].values.astype(float)

    if normalize:
        # normalize per position (row)
        row_sum = arr.sum(axis=1)
        # avoid division by zero
        row_sum[row_sum == 0] = 1.0
        arr = arr / row_sum[:, None]

    # choose figure width based on number of positions (avoid extremely wide figures)
    n_pos = len(x)
    fig_w = int(max(8, min(0.25 * n_pos, 40)))
    fig_h = 6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bottom = np.zeros(n_pos)
    # draw bars stacked
    for idx, col in enumerate(op_cols):
        ax.bar(x, arr[:, idx], bottom=bottom, label=col, width=1.0)
        bottom = bottom + arr[:, idx]

    ax.set_xlabel('position')
    ax.set_ylabel('proportion' if normalize else 'count')
    ax.set_title(('Normalized ' if normalize else '') + 'Operation counts per position (stacked)')
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Put legend to the right if there are multiple operation columns
    if len(op_cols) > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save:
        try:
            if _should_write(out_png):
                plt.savefig(out_png)
                print(f"Saved positions stacked bar plot: {out_png}")
                # write data-embedded TeX file for positions bar plot
                tex_path = out_png.replace('.png', '.tex')
                if _should_write(tex_path):
                    write_tex_stacked_from_df(tex_path, df, 'position', op_cols, 'position', 'count', 'Operation counts per position (stacked)')
            else:
                print(f"Skipping existing positions plot: {out_png}")
        except Exception as e:
            print(f"Failed to save positions bar plot {out_png}: {e}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_nodes_edges_per_position(mat_all: np.ndarray, nodes_vals: np.ndarray, edges_vals: np.ndarray, out_prefix: str, save: bool = True, show: bool = False):
    """Create stacked bar plots per position with two stacks: total nodes and total edges.

    mat_all: (n_paths x width) binary matrix indicating presence of any operation at (path,position)
    nodes_vals, edges_vals: arrays of length n_paths with the number of nodes/edges for each path
    out_prefix: prefix filename (path) where to store PNGs (absolute and normalized)
    """
    if mat_all is None or mat_all.size == 0:
        print('No position matrix provided for nodes/edges plotting.')
        return
    n_paths, width = mat_all.shape
    # ensure vectors length
    n = min(n_paths, len(nodes_vals), len(edges_vals))
    if n_paths != len(nodes_vals) or n_paths != len(edges_vals):
        print(f'Warning: matrix paths={n_paths}, nodes_vals={len(nodes_vals)}, edges_vals={len(edges_vals)}; using min={n}')

    # Trim to n
    mat = mat_all[:n, :]
    nodes = nodes_vals[:n]
    edges = edges_vals[:n]

    # sum per position
    nodes_sum = mat.T.dot(nodes)
    edges_sum = mat.T.dot(edges)

    positions = np.arange(width)
    df = pd.DataFrame({'position': positions, 'nodes': nodes_sum, 'edges': edges_sum})

    # absolute stacked
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df['position'], df['nodes'], label='nodes')
    ax.bar(df['position'], df['edges'], bottom=df['nodes'], label='edges')
    ax.set_xlabel('position')
    ax.set_ylabel('count (sum across paths)')
    ax.set_title('Sum of nodes and edges across paths per position (stacked)')
    ax.legend()
    plt.tight_layout()
    out_abs = out_prefix + '_nodes_edges_absolute.png'
    if save:
        if _should_write(out_abs):
            plt.savefig(out_abs)
            print(f'Saved nodes/edges per-position absolute plot: {out_abs}')
            # write data-embedded TeX file for nodes/edges absolute plot
            tex_path = out_abs.replace('.png', '.tex')
            if _should_write(tex_path):
                write_tex_nodes_edges_data(tex_path, positions, nodes_sum, edges_sum, caption='Sum of nodes and edges across paths per position (stacked)', normalized=False)
        else:
            print(f"Skipping existing nodes/edges absolute plot: {out_abs}")
    if show:
        plt.show()
    else:
        plt.close()

    # normalized per-position
    total = df['nodes'] + df['edges']
    total[total == 0] = 1.0
    nodes_prop = df['nodes'] / total
    edges_prop = df['edges'] / total
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df['position'], nodes_prop, label='nodes')
    ax.bar(df['position'], edges_prop, bottom=nodes_prop, label='edges')
    ax.set_xlabel('position')
    ax.set_ylabel('proportion')
    ax.set_title('Proportion of nodes vs edges across paths per position (stacked)')
    ax.legend()
    plt.tight_layout()
    out_norm = out_prefix + '_nodes_edges_normalized.png'
    if save:
        if _should_write(out_norm):
            plt.savefig(out_norm)
            print(f'Saved nodes/edges per-position normalized plot: {out_norm}')
            # write data-embedded TeX file for nodes/edges normalized plot
            tex_path = out_norm.replace('.png', '.tex')
            if _should_write(tex_path):
                write_tex_nodes_edges_data(tex_path, positions, nodes_sum, edges_sum, caption='Proportion of nodes vs edges across paths per position (stacked)', normalized=True)
        else:
            print(f"Skipping existing nodes/edges normalized plot: {out_norm}")
    if show:
        plt.show()
    else:
        plt.close()


def write_tex_line_hist(tex_path: str, values: np.ndarray, title: str = None):
    """Write a standalone TeX (PGFPlots) that contains a line plot and histogram using embedded data."""
    try:
        vals = np.asarray(values).astype(float)
        n = len(vals)
        bins = min(50, max(5, n // 2))
        hist_counts, bin_edges = np.histogram(vals, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        with open(tex_path, 'w') as f:
            f.write('\\documentclass{standalone}\n')
            f.write('\\usepackage{pgfplots}\n')
            f.write('\\pgfplotsset{compat=1.18}\n')
            f.write('\\begin{document}\n')
            f.write('\\begin{tikzpicture}\n')
            # left: line plot (50% width)
            f.write('  \\begin{axis}[name=plotA, width=0.55\\textwidth, height=0.45\\textwidth, xlabel={index}, ylabel={value}]\\n')
            f.write('  \\addplot table[row sep=\\\\]{\\n')
            for i, v in enumerate(vals):
                f.write(f"{i} {float(v)}\\\\\n")
            f.write('  };\\n')
            f.write('  \\end{axis}\\n')
            # right: histogram
            f.write('  \\begin{axis}[at={(plotA.east)}, anchor=west, xshift=1em, width=0.35\\textwidth, height=0.45\\textwidth, xlabel={value}, ylabel={count}]\\n')
            f.write('  \\addplot[ybar] table[row sep=\\\\]{\\n')
            for c, bc in zip(hist_counts, bin_centers):
                f.write(f"{float(bc)} {int(c)}\\\\\n")
            f.write('  };\\n')
            f.write('  \\end{axis}\\n')
            if title:
                safe = title.replace('%', '%%')
                f.write('% ' + safe + '\n')
            f.write('\\end{tikzpicture}\\n')
            f.write('\\end{document}\\n')
        print(f"Wrote line+hist TeX (data embedded): {tex_path}")
    except Exception as e:
        print(f"Failed to write line+hist TeX {tex_path}: {e}")


def write_tex_positions_counts(tex_path: str, positions: np.ndarray, counts: np.ndarray, title: str = None):
    try:
        with open(tex_path, 'w') as f:
            f.write('\\documentclass{standalone}\n')
            f.write('\\usepackage{pgfplots}\n')
            f.write('\\pgfplotsset{compat=1.18}\n')
            f.write('\\begin{document}\n')
            f.write('\\begin{tikzpicture}\n')
            f.write('  \\begin{axis}[width=\\textwidth, height=0.45\\textwidth, xlabel={position}, ylabel={count}]\\n')
            f.write('  \\addplot table[row sep=\\\\]{\\n')
            for p, c in zip(positions, counts):
                f.write(f"{int(p)} {int(c)}\\\\\n")
            f.write('  };\\n')
            f.write('  \\end{axis}\\n')
            if title:
                f.write('% ' + title.replace('%', '%%') + '\n')
            f.write('\\end{tikzpicture}\\n')
            f.write('\\end{document}\\n')
        print(f"Wrote positions counts TeX (data embedded): {tex_path}")
    except Exception as e:
        print(f"Failed writing positions counts TeX {tex_path}: {e}")


def write_tex_positions_heatmap(tex_path: str, mat: np.ndarray, title: str = None):
    try:
        nrows, ncols = mat.shape
        with open(tex_path, 'w') as f:
            f.write('\\documentclass{standalone}\n')
            f.write('\\usepackage{pgfplots}\n')
            f.write('\\pgfplotsset{compat=1.18}\n')
            f.write('\\begin{document}\n')
            f.write('\\begin{tikzpicture}\n')
            f.write('  \\begin{axis}[width=\\textwidth, height=0.6\\textwidth, xlabel={position}, ylabel={path index}, colormap/viridis]\\n')
            f.write('  \\addplot [matrix plot*, point meta=explicit] table[row sep=\\\\]{\\n')
            f.write('x y value\\\\\n')
            for i in range(nrows):
                for j in range(ncols):
                    f.write(f"{j} {i} {int(mat[i, j])}\\\\\n")
            f.write('  };\\n')
            f.write('  \\end{axis}\\n')
            if title:
                f.write('% ' + title.replace('%', '%%') + '\n')
            f.write('\\end{tikzpicture}\\n')
            f.write('\\end{document}\\n')
        print(f"Wrote positions heatmap TeX (data embedded): {tex_path}")
    except Exception as e:
        print(f"Failed writing positions heatmap TeX {tex_path}: {e}")


def write_tex_nodes_edges_data(tex_path: str, positions: np.ndarray, nodes_sum: np.ndarray, edges_sum: np.ndarray, caption: str = None, normalized: bool = False):
    """Write a standalone LaTeX file that uses PGFPlots to draw a stacked bar chart from the provided arrays.

    The generated .tex embeds the numeric data as a literal table so no external images are required.
    tex_path: absolute path to write (will end with .tex)
    positions, nodes_sum, edges_sum: 1D numpy arrays of equal length
    normalized: if True, the values are treated as proportions (expected between 0 and 1)
    """
    try:
        dirname = os.path.dirname(tex_path)
        os.makedirs(dirname, exist_ok=True)
        with open(tex_path, 'w') as f:
            f.write('\\documentclass{standalone}\n')
            f.write('\\usepackage{pgfplots}\n')
            f.write('\\pgfplotsset{compat=1.18}\n')
            f.write('\\usepackage{siunitx}\n')
            f.write('\\begin{document}\n')
            f.write('\\begin{tikzpicture}\n')
            f.write('  \\begin{axis}[ybar stacked, bar width=0.8, width=\\textwidth, height=0.5\\textwidth, xlabel={position}, ylabel={' + ('proportion' if normalized else 'count') + '}, legend pos=outer north east, enlarge x limits=0.02, xtick=data]\n')
            f.write('  % data: position nodes edges\n')
            f.write('  \\pgfplotstableread{\n')
            # write table header
            f.write('position nodes edges\n')
            for p, n, e in zip(positions, nodes_sum, edges_sum):
                # format numeric values (use float for safety)
                f.write(f"{int(p)} {float(n)} {float(e)}\n")
            f.write('  }\\loadedtable\n')
            f.write('  \\addplot table[x=position,y=nodes]{\\loadedtable};\n')
            f.write('  \\addplot table[x=position,y=edges]{\\loadedtable};\n')
            f.write('  \\legend{nodes,edges}\n')
            f.write('  \\end{axis}\n')
            if caption:
                safe_caption = caption.replace('%', '%%')
                f.write('  % ' + safe_caption + '\n')
            f.write('\\end{tikzpicture}\n')
            f.write('\\end{document}\n')
        print(f"Wrote nodes/edges TeX (data embedded): {tex_path}")
    except Exception as e:
        print(f"Failed to write nodes/edges TeX {tex_path}: {e}")


def write_tex_stacked_from_df(tex_path: str, df: pd.DataFrame, x_col: str, stack_cols: list, xlabel: str, ylabel: str, title: str, normalized: bool = False):
    """Write a standalone LaTeX (PGFPlots) file that draws a stacked bar chart from df.

    df: DataFrame containing x_col and the stack_cols to stack
    tex_path: output .tex file path
    """
    try:
        dirname = os.path.dirname(tex_path)
        os.makedirs(dirname, exist_ok=True)
        with open(tex_path, 'w') as f:
            f.write('\\documentclass{standalone}\n')
            f.write('\\usepackage{pgfplots}\n')
            f.write('\\pgfplotsset{compat=1.18}\n')
            f.write('\\begin{document}\n')
            f.write('\\begin{tikzpicture}\n')
            ylabel_safe = ylabel.replace('%', '%%')
            f.write('  \\begin{axis}[ybar stacked, bar width=0.8, width=\\textwidth, height=0.5\\textwidth, xlabel={' + xlabel + '}, ylabel={' + ylabel_safe + '}, legend pos=outer north east, enlarge x limits=0.02, xtick=data]\n')
            # write table header using sanitized column names (replace '_' -> '-') to avoid catcode issues
            sanitized_cols = [c.replace('_', '-') for c in stack_cols]
            header = ' '.join([x_col] + sanitized_cols)
            f.write('  \\pgfplotstableread{\n')
            f.write(header + '\n')
            for _, row in df.iterrows():
                vals = [str(int(row[x_col]))]
                for c in stack_cols:
                    val = row[c]
                    if pd.isna(val):
                        vals.append('0')
                    else:
                        vals.append(str(float(val)))
                f.write(' '.join(vals) + '\n')
            f.write('  }\\loadedtable\n')
            # addplots using sanitized column names
            for orig, san in zip(stack_cols, sanitized_cols):
                f.write('  \\addplot table[x=' + x_col + ',y=' + san + ']{\\loadedtable};\n')
            # legend (escape underscores for display)
            legend_items_escaped = ','.join([c.replace('_', '\\_') for c in stack_cols])
            f.write('  \\legend{' + legend_items_escaped + '}\n')
            f.write('  \\end{axis}\n')
            if title:
                safe_title = title.replace('%', '%%')
                f.write('  % ' + safe_title + '\n')
            f.write('\\end{tikzpicture}\n')
            f.write('\\end{document}\n')
        print(f"Wrote stacked-bar TeX (data embedded): {tex_path}")
    except Exception as e:
        print(f"Failed to write stacked-bar TeX {tex_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Plot edit path statistics CSVs')
    # default directory used when user doesn't provide one explicitly
    default_dir = 'Results/Paths/F2/MUTAG/Evaluation'
    parser.add_argument('directory', nargs='?', default=default_dir,
                        help='Directory containing CSV files (Evaluation folder). Defaults to %(default)s')

    # mirror bgf_to_pt.py args so users can pass -s/-m/-d like the converter
    parser.add_argument('-s', '--path_strategy', dest='strategy', default='Rnd_d-IsoN',
                        help=(
                            "Generating path strategy name used inside Results/Paths_{strategy}/. "
                            "Default: 'Rnd_d-IsoN'."
                        ))
    parser.add_argument('-d', '--db', dest='database', default='MUTAG',
                        help=(
                            "Database name used inside Results/Paths_{strategy}/<method>/{database}/. "
                            "Default: 'MUTAG'."
                        ))
    parser.add_argument('-m', '--method', dest='method', default='F2',
                        help=(
                            "Method name used inside Results/Paths_{strategy}/{method}/{database}/. "
                            "Default: 'F2'."
                        ))

    parser.add_argument('--save', dest='save', action='store_true', help='Save plots as PNGs (default)')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do not save plots')
    parser.set_defaults(save=True)
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    parser.add_argument('--buckets', type=int, default=10, help='Number of buckets to aggregate positions into for bucketed plots (default: 10)')
    args = parser.parse_args()

    # If the user left the directory positional argument as the default, construct it from the provided strategy/method/database
    if args.directory == default_dir:
        directory = os.path.join('Results', f'Paths_{args.strategy}', args.method, args.database, 'Evaluation')
    else:
        directory = args.directory
        # If the user provided a directory that still contains 'Paths' and also provided a strategy,
        # replace the generic 'Paths' with the strategy-specific folder to keep backward compatibility.
        if args.strategy and 'Paths' in directory and 'Paths_' not in directory:
            directory = directory.replace('Paths', f'Paths_{args.strategy}')

    # expose args.strategy to existing code that used args.strategy variable name
    # (rest of function uses 'directory' and 'args')

    # set python output directory as a sibling to the evaluation folder (same level as Evaluation)
    global python_output_dir
    parent_dir = os.path.dirname(os.path.abspath(directory))
    python_output_dir = os.path.join(parent_dir, 'Evaluation_Python')
    os.makedirs(python_output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        sys.exit(0)

    # collect per-positions counts across files so we can make a combined plot
    positions_counts = {}
    for csv in csv_files:
        try:
            ret = plot_csv_file(csv, save=args.save, show=args.show)
        except Exception as e:
            print(f"Error processing {csv}: {e}")
            ret = None
        if ret and isinstance(ret, tuple) and len(ret) == 3:
            basename, counts_df, mat = ret
            # normalize column name (strip trailing '_Positions')
            key = basename.replace('_Positions', '')
            positions_counts[key] = (counts_df, mat)

    # If we have multiple operation counts, create a combined overlay plot and CSV
    if positions_counts:
        # determine global max position
        max_pos = 0
        for df, _ in positions_counts.values():
            if not df.empty:
                max_p = int(df['position'].max())
                if max_p > max_pos:
                    max_pos = max_p
        positions = np.arange(max_pos + 1)
        combined = pd.DataFrame({'position': positions})
        for name, (df, mat) in positions_counts.items():
            # merge counts; fill missing positions with 0
            merged = pd.merge(combined[['position']], df, on='position', how='left')
            combined[name] = merged['count'].fillna(0).astype(int)

        combined_csv = py_out('Combined_Operations_counts.csv')
        if _should_write(combined_csv):
            combined.to_csv(combined_csv, index=False)
            print(f"Wrote combined counts CSV: {combined_csv}")
        else:
            print(f"Skipping existing combined counts CSV: {combined_csv}")

        # plot overlay as stacked bar plot instead of lines
        op_cols = [k for k in combined.columns if k != 'position']
        x = combined['position'].values
        fig, ax = plt.subplots(figsize=(12, 5))
        bottom = np.zeros(len(x))
        for col in op_cols:
            ax.bar(x, combined[col].values, bottom=bottom, label=col, width=1.0)
            bottom = bottom + combined[col].values
        ax.set_title('Combined operation counts per position (stacked)')
        ax.set_xlabel('position')
        ax.set_ylabel('count')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        if len(op_cols) > 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        out_combined = py_out('Combined_Operations_counts.png')
        if args.save:
            if _should_write(out_combined):
                plt.savefig(out_combined)
                print(f"Saved combined plot: {out_combined}")
                # write data-embedded TeX file for combined plot
                tex_path = out_combined.replace('.png', '.tex')
                if _should_write(tex_path):
                    write_tex_stacked_from_df(tex_path, combined, 'position', op_cols, 'position', 'count', 'Combined operation counts per position (stacked)')
            else:
                print(f"Skipping existing combined plot: {out_combined}")
        if args.show:
            plt.show()
        else:
            plt.close()

        # Bucketed counts - absolute and normalized stacked plots
        bucketed_df = bucket_combined_counts_df(combined, n_buckets=args.buckets)
        bucketed_csv = py_out('Bucketed_Operations_counts.csv')
        if _should_write(bucketed_csv):
            bucketed_df.to_csv(bucketed_csv, index=False)
            print(f"Wrote bucketed counts CSV: {bucketed_csv}")
        else:
            print(f"Skipping existing bucketed counts CSV: {bucketed_csv}")

        # absolute stacked
        plot_buckets_stacked(bucketed_df, py_out('Bucketed_Operations_counts_absolute.png'), normalize=False, save=args.save, show=args.show)
        # normalized stacked
        plot_buckets_stacked(bucketed_df, py_out('Bucketed_Operations_counts_normalized.png'), normalize=True, save=args.save, show=args.show)

        # Per-position stacked bar plots - absolute and normalized
        plot_positions_bars(combined, py_out('Positions_Operations_counts_absolute.png'), normalize=False, save=args.save, show=args.show)
        plot_positions_bars(combined, py_out('Positions_Operations_counts_normalized.png'), normalize=True, save=args.save, show=args.show)

        # Nodes/Edges per position - build a combined presence matrix across all mats
        mats = [mat for (_df, mat) in positions_counts.values()]
        if mats:
            # determine unified sizes
            max_rows = max(m.shape[0] for m in mats)
            width = max_pos + 1
            mat_all = np.zeros((max_rows, width), dtype=int)
            for m in mats:
                r, c = m.shape
                # OR into mat_all; positions align at column 0
                mat_all[:r, :c] |= m

            # read global node/edge counts from Evaluation folder
            nodes_csv = os.path.join(directory, 'Number_of_Nodes.csv')
            edges_csv = os.path.join(directory, 'Number_of_Edges.csv')
            if not os.path.isfile(nodes_csv) or not os.path.isfile(edges_csv):
                print(f"Number_of_Nodes.csv or Number_of_Edges.csv not found in {directory}; skipping nodes/edges per-position plots")
            else:
                try:
                    nodes_df = pd.read_csv(nodes_csv)
                    edges_df = pd.read_csv(edges_csv)
                    if 'value' not in nodes_df.columns or 'value' not in edges_df.columns:
                        print('Number_of_Nodes/Edges CSV do not contain a "value" column; skipping')
                    else:
                        node_vals = nodes_df['value'].fillna(0).astype(float).values
                        edge_vals = edges_df['value'].fillna(0).astype(float).values
                        # create combined plot
                        out_prefix = py_out('Combined')
                        plot_nodes_edges_per_position(mat_all, node_vals, edge_vals, out_prefix, save=args.save, show=args.show)
                except Exception as e:
                    print(f"Error reading node/edge CSVs: {e}")


if __name__ == '__main__':
    main()
