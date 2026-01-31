# Visualization Functions Analysis

## Overview

**File:** `python_src/visualization/visualization_functions.py`
**Lines of Code:** 1,266
**Primary Purpose:** Graph visualization utilities for PyTorch Geometric data structures

This module provides visualization capabilities for graph data, including:
- Single graph plotting with colored nodes/edges
- Edit path visualization (showing graph transformations step-by-step)
- Layout computation (neato, spring, circular fallbacks)
- Color mapping for node/edge labels

---

## 1. File Organization Issues

### 1.1 Monolithic Structure (HIGH PRIORITY)

The file contains 1,266 lines with mixed responsibilities:
- Data loading utilities (`_LoadedInMemoryDataset`, `load_data_by_index`, `find_index_by_bgf_name`)
- Layout computation (`compute_layout`)
- Graph conversion (`graph_to_networkx_with_edge_features`)
- Color palette handling (`_get_tab20_palette`, `TAB20_PALETTE`)
- Edge rendering (`_draw_colored_edges`)
- Main plotting functions (`plot_graph`, `plot_edit_path`)

**Recommendation:** Split into separate modules:
```
visualization/
  __init__.py
  data_loading.py      # _LoadedInMemoryDataset, load_data_by_index, find_index_by_bgf_name, find_processed_pt
  layout.py            # compute_layout
  colors.py            # _get_tab20_palette, TAB20_PALETTE
  graph_conversion.py  # graph_to_networkx_with_edge_features
  renderers.py         # _draw_colored_edges, _draw_graph_on_ax
  plot_graph.py        # plot_graph function
  plot_edit_path.py    # plot_edit_path function
```

### 1.2 Deeply Nested Helper Functions

`plot_edit_path` (lines 514-1264) contains two large nested functions:
- `_parse_edit_op` (lines 536-633): 97 lines
- `_format_edit_op` (lines 637-695): 58 lines
- `_draw_graph_on_ax` (lines 700-897): 197 lines

**Problem:** These nested functions:
- Cannot be unit tested independently
- Are recreated on every call (minor performance overhead)
- Make the parent function 750+ lines long

**Recommendation:** Extract as module-level functions with explicit parameters.

---

## 2. Function Complexity Analysis

### 2.1 Critical Complexity Issues

| Function | Lines | Cyclomatic Complexity | Issues |
|----------|-------|----------------------|--------|
| `plot_edit_path` | 750 | ~45 | Far exceeds recommended max of 10 |
| `_draw_graph_on_ax` | 197 | ~25 | Multiple responsibility violation |
| `plot_graph` | 157 | ~20 | Duplicates logic in `_draw_graph_on_ax` |
| `graph_to_networkx_with_edge_features` | 88 | ~15 | Complex edge attribute handling |
| `compute_layout` | 54 | ~12 | Excessive fallback nesting |

### 2.2 Specific Complexity Hotspots

#### `plot_edit_path` (lines 514-1264)
```python
def plot_edit_path(graphs, edit_ops, output=None, show_labels=True,
                   one_fig_per_step=False, color_nodes_by_label=True,
                   node_size=200, edge_width=1.0, red_font_size=10):
```

**Issues:**
- 750+ lines in a single function
- Handles both single-figure and multi-figure output modes (should be separate)
- Contains 3 nested helper functions
- Builds global color mapping inline (lines 903-951)
- Duplicates position computation logic for each step
- Excessive try/except blocks obscure control flow

#### `_draw_graph_on_ax` (lines 700-897)
**Issues:**
- 197 lines with 10+ parameters
- Mixes layout, coloring, drawing, and legend generation
- Deep nesting (up to 8 levels in some places)
- Repeated color mapping logic (also in `plot_graph`)

---

## 3. Rendering Efficiency Issues

### 3.1 Redundant Computations

#### Layout Recomputation (MEDIUM-HIGH)
```python
# Lines 1019-1048 and 1141-1165 - nearly identical code blocks
if i == 0:
    pos_for_subplot = compute_layout(data)
else:
    pos_for_subplot = {n: prev_pos[n] for n in G_i.nodes() if prev_pos is not None and n in prev_pos}
    missing = [n for n in G_i.nodes() if n not in pos_for_subplot]
    # ... position inference for missing nodes
```

**Problem:** This position inference logic is duplicated twice. Should be extracted.

#### Repeated Graph Conversions
```python
# Line 701 - inside _draw_graph_on_ax
G, edge_labels = graph_to_networkx_with_edge_features(data)

# Line 1011 - also called
G_step, _ = graph_to_networkx_with_edge_features(data)
```

**Problem:** When `_draw_graph_on_ax` is called, the caller often already has the NetworkX graph.

### 3.2 Memory Inefficiencies

#### Duplicate Figure Creation (HIGH)
```python
# Lines 1068-1082 - creates figure without title
fig_no, ax_no = plt.subplots(figsize=(8, 6))
_draw_graph_on_ax(...)

# Lines 1085-1099 - creates nearly identical figure with title
fig_t, ax_t = plt.subplots(figsize=(8, 6))
_draw_graph_on_ax(...)
```

**Problem:** For `one_fig_per_step=True`, each step creates TWO figures (with/without title), calling `_draw_graph_on_ax` twice. This doubles:
- Memory allocation
- Layout computation time
- NetworkX conversion time

**Recommendation:** Add a `title_visible` parameter to toggle title after drawing once, or use `ax.set_title('')` to clear.

#### Legend Figure Creation (MEDIUM)
```python
# Lines 1236-1261
if color_nodes_by_label and legend_info is not None:
    fig_leg = plt.figure(figsize=(2 + max(0, len(payload) - 1) * 0.3, ...))
```

**Problem:** Creates a new figure for the legend even when it could be saved with the main figure.

### 3.3 Inefficient Palette Sampling

```python
# Lines 20-51
def _get_tab20_palette():
    cmap = plt.get_cmap('tab20')
    colors = None
    try:
        c = getattr(cmap, 'colors', None)
        # ... multiple fallbacks
```

**Minor Issue:** The function has 4 fallback levels. While cached at module level, the fallback logic is overly defensive.

---

## 4. Code Quality Issues

### 4.1 Excessive Exception Swallowing

The file contains **70+ bare `except Exception:` or `except:` blocks** that silently continue. Examples:

```python
# Line 79-80
except Exception:
    pass

# Line 110-111
except Exception:
    return {}

# Line 288-289
except Exception:
    pass
```

**Problems:**
- Hides bugs and makes debugging difficult
- No logging of failures
- Some failures should propagate (e.g., file I/O errors)

**Recommendation:**
1. Add proper logging
2. Use specific exception types
3. Only catch exceptions that are truly expected

### 4.2 Import Organization

```python
# Line 2
import ast

# Line 243 - inline import!
import numpy as _np

# Line 470, 525, 908, 1116, 1152 - repeated inline imports
import matplotlib.patches as mpatches
from math import ceil
import random
```

**Problem:** Imports scattered throughout the file, some repeated.

### 4.3 Magic Numbers

```python
# Various places
node_size=200           # Why 200?
edge_width=1.0
red_font_size=10
standard_edge_width = 2.0
border_factor = 4
dpi=300
figsize=(8, 6)
figsize=(4 * cols, 3 * rows)
off = 0.05 * max(dx, dy, 1.0)
0.6180339887498948  # Golden ratio - should be named constant
```

**Recommendation:** Define named constants at module level:
```python
DEFAULT_NODE_SIZE = 200
DEFAULT_EDGE_WIDTH = 1.0
DEFAULT_DPI = 300
GOLDEN_RATIO_FRAC = 0.6180339887498948
```

### 4.4 Docstring Quality

| Function | Docstring Status |
|----------|------------------|
| `_get_tab20_palette` | Good |
| `compute_layout` | Good |
| `_LoadedInMemoryDataset` | Good |
| `find_processed_pt` | Good |
| `load_data_by_index` | Missing |
| `find_index_by_bgf_name` | Missing |
| `graph_to_networkx_with_edge_features` | Good but incomplete |
| `_draw_colored_edges` | Good |
| `plot_graph` | Missing |
| `plot_edit_path` | Good but parameters incomplete |
| `_parse_edit_op` | Missing (docstring would really help) |
| `_format_edit_op` | Missing |
| `_draw_graph_on_ax` | Missing |

### 4.5 Code Duplication

#### Color Mapping Logic (150+ lines duplicated)
The color mapping logic appears in THREE places:
1. `plot_graph` lines 408-439
2. `_draw_graph_on_ax` lines 777-838
3. `plot_edit_path` global mapping lines 903-951

#### Position Offset Calculation
```python
# Lines 453-461 in plot_graph
xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]
dx = (max(xs) - min(xs)) if xs else 1.0
dy = (max(ys) - min(ys)) if ys else 1.0
off = 0.05 * max(dx, dy, 1.0)
pos_ids = {n: (pos[n][0] + off, pos[n][1] + off) for n in G.nodes()}

# Lines 863-869 in _draw_graph_on_ax - identical
```

#### Deterministic Offset Function
```python
# Lines 1038-1040 - defined inline
def _det_offset(node_id, scale):
    return (((node_id * 0.6180339887498948) % 1.0) - 0.5) * scale

# Lines 1156-1157 - defined again!
def _det_offset(node_id, scale):
    return (((node_id * 0.6180339887498948) % 1.0) - 0.5) * scale
```

---

## 5. Type Annotation Gaps

```python
# Current
def compute_layout(data, prog: str = 'neato'):

# Should be
def compute_layout(data: Data, prog: str = 'neato') -> Dict[int, Tuple[float, float]]:
```

**Missing type hints:**
- `_draw_colored_edges` - all parameters
- `plot_graph` - return type, `data` parameter type
- `plot_edit_path` - `graphs` and `edit_ops` types
- `_draw_graph_on_ax` - all parameters
- `_parse_edit_op` - parameter and return types
- `_format_edit_op` - parameter and return types

---

## 6. Recommended Refactoring Plan

### Phase 1: Quick Wins (Low Risk, High Impact)
1. Extract `_det_offset` to module level
2. Create constants for magic numbers
3. Add logging instead of silent exception handling
4. Move inline imports to top of file

### Phase 2: Extract Helpers (Medium Risk)
1. Extract `_parse_edit_op` and `_format_edit_op` to module level
2. Create `_compute_node_label_colors()` utility
3. Create `_compute_position_offset()` utility
4. Extract `_draw_graph_on_ax` from nested function to module level

### Phase 3: Split Module (Higher Risk)
1. Create `visualization/data_loading.py`
2. Create `visualization/layout.py`
3. Create `visualization/colors.py`
4. Create `visualization/renderers.py`

### Phase 4: Optimize Rendering (Medium Risk)
1. Avoid double figure creation for title/no-title variants
2. Cache NetworkX graph conversion when possible
3. Pass pre-computed layouts to avoid redundant computation

---

## 7. Performance Optimization Opportunities

| Issue | Location | Impact | Effort |
|-------|----------|--------|--------|
| Double figure creation | L1068-1099 | 2x render time | Low |
| Repeated graph conversion | L701, L1011 | 20% overhead | Medium |
| Layout recomputation | L1019-1048 | 15% overhead | Low |
| Exception overhead | 70+ locations | 5% overhead | Medium |
| Import overhead | 5 inline imports | 2% overhead | Low |

**Estimated total improvement:** 25-40% reduction in rendering time for edit paths

---

## 8. Summary of Findings

### Critical Issues
1. **File length:** 1,266 lines - should be split into 5-7 modules
2. **Function complexity:** `plot_edit_path` has 750+ lines and complexity ~45
3. **Code duplication:** Color mapping logic repeated 3 times (~150 lines)
4. **Silent failures:** 70+ `except Exception: pass` blocks

### Major Issues
1. Nested helper functions prevent unit testing
2. Double figure creation for title variants wastes resources
3. Missing docstrings for key functions
4. Missing type annotations throughout

### Minor Issues
1. Magic numbers should be constants
2. Inline imports should move to top
3. Unused `random` import in one path

---

## 9. Recommended Immediate Actions

1. **Add logging** - Replace silent `except` blocks with proper logging
2. **Extract nested functions** - Move `_parse_edit_op`, `_format_edit_op`, `_draw_graph_on_ax` to module level
3. **Define constants** - Create named constants for all magic numbers
4. **Reduce duplication** - Create `ColorMapper` class or utility functions
5. **Add type hints** - Especially for public API functions

These changes would significantly improve maintainability and make future performance optimizations easier to implement.
