# GEDPaths Project Optimization Report

**Analysis Date:** 2026-01-31
**Project:** GEDPaths - Graph Edit Distance Mappings and Edit Paths
**Scope:** C++ Core, Python Utilities, Build System

---

## Executive Summary

This comprehensive analysis of the GEDPaths project identifies **75+ optimization opportunities** across performance, memory usage, code quality, and maintainability. The project is a well-structured C++20 codebase with Python utilities for computing Graph Edit Distance (GED) mappings, but several areas can be significantly improved.

### Key Findings Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| C++ Performance | 2 | 5 | 8 | 4 | 19 |
| Memory Management | 3 | 3 | 6 | 3 | 15 |
| Build System | 0 | 3 | 2 | 1 | 6 |
| Python Utilities | 2 | 4 | 8 | 9 | 23 |
| Code Quality | 1 | 4 | 6 | 4 | 15 |
| **Total** | **8** | **19** | **30** | **21** | **78** |

### Estimated Performance Gains

| Optimization Area | Estimated Improvement |
|-------------------|----------------------|
| Compiler Flags (-O3, -march=native, LTO) | 20-40% |
| Algorithm Complexity (O(n*m) to O(n)) | 10-100x for large datasets |
| Memory Allocation Patterns | 50-90% reduction in key areas |
| Python Converter Performance | 2-5x faster conversion |
| PGO (Profile-Guided Optimization) | Additional 10-25% |

---

## Impact/Effort Matrix

### Priority 1: Quick Wins (High Impact, Low Effort)

These optimizations can be implemented in hours and provide significant benefits:

| ID | Issue | File:Line | Impact | Est. Time |
|----|-------|-----------|--------|-----------|
| QW-1 | Use `-O3` instead of `-O2` | CMakeLists.txt:8 | +10-25% perf | 5 min |
| QW-2 | Add `-march=native` | CMakeLists.txt:8 | +10-40% SIMD | 5 min |
| QW-3 | Fix wl_analysis.py hardcoded values | wl_analysis.py:52-58 | Script broken | 5 min |
| QW-4 | Convert existing_pairs to `unordered_set` | create_edit_mappings.h:272-283 | O(n*m) to O(n) | 30 min |
| QW-5 | Add `reserve()` calls to vectors | Multiple files | 2-3x fewer allocs | 30 min |
| QW-6 | Replace `exit(1)` with return | create_edit_mappings.h:37 | RAII safety | 10 min |
| QW-7 | Element size constant lookup | torch_geometric_exporter.py:49 | N fewer allocs | 10 min |

### Priority 2: Medium Effort Optimizations (High Impact, Medium Effort)

| ID | Issue | File:Line | Impact | Est. Time |
|----|-------|-----------|--------|-----------|
| ME-1 | Enable LTO (Link-Time Optimization) | CMakeLists.txt | +5-15% perf | 1 hour |
| ME-2 | Parallel invalid mapping processing | create_edit_mappings.h:106-129 | 10-30x speedup | 4 hours |
| ME-3 | Bulk edge reading in converter | torch_geometric_exporter.py:188-199 | 10-50x faster | 2 hours |
| ME-4 | Store reference instead of copy in EditPathStatistics | analyze_edit_path_graphs.h:146-147 | 50-90% memory | 2 hours |
| ME-5 | Implement PGO build workflow | CMakeLists.txt | +10-25% perf | 4 hours |
| ME-6 | Add concurrent downloads | data_loader.py:196-265 | 5x faster | 2 hours |
| ME-7 | Replace std::map with unordered_map | analyze_edit_path_graphs.h:163-164 | 2-5x faster | 1 hour |

### Priority 3: Architectural Improvements (Medium Impact, Higher Effort)

| ID | Issue | File | Impact | Est. Time |
|----|-------|------|--------|-----------|
| AI-1 | Create unified CLI argument parser | New: src/arg_parser.h | -300 lines | 1 day |
| AI-2 | Split visualization_functions.py | visualization_functions.py | Maintainability | 2 days |
| AI-3 | Merge statistics classes | analyze_*.h | -100 lines | 1 day |
| AI-4 | Create path utilities | New: src/path_utils.h | -100 lines | 4 hours |
| AI-5 | Standardize error handling | All C++ files | Safety | 2 days |

---

## Detailed Findings

### 1. C++ Performance Analysis

#### 1.1 Critical: Linear Search Patterns (O(n*m) complexity)

**Location:** `src/create_edit_mappings.h`

Three critical linear search patterns create O(n*m) complexity where O(n) is achievable:

**Issue 1: Line 158 - Duplicate detection in `get_existing_mappings`**
```cpp
if (ranges::find(existing_graph_ids, res.graph_ids) == existing_graph_ids.end()) {
    results.emplace_back(res);
    existing_graph_ids.emplace_back(res.graph_ids);
}
```
- **Current:** O(n) per lookup in loop = O(n*m) total
- **Fix:** Use `std::unordered_set<std::pair<INDEX, INDEX>, PairHash>`
- **Impact:** 10-100x speedup for large result sets

**Issue 2: Lines 272-283 - Pair filtering in `create_edit_mappings`**
```cpp
for (const auto& pair : existing_pairs) {
    auto it = ranges::find(graph_pairs, pair);  // O(n) per call
    // ...
}
```
- **Current:** O(m * n) for existing pairs search
- **Fix:** Convert `existing_pairs` to hash set before loop
- **Impact:** Critical for 1M+ pair workloads

**Issue 3: Line 283 - Next pairs filtering**
```cpp
if (ranges::find(existing_pairs, pair) == existing_pairs.end()) {
    next_graph_pairs.emplace_back(pair);
}
```
- **Current:** O((n-max_index) * m) complexity
- **Fix:** Same unordered_set solution

**Recommended Fix:**
```cpp
struct PairHash {
    size_t operator()(const std::pair<INDEX, INDEX>& p) const {
        return std::hash<INDEX>{}(p.first) ^ (std::hash<INDEX>{}(p.second) << 1);
    }
};
using PairSet = std::unordered_set<std::pair<INDEX, INDEX>, PairHash>;
```

#### 1.2 High: Race Condition Workaround Forces Single-Threading

**Location:** `src/create_edit_mappings.h:74-94`

The code explicitly forces single-threaded execution for F1/F2 MIP-based methods:

```cpp
// Comment: "errors seem to come from parallelization in F2/F1"
if (ged_method == ged::Options::GEDMethod::F2 || ged_method == ged::Options::GEDMethod::F1) {
    modified_method_options.replace(start_of_number, end_of_threads - start_of_number, "1");
}
```

- **Impact:** Invalid mapping fixes run 30x slower (typical config uses 30 threads)
- **Root Cause:** Race conditions in GEDLIB's MIP solver integration
- **Recommendation:** Investigate parallel batch processing with shared GED environment

#### 1.3 High: Sequential Invalid Mapping Processing

**Location:** `src/create_edit_mappings.h:106-129`

```cpp
for (const auto &id : invalid_mappings) {
    auto fixed_result = create_edit_mappings_single(...);  // Creates new GED env each time
    // ...
}
```

- **Issue:** Sequential loop, new GED environment per iteration (expensive)
- **Recommendation:** Create single GED environment, batch process with `#pragma omp parallel for`
- **Estimated Impact:** 10-30x speedup for large invalid mapping sets

#### 1.4 Medium: Two-Pass Statistics Computation

**Location:** `src/analyze_mappings.h:34-48` and `src/analyze_edit_path_graphs.h:53-66`

Both statistics implementations use two passes over data (one for sum/min/max, one for variance).

- **Recommendation:** Use Welford's online algorithm for single-pass variance:
```cpp
double mean = 0, M2 = 0;
for (size_t i = 0; i < vals.size(); ++i) {
    double delta = vals[i] - mean;
    mean += delta / (i + 1);
    M2 += delta * (vals[i] - mean);
}
double variance = M2 / n;
```
- **Impact:** 2x faster statistics computation, better cache locality

#### 1.5 Medium: Debug Output in Tight Loop

**Location:** `src/analyze_edit_path_graphs.h:173-175`

```cpp
for (const auto& entry : _edit_path_info) {
    std::cout << "Processing edit paths for source graph: " << source_graph_name << std::endl;
```

- **Issue:** Console I/O in loop significantly slows processing
- **Recommendation:** Make debug output conditional:
```cpp
#ifndef NDEBUG
    std::cout << "Processing..." << std::endl;
#endif
```

---

### 2. Memory Management Analysis

#### 2.1 Critical: Deep Copy in EditPathStatistics Constructor

**Location:** `src/analyze_edit_path_graphs.h:146-147`

```cpp
EditPathStatistics::EditPathStatistics(const GraphData<UDataGraph> &edit_paths,
    const std::vector<std::tuple<INDEX, INDEX, INDEX, EditOperation>> &edit_path_info)
    : _edit_paths(edit_paths), _edit_path_info(edit_path_info) {
```

- **Issue:** Copies entire `GraphData<UDataGraph>` (potentially GB of data)
- **Recommendation:** Store reference or use move semantics:
```cpp
class EditPathStatistics {
private:
    const GraphData<UDataGraph>& _edit_paths;  // Reference instead of copy
```
- **Impact:** 50-90% memory reduction, eliminates O(n) deep copy

#### 2.2 Critical: ValueStatistics Vector Copy

**Location:** `src/analyze_edit_path_graphs.h:49-51`

```cpp
ValueStatistics::ValueStatistics(const std::string &name, const std::vector<double> &values) {
    _name = name;
    _values = values;  // Full vector copy
```

- **Issue:** Copies entire values vector for each of 11 statistics objects
- **Recommendation:** Use move semantics:
```cpp
ValueStatistics::ValueStatistics(std::string name, std::vector<double> values)
    : _name(std::move(name)), _values(std::move(values)) {
```

#### 2.3 High: Missing reserve() Calls

**Locations:**
- `src/create_edit_mappings.h`: Lines 105, 153, 224, 267
- `src/create_edit_paths.h`: Line 63
- `src/analyze_edit_path_graphs.h`: Lines 148-158

```cpp
// Example - Line 224
std::vector<std::pair<INDEX, INDEX>> graph_pairs;
// Missing: graph_pairs.reserve(max_number_of_pairs);
```

- **Impact:** Multiple reallocations copying all elements
- **Recommendation:** Add `reserve()` before all loops that populate vectors

#### 2.4 High: Valid Results Copy Loop

**Location:** `src/create_edit_paths.h:63-68`

```cpp
std::vector<GEDEvaluation<UDataGraph>> valid_results;
for (size_t i = 0; i < results.size(); ++i) {
    if (std::find(invalids.begin(), invalids.end(), static_cast<int>(i)) == invalids.end()) {
        valid_results.push_back(results[i]);  // Copies GEDEvaluation
    }
}
```

- **Issues:**
  1. Linear search in invalids (O(n*m) total)
  2. Copies expensive `GEDEvaluation` objects
- **Fix:**
```cpp
std::unordered_set<int> invalid_set(invalids.begin(), invalids.end());
valid_results.reserve(results.size());
for (size_t i = 0; i < results.size(); ++i) {
    if (!invalid_set.count(static_cast<int>(i))) {
        valid_results.push_back(std::move(results[i]));  // Move instead of copy
    }
}
```

#### 2.5 Medium: Temporary Vector for Single Validity Check

**Location:** `src/create_edit_mappings.h:110, 118`

```cpp
if (CheckResultsValidity(std::vector<GEDEvaluation<UDataGraph>>{fixed_result}).empty()) {
```

- **Issue:** Creates temporary vector for single-element validation
- **Recommendation:** Add single-element overload:
```cpp
bool CheckSingleResultValidity(const GEDEvaluation<UDataGraph>& result);
```

---

### 3. Build System Analysis

#### 3.1 High: Optimization Level (-O2 vs -O3)

**Location:** `CMakeLists.txt:8`

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")
```

**Current:** `-O2` provides conservative optimization

**Recommended:**
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
```

**What `-O3` enables beyond `-O2`:**
- Aggressive function inlining (`-finline-functions`)
- Loop vectorization (`-ftree-vectorize`)
- Loop unrolling with higher thresholds
- Function splitting for better cache utilization

**Expected Impact:** +10-25% performance for compute-heavy operations

#### 3.2 High: Missing Architecture-Specific Optimization

**Location:** `CMakeLists.txt`

**Current:** No architecture flags (targets baseline x86-64, SSE2 only)

**Recommended:**
```cmake
option(USE_NATIVE_ARCH "Enable -march=native for optimal local performance" ON)
if(USE_NATIVE_ARCH)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
endif()
```

**Benefits:**
- Enables AVX/AVX2/AVX-512 instructions
- FMA (Fused Multiply-Add)
- Better prefetching patterns

**Expected Impact:** +10-40% for SIMD-enabled operations (Eigen heavily benefits)

**Note:** Reduces portability - binary won't run on older CPUs

#### 3.3 High: No Link-Time Optimization (LTO)

**Location:** `CMakeLists.txt`

**Recommended:**
```cmake
option(USE_LTO "Enable Link-Time Optimization" ON)
if(USE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()
```

**Benefits:**
- Cross-module inlining
- Whole-program dead code elimination
- Better devirtualization

**Expected Impact:** +5-15% performance

**Trade-off:** Longer link times (2-5x)

#### 3.4 Medium: Profile-Guided Optimization (PGO) Potential

GEDPaths is an excellent candidate for PGO:
- Clear hot code paths in GED computation
- Deterministic workloads for stable profiling

**PGO Workflow:**
```bash
# Phase 1: Instrumented build
cmake -DCMAKE_CXX_FLAGS="-O3 -fprofile-generate=/tmp/pgo-data" ..
make && ./CreateMappings -db MUTAG -num_pairs 1000 -method F2

# Phase 2: Optimized build
cmake -DCMAKE_CXX_FLAGS="-O3 -fprofile-use=/tmp/pgo-data" ..
```

**Expected Impact:** Additional +10-25% on top of other optimizations

#### 3.5 Medium: Missing Library Linkage

**Location:** `CMakeLists.txt:49-63`

Only 2 of 7 targets have explicit `target_link_libraries()` calls:
- Missing: AnalyzePaths, AnalyzeMappings, Test, ConvertPrecomputed

Currently relies on deprecated global `link_directories()`.

---

### 4. Python Utilities Analysis

#### 4.1 Critical: Hardcoded Values Override CLI Arguments

**Location:** `python_src/wl_analysis.py:52-58`

```python
db_name = "NCI1"                    # OVERWRITES args.db_name!
strategy = "Rnd_d-IsoN"             # OVERWRITES args.strategy!
bgf_path = f"Results/Paths_{strategy}/F2/{db_name}/{db_name}_edit_paths.bgf"
```

- **Impact:** Script is non-functional for any dataset except NCI1
- **Fix:** Delete lines 52-58

#### 4.2 Critical: Element-by-Element Edge Reading

**Location:** `python_src/converter/torch_geometric_exporter.py:188-199`

```python
for e_i in range(m):
    u = _read_size_t(f, endian, size_t_bytes)
    v = _read_size_t(f, endian, size_t_bytes)
    # ... per-edge operations
```

- **Issue:** For 10K edges, creates 20K+ struct.unpack calls
- **Fix:** Bulk read all edges:
```python
edge_bytes = m * 2 * size_t_bytes
edge_data = _read_exact(f, edge_bytes)
edge_arr = np.frombuffer(edge_data, dtype=st_dtype).reshape(m, 2).T
```
- **Expected Impact:** 10-50x faster edge reading

#### 4.3 High: Temporary Tensor for Element Size

**Location:** `python_src/converter/torch_geometric_exporter.py:49`

```python
buf = _read_exact(f, count * torch.tensor([], dtype=dtype).element_size())
```

- **Issue:** Creates temporary tensor just to get element size
- **Fix:**
```python
_DTYPE_SIZES = {torch.float32: 4, torch.float64: 8, torch.int64: 8, ...}
elem_size = _DTYPE_SIZES[dtype]
```

#### 4.4 High: No Concurrent Downloads

**Location:** `python_src/data_loader.py:196-265`

Downloads execute sequentially. For multiple datasets, this is 5x slower than necessary.

**Fix:**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(download_dataset, req): req for req in datasets}
    for future in as_completed(futures):
        # Process result
```

#### 4.5 High: Visualization File Complexity

**Location:** `python_src/visualization/visualization_functions.py`

- **Size:** 1,266 lines (should be 5-7 modules)
- **Complexity:** `plot_edit_path` is 750+ lines with cyclomatic complexity ~45
- **Issues:**
  - 70+ silent `except Exception: pass` blocks
  - 3 nested helper functions that can't be unit tested
  - Double figure creation for title/no-title variants (2x render overhead)
  - Color mapping logic duplicated 3 times (~150 lines)

**Recommendations:**
1. Split into modules: `data_loading.py`, `layout.py`, `colors.py`, `renderers.py`, `plot_graph.py`, `plot_edit_path.py`
2. Extract nested functions to module level
3. Add logging instead of silent exceptions
4. Create `ColorMapper` utility class

---

### 5. Code Quality Analysis

#### 5.1 Critical: Process Termination Bypasses RAII

**Location:** `src/create_edit_mappings.h:37`

```cpp
if (source_id >= graphs.graphData.size() || target_id >= graphs.graphData.size()) {
    std::cerr << "Single source/target IDs out of range..." << std::endl;
    exit(1);  // DANGEROUS
}
```

- **Issue:** `exit(1)` bypasses destructors, causes resource leaks
- **Fix:** Return `std::optional<GEDEvaluation>` or error code

#### 5.2 High: Silent Exception Catching

**Location:** `src/analyze_edit_path_graphs.h:84-88, 328`

```cpp
try {
    fs::create_directories(output_dir);
} catch (...) {
    // Silent - hides permission errors
}
```

- **Fix:** Catch specific exceptions, log warnings:
```cpp
catch (const std::filesystem::filesystem_error& e) {
    std::cerr << "Warning: Could not create '" << output_dir << "': " << e.what() << "\n";
}
```

#### 5.3 High: Inconsistent Error Output

**Locations:** Multiple files

| Issue | Files | Fix |
|-------|-------|-----|
| `std::cout` for errors | create_edit_mappings.h:126,219 | Use `std::cerr` |
| German error messages | create_edit_paths.h:96,100 | Translate to English |
| Missing bounds check | All argument parsing | Check `argc` before `argv[i+1]` |

#### 5.4 High: Code Duplication (~1,200-1,500 lines)

| Category | Est. Lines | Files Affected |
|----------|-----------|----------------|
| CLI Argument Parsing | 250-300 | 4 C++ executables |
| Path Construction | 100-120 | Multiple headers |
| Statistics Computation | 80-100 | 2 header files |
| Validity Reporting | 60-80 | 2 files |
| CSV Writing | 60-80 | 3 files |
| Python InMemoryDataset | 60-80 | 2 Python files |
| Color Palette | 100-150 | Within visualization |

**Recommendation:** Create shared utilities:
- `src/arg_parser.h` - Common CLI arguments
- `src/path_utils.h` - Path construction using `std::filesystem`
- `src/statistics.h` - Unified statistics class
- `python_src/utils/dataset_loader.py` - PyG data loading

---

## Implementation Roadmap

### Week 1: Quick Wins

| Day | Task | Impact |
|-----|------|--------|
| 1 | Update CMakeLists.txt (-O3, -march=native, LTO) | 20-40% perf |
| 1 | Fix wl_analysis.py hardcoded values | Script functional |
| 2 | Add `reserve()` calls to all vectors | Fewer allocations |
| 2 | Replace `exit(1)` with return code | Safety |
| 3 | Convert linear searches to hash lookups | O(n*m) to O(n) |
| 4-5 | Python converter bulk edge reading | 10-50x faster |

### Week 2: Medium Effort Optimizations

| Day | Task | Impact |
|-----|------|--------|
| 1-2 | Refactor EditPathStatistics to use references | 50-90% memory |
| 3-4 | Add concurrent downloads to data_loader.py | 5x faster |
| 5 | Replace std::map with unordered_map | 2-5x faster |

### Week 3: Architectural Improvements

| Day | Task | Impact |
|-----|------|--------|
| 1-2 | Create unified CLI argument parser | -300 lines |
| 3-4 | Merge statistics classes | -100 lines |
| 5 | Standardize error handling | Safety |

### Week 4: Code Quality

| Day | Task | Impact |
|-----|------|--------|
| 1-3 | Begin visualization module split | Maintainability |
| 4-5 | Create path utilities, CSV writer | Consistency |

---

## Verification and Testing

### Performance Benchmarks

Run before and after optimizations:

```bash
# Time mapping creation
time ./CreateMappings -db AIDS -num_pairs 5000 -method F2 -method_options threads 30

# Profile with perf
perf stat -e cache-misses,cache-references ./CreateMappings -db MUTAG -num_pairs 1000

# Memory profiling
valgrind --tool=massif ./CreateMappings -db MUTAG -num_pairs 500
```

### Python Profiling

```python
import cProfile
cProfile.run('''
from python_src.converter.torch_geometric_exporter import bgf_to_pyg_data_list
bgf_to_pyg_data_list("Results/Paths_Rnd/F2/MUTAG/MUTAG_edit_paths.bgf")
''', 'converter.stats')
```

### Correctness Verification

After compiler flag changes:
```bash
./CreateMappings -db MUTAG -num_pairs 100 -method F2
diff results_optimized.txt results_baseline.txt
```

---

## Appendix A: Complete File Reference

### C++ Files Analyzed

| File | Lines | Key Issues |
|------|-------|------------|
| `src/create_edit_mappings.h` | 320 | Linear searches, race condition workaround, exit(1) |
| `src/create_edit_paths.h` | 111 | Linear search, German messages, missing reserve |
| `src/analyze_edit_path_graphs.h` | 385 | Deep copy, std::map, debug output |
| `src/analyze_mappings.h` | 183 | Two-pass statistics |
| `src/convert.h` | 200 | Good patterns, CSV duplication |
| `create_edit_mappings.cpp` | 120 | CLI parsing duplication |
| `create_edit_paths.cpp` | 100 | CLI parsing duplication |
| `analyze_edit_path_graphs.cpp` | 60 | Incomplete -help |
| `analyze_mappings.cpp` | 55 | CLI parsing duplication |
| `CMakeLists.txt` | 70 | Conservative flags, missing LTO |

### Python Files Analyzed

| File | Lines | Key Issues |
|------|-------|------------|
| `torch_geometric_exporter.py` | 378 | Element-by-element I/O, dtype conversions |
| `visualization_functions.py` | 1,266 | Monolithic, high complexity, duplication |
| `data_loader.py` | 271 | Sequential downloads, no retry |
| `wl_analysis.py` | 77 | Hardcoded values override CLI |
| `GEDPathsInMemory.py` | 275 | Generally good |
| `extract_gedlib_defaults.py` | 110 | Broad exception handling |

---

## Appendix B: Recommended CMakeLists.txt

```cmake
# ============================================
# Optimized Compiler Configuration
# ============================================
cmake_minimum_required(VERSION 3.20)
project(GNNGED CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_NATIVE_ARCH "Enable -march=native for optimal local performance" ON)
option(USE_LTO "Enable Link-Time Optimization for Release builds" ON)
option(USE_PGO_GENERATE "Build with profile generation instrumentation" OFF)
option(USE_PGO_USE "Build with profile-guided optimization" OFF)

# Base release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Architecture-specific optimization
if(USE_NATIVE_ARCH)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
    message(STATUS "Native architecture optimization enabled")
endif()

# Link-Time Optimization
if(USE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        message(STATUS "LTO enabled for Release builds")
    else()
        message(WARNING "LTO not supported: ${ipo_output}")
    endif()
endif()

# Profile-Guided Optimization
if(USE_PGO_GENERATE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate=/tmp/pgo-data")
    message(STATUS "PGO profile generation enabled")
elseif(USE_PGO_USE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-use=/tmp/pgo-data")
    message(STATUS "PGO profile use enabled")
endif()

# Warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# OpenMP
find_package(OpenMP REQUIRED)
```

---

## Appendix C: Hash Function for Pair Lookups

```cpp
// Add to a common header (e.g., src/utils.h)
struct PairHash {
    size_t operator()(const std::pair<INDEX, INDEX>& p) const noexcept {
        // Combine hashes using XOR and bit rotation
        size_t h1 = std::hash<INDEX>{}(p.first);
        size_t h2 = std::hash<INDEX>{}(p.second);
        return h1 ^ (h2 << 1) ^ (h2 >> 63);
    }
};

// Type alias for convenience
using PairSet = std::unordered_set<std::pair<INDEX, INDEX>, PairHash>;
using PairMap = std::unordered_map<std::pair<INDEX, INDEX>, size_t, PairHash>;
```

---

## Conclusion

This analysis identifies significant optimization opportunities across the GEDPaths codebase. The most impactful changes are:

1. **Compiler flags** (+20-40% performance with minimal risk)
2. **Algorithm complexity fixes** (O(n*m) to O(n) for pair filtering)
3. **Memory allocation patterns** (50-90% reduction in key areas)
4. **Python converter I/O** (2-5x faster conversion)

Implementing the "Quick Wins" in Week 1 would provide substantial performance improvements with minimal development risk. The architectural improvements in Weeks 3-4 would improve long-term maintainability.

**Total Estimated Performance Improvement:** 2-5x faster end-to-end pipeline execution for large datasets.

---

*Report generated by auto-claude analysis pipeline*
*Task: 001-analyze-the-project-and-suggest-optimizations*
*Subtask: 6-1*
