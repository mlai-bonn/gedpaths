# Next Steps: GNNGED Optimization Implementation

## Overview

This document provides an actionable roadmap for implementing 78 identified optimizations across the GNNGED project. The full analysis is available in [`specs/OPTIMIZATION_REPORT.md`](./OPTIMIZATION_REPORT.md).

**Expected Impact:** 2-5x faster pipeline execution for large datasets through compiler optimizations, algorithmic improvements, and memory management enhancements.

**Priority Distribution:** 8 Critical | 19 High | 30 Medium | 21 Low

---

## Quick Reference: Top 10 Optimizations

| Priority | Issue | File | Expected Impact | Effort |
|----------|-------|------|-----------------|--------|
| **Critical** | Enable `-O3` and LTO compiler flags | `CMakeLists.txt` | 20-40% speedup | 5 min |
| **Critical** | Replace linear searches with hash maps | `create_edit_mappings.h` | 10-100x speedup (large datasets) | 1 hour |
| **Critical** | Fix Python bulk edge reading | `torch_geometric_exporter.py` | 10-50x faster conversion | 30 min |
| **Critical** | Eliminate deep copies in EditPathStatistics | `EditPathStatistics` constructor | Major memory reduction | 2 hours |
| **High** | Add vector `.reserve()` calls | Multiple C++ files | 50-90% memory reduction | 2 hours |
| **High** | Fix hardcoded CLI values in wl_analysis | `wl_analysis.py` | Restore CLI functionality | 15 min |
| **High** | Use `emplace_back()` instead of `push_back()` | Multiple C++ files | Reduce allocations | 1 hour |
| **Medium** | Implement concurrent dataset downloads | `download_datasets.sh` | 3-5x faster setup | 2 hours |
| **Medium** | Unified CLI argument parser | Multiple Python scripts | Better UX, consistency | 4 hours |
| **Medium** | Refactor statistics classes | Statistics files | Eliminate 300+ LOC duplication | 8 hours |

---

## Implementation Priority

### Phase 1: Quick Wins (Week 1)
**Focus:** High impact, low effort optimizations that deliver immediate results

#### Compiler Optimizations (5 minutes)
- **File:** `CMakeLists.txt`
- **Action:** Change `-O2` to `-O3 -march=native`, enable `-flto`
- **Impact:** 20-40% performance gain across all operations
- **Risk:** Very low

#### Hash-Based Lookups (1 hour)
- **File:** `create_edit_mappings.h`
- **Action:** Replace `find_if()` linear searches with `std::unordered_map<string, size_t>`
- **Impact:** O(n*m) → O(n+m), 10-100x speedup on large datasets
- **Risk:** Low, straightforward refactor

#### Python Bulk Edge Reading (30 minutes)
- **File:** `python_src/converter/torch_geometric_exporter.py:142-149`
- **Action:** Read entire `edge_index` tensor instead of element-by-element
- **Code:**
  ```python
  # Replace: edge_list.append([edge_index[0][i].item(), edge_index[1][i].item()])
  # With: edge_list = edge_index.t().tolist()
  ```
- **Impact:** 10-50x faster PyTorch Geometric conversion
- **Risk:** Very low

#### Fix CLI Hardcoded Values (15 minutes)
- **File:** `python_src/wl_analysis.py:193-194`
- **Action:** Remove hardcoded `target_file` and `output_file` overrides
- **Impact:** CLI arguments actually work as expected
- **Risk:** None

#### Vector Reserve Calls (2 hours)
- **Files:** `readGraphFromFile.cpp`, `readTrainValTest.cpp`, `readEditMappings.cpp`
- **Action:** Add `.reserve(expected_size)` before loops with known sizes
- **Examples:**
  - `file_lines.reserve(file_size / avg_line_length)`
  - `edges.reserve(num_edges * 2)`
- **Impact:** 50-90% reduction in memory allocations
- **Risk:** Low

#### Use `emplace_back()` (1 hour)
- **Files:** Multiple C++ files using `push_back()`
- **Action:** Replace `vec.push_back(T(...))` with `vec.emplace_back(...)`
- **Impact:** Eliminate temporary object construction
- **Risk:** Very low

**Phase 1 Total:** ~5 hours | Expected speedup: 2-3x on typical workloads

---

### Phase 2: Core Optimizations (Week 2)
**Focus:** Higher-effort items with significant impact on memory and performance

#### Eliminate Deep Copies (2 hours)
- **File:** `EditPathStatistics` constructor
- **Action:** Use `const&` parameters or move semantics instead of pass-by-value
- **Impact:** Significant memory reduction for large graphs
- **Risk:** Medium (need to verify ownership semantics)

#### Move Semantics for Returns (3 hours)
- **Files:** Multiple functions returning large objects (Graph, Statistics)
- **Action:** Return by value and rely on RVO, or explicitly `std::move()`
- **Impact:** Eliminate unnecessary copies
- **Risk:** Low with C++11 and later

#### Replace Raw Pointers (4 hours)
- **Files:** ~10 files using raw pointer arrays
- **Action:** Replace with `std::vector` or `std::unique_ptr`
- **Impact:** Automatic memory management, prevent leaks
- **Risk:** Medium (need careful testing)

#### Concurrent Dataset Downloads (2 hours)
- **File:** `download_datasets.sh`
- **Action:** Add `parallel` or background jobs with `&` + `wait`
- **Impact:** 3-5x faster dataset preparation
- **Risk:** Low

#### Error Handling Refactor (6 hours)
- **Files:** Multiple files using `exit(1)` or no error checks
- **Action:** Replace with exceptions or error return codes
- **Impact:** Better resource cleanup, testability
- **Risk:** Medium (changes error flow)

**Phase 2 Total:** ~17 hours | Expected improvement: Better memory usage, robustness

---

### Phase 3: Architectural Improvements (Week 3)
**Focus:** Medium priority items requiring larger refactoring

#### Unified CLI Parser (4 hours)
- **Files:** `python_src/*.py` (10+ scripts with inconsistent arg parsing)
- **Action:** Create `common/arg_parser.py` with standard arguments
- **Impact:** Consistent UX, easier maintenance
- **Risk:** Low

#### Statistics Class Refactoring (8 hours)
- **Files:** `EditPathStatistics`, `EditStatistics`, `MappingStatistics`
- **Action:** Extract common interface, eliminate 300+ lines of duplication
- **Impact:** Maintainability, reduced code size
- **Risk:** Medium (touches core functionality)

#### Module Splitting (6 hours)
- **Files:** Large `.cpp` files (e.g., 500+ line files)
- **Action:** Split into logical modules with clear responsibilities
- **Impact:** Improved compilation times, testability
- **Risk:** Low

#### Config File System (4 hours)
- **Action:** Add YAML/JSON config file support for common parameters
- **Impact:** Easier experiment management
- **Risk:** Low

**Phase 3 Total:** ~22 hours | Expected improvement: Code quality, maintainability

---

### Phase 4: Code Quality (Week 4)
**Focus:** Low-to-medium priority maintainability improvements

#### Code Deduplication (8-12 hours)
- **Files:** Multiple files with duplicated patterns (~1,200-1,500 LOC)
- **Action:** Extract common utilities for:
  - File I/O patterns
  - Graph operations
  - Statistics printing
- **Impact:** Reduced LOC, easier updates
- **Risk:** Low

#### Documentation Pass (4 hours)
- **Action:** Add docstrings to Python functions, Doxygen comments to C++ classes
- **Impact:** Developer onboarding, API clarity
- **Risk:** None

#### Test Coverage (8 hours)
- **Action:** Add unit tests for core algorithms, integration tests for pipeline
- **Impact:** Regression prevention, confidence in changes
- **Risk:** Low

#### Linter/Formatter Setup (2 hours)
- **Action:** Configure clang-format, black, isort; add pre-commit hooks
- **Impact:** Consistent code style
- **Risk:** None

**Phase 4 Total:** ~22-26 hours | Expected improvement: Long-term maintainability

---

## Implementation Strategy

### Testing Approach
1. **Baseline:** Run full pipeline on sample datasets and record timings before changes
2. **Incremental:** Test each optimization individually to verify impact
3. **Regression:** Ensure outputs match baseline (hash comparison)
4. **Benchmarking:** Use real datasets to measure performance gains

### Rollout Plan
- Implement Phase 1 optimizations in parallel (independent changes)
- Test Phase 1 thoroughly before proceeding to Phase 2
- Track performance metrics after each phase
- Document any issues or unexpected behaviors

### Dependencies
- Phase 1 items are independent and can be done in any order
- Phase 2 depends on Phase 1 testing completion
- Phase 3 requires stable codebase from Phases 1-2
- Phase 4 can run in parallel with Phase 3

### Risk Mitigation
- Create feature branches for medium/high-risk changes
- Use git tags to mark stable versions after each phase
- Keep original functions as `_legacy` versions until testing complete
- Add assertions to verify algorithmic correctness

---

## Quick Command Reference

### Build with optimizations
```bash
cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
```

### Run benchmark
```bash
python3 python_src/benchmark_pipeline.py --dataset AIDS --record-timing
```

### Profile hotspots
```bash
perf record -g ./build/gnnged_edit --source data/test.graph --target data/test.graph
perf report
```

### Memory profiling
```bash
valgrind --tool=massif ./build/gnnged_edit --source data/test.graph --target data/test.graph
```

---

## References

- **Full Analysis:** [`specs/OPTIMIZATION_REPORT.md`](./OPTIMIZATION_REPORT.md) (745 lines, detailed technical analysis)
- **Auto-Claude Task:** `.auto-claude/specs/002-please-implement-the-optimizations-that-were-found/spec.md`
- **Project Structure:** Root `README.md`

---

**Last Updated:** 2026-02-11
**Report Generated:** 2026-01-31
**Total Optimizations:** 78 (8 Critical, 19 High, 30 Medium, 21 Low)
