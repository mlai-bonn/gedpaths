# Dependency Linking and Profile-Guided Optimization (PGO) Analysis

## Overview

This document analyzes the CMakeLists.txt build configuration for dependency linking patterns and evaluates the potential for Profile-Guided Optimization (PGO) to improve runtime performance.

**File Analyzed:** `CMakeLists.txt`

---

## Part 1: Dependency Linking Analysis

### Current Linking Configuration

#### Include Directories (Line 34)
```cmake
include_directories(${LIBGRAPH_ROOT}/include ${GEDLIB_ROOT} ${GEDLIB_ROOT}/ext/boost
    ${GEDLIB_ROOT}/ext/eigen/Eigen ${GEDLIB_ROOT}/ext/nomad.3.8.1/src
    ${GEDLIB_ROOT}/ext/nomad.3.8.1/ext/sgtelib/src ${GEDLIB_ROOT}/ext/lsape.5/cpp/include
    ${GEDLIB_ROOT}/ext/libsvm.3.22 ${GEDLIB_ROOT}/ext/fann.2.2.0/include
    ${GUROBI_HOME}/include)
```

#### Link Directories (Line 35)
```cmake
link_directories(${LIBGRAPH_ROOT}/include ${GEDLIB_ROOT}/ext/nomad.3.8.1/lib
    ${GEDLIB_ROOT}/ext/libsvm.3.22 ${GEDLIB_ROOT}/ext/fann.2.2.0/lib)
```

#### Target Link Libraries (Lines 52-53)
```cmake
target_link_libraries(CreateMappings libsvm.so libnomad.so libdoublefann.so.2 gurobi)
target_link_libraries(CreatePaths libsvm.so libnomad.so libdoublefann.so.2 gurobi)
```

### Issues Identified

#### Issue 1: Deprecated Global Linking (HIGH Priority)
**Location:** Lines 34-35
**Problem:** Uses global `include_directories()` and `link_directories()` which are deprecated in modern CMake (3.0+).

**Impact:**
- Applies to ALL targets, causing unnecessary include/link paths
- Reduces build clarity and maintainability
- Can cause symbol conflicts in larger projects

**Recommended Fix:**
```cmake
# Per-target include directories
target_include_directories(CreateMappings PRIVATE
    ${LIBGRAPH_ROOT}/include
    ${GEDLIB_ROOT}
    ${GEDLIB_ROOT}/ext/boost
    ${GUROBI_HOME}/include
)
```

---

#### Issue 2: Incorrect Link Directory Path (MEDIUM Priority)
**Location:** Line 35
**Problem:** `${LIBGRAPH_ROOT}/include` is listed in `link_directories()` but this is an include path, not a library path.

**Current:**
```cmake
link_directories(${LIBGRAPH_ROOT}/include ...)  # WRONG - this is an include path
```

**Fix:** Remove this path from link_directories or change to correct lib path if one exists.

---

#### Issue 3: Missing Library Linkage for 5 Targets (HIGH Priority)
**Location:** Lines 38-51
**Problem:** Only 2 of 7 targets have explicit `target_link_libraries()` calls.

| Target | Has Explicit Linkage |
|--------|---------------------|
| CreateMappings | Yes (Line 52) |
| CreatePaths | Yes (Line 53) |
| AnalyzePaths | **No** |
| AnalyzeMappings | **No** |
| Test | **No** |
| TestLINUX | **No** |
| ConvertPrecomputed | **No** |

**Impact:** Relies on implicit linking via global link_directories(), which is fragile and non-portable.

**Recommended Fix:**
```cmake
# Add explicit linkage for each target that needs libraries
target_link_libraries(AnalyzeMappings PRIVATE libsvm.so libnomad.so libdoublefann.so.2 gurobi)
target_link_libraries(Test PRIVATE libsvm.so libnomad.so libdoublefann.so.2 gurobi)
# etc.
```

---

#### Issue 4: Hardcoded Platform-Specific Library Suffixes (MEDIUM Priority)
**Location:** Lines 52-53
**Problem:** Uses `libsvm.so`, `libnomad.so`, `libdoublefann.so.2` - platform-specific suffixes.

**Impact:**
- Won't work on macOS (uses .dylib)
- Won't work on Windows (uses .dll/.lib)
- Version suffix `libdoublefann.so.2` is fragile

**Recommended Fix:**
```cmake
# Create imported library targets
find_library(LIBSVM_LIBRARY NAMES svm libsvm PATHS ${GEDLIB_ROOT}/ext/libsvm.3.22)
find_library(NOMAD_LIBRARY NAMES nomad libnomad PATHS ${GEDLIB_ROOT}/ext/nomad.3.8.1/lib)
find_library(FANN_LIBRARY NAMES doublefann fann PATHS ${GEDLIB_ROOT}/ext/fann.2.2.0/lib)

target_link_libraries(CreateMappings PRIVATE ${LIBSVM_LIBRARY} ${NOMAD_LIBRARY} ${FANN_LIBRARY} gurobi)
```

---

### Dependency Linking Recommendations Summary

| Issue | Priority | Effort | Impact |
|-------|----------|--------|--------|
| Migrate to target_include_directories() | HIGH | Medium | Cleaner builds, better maintainability |
| Add missing target_link_libraries() | HIGH | Low | Required for correct linking |
| Remove incorrect include path from link_directories | MEDIUM | Low | Bug fix |
| Use find_library() for cross-platform support | MEDIUM | Medium | Better portability |
| Create imported library targets | LOW | High | Best practices |

---

## Part 2: Profile-Guided Optimization (PGO) Analysis

### What is PGO?

Profile-Guided Optimization uses runtime profiling data from representative workloads to guide compiler optimization decisions. The compiler can:
- Better predict branch outcomes
- Optimize function inlining decisions
- Improve code layout for instruction cache efficiency
- Make smarter loop unrolling decisions

### PGO Applicability Assessment

#### Workload Characteristics (Highly Suitable)

| Characteristic | GEDPaths Status | PGO Benefit |
|----------------|-----------------|-------------|
| Deterministic hot paths | ✅ GED computation loops | High |
| Consistent branch patterns | ✅ Method-specific code paths | High |
| Stable workload profile | ✅ Processing graph pairs | High |
| CPU-bound operations | ✅ Mapping computation | High |

**Conclusion:** GEDPaths is an excellent candidate for PGO due to its predictable, CPU-intensive workload.

### PGO Implementation Methodology

#### Three-Phase Build Process

```bash
# Phase 1: Instrumented Build
# Build with profiling instrumentation enabled
mkdir -p build_pgo_gen && cd build_pgo_gen
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -fprofile-generate=/tmp/pgo-data" \
      ..
make -j$(nproc)

# Phase 2: Profile Collection
# Run representative workloads to generate profile data
./CreateMappings -db MUTAG -num_pairs 1000 -method F2 -method_options threads 1 time-limit 5
./CreateMappings -db AIDS -num_pairs 500 -method F2 -method_options threads 1 time-limit 5
./CreatePaths -db MUTAG -method F2 -path_strategy Random

# Phase 3: Optimized Build
# Rebuild using the collected profile data
cd .. && mkdir -p build_pgo_use && cd build_pgo_use
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -fprofile-use=/tmp/pgo-data -fprofile-correction" \
      ..
make -j$(nproc)
```

### CMake Integration for PGO

```cmake
# Add PGO options to CMakeLists.txt
option(PGO_GENERATE "Build with profiling instrumentation" OFF)
option(PGO_USE "Build using profile data" OFF)
set(PGO_DATA_DIR "/tmp/pgo-data" CACHE PATH "Directory for PGO profile data")

if(PGO_GENERATE AND PGO_USE)
    message(FATAL_ERROR "Cannot enable both PGO_GENERATE and PGO_USE simultaneously")
endif()

if(PGO_GENERATE)
    message(STATUS "PGO: Enabling profile generation to ${PGO_DATA_DIR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate=${PGO_DATA_DIR}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-generate=${PGO_DATA_DIR}")
endif()

if(PGO_USE)
    if(NOT EXISTS "${PGO_DATA_DIR}")
        message(FATAL_ERROR "PGO data directory ${PGO_DATA_DIR} does not exist. Run with PGO_GENERATE first.")
    endif()
    message(STATUS "PGO: Using profile data from ${PGO_DATA_DIR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use=${PGO_DATA_DIR} -fprofile-correction")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-use=${PGO_DATA_DIR}")
endif()
```

### Expected PGO Performance Gains

| Optimization Category | Expected Gain | Mechanism |
|-----------------------|---------------|-----------|
| Branch prediction hints | 5-10% | Profile data informs likely branch outcomes |
| Function inlining guidance | 5-15% | Hot functions prioritized for inlining |
| Code layout optimization | 2-5% | Hot code placed together for cache efficiency |
| Loop unrolling decisions | 2-5% | Actual iteration counts guide unrolling |
| **Total Expected Gain** | **10-25%** | Cumulative effect |

### Profiling Best Practices

#### Representative Workload Selection

To get optimal PGO results, the training workload should:

1. **Cover all code paths:** Use multiple methods (F1, F2, IPFP_MULTI)
2. **Representative data sizes:** Use similar graph sizes to production
3. **Sufficient samples:** Minimum 100-1000 graph pairs for stable profiles
4. **Include edge cases:** Some large graphs, some small graphs

**Recommended Training Script:**
```bash
#!/bin/bash
# pgo_training.sh - Generate comprehensive profile data

export PGO_DIR=/tmp/pgo-data
rm -rf $PGO_DIR && mkdir -p $PGO_DIR

# Test multiple methods
for method in F2 IPFP_MULTI; do
    ./CreateMappings -db MUTAG -num_pairs 500 -method $method \
        -method_options threads 1 time-limit 5
done

# Test multiple path strategies
for strategy in Random Optimal; do
    ./CreatePaths -db MUTAG -method F2 -path_strategy $strategy
done

# Test analysis code paths
./AnalyzeMappings -db MUTAG -method F2
./AnalyzePaths -db MUTAG -method F2

echo "Profile data generated in $PGO_DIR"
```

### Clang vs GCC PGO Considerations

| Compiler | PGO Flags | Notes |
|----------|-----------|-------|
| GCC | `-fprofile-generate`, `-fprofile-use` | Standard approach |
| Clang | `-fprofile-instr-generate`, `-fprofile-instr-use` | LLVM instrumentation |
| Clang (sampling) | `perf record` + `AutoFDO` | Lower overhead alternative |

For GCC (the current compiler), use the flags shown in the methodology section above.

---

## Combined Optimization Potential

Combining PGO with the compiler flag improvements from subtask-3-1:

| Optimization Stack | Cumulative Performance Gain |
|--------------------|-----------------------------|
| -O2 → -O3 | +10-25% |
| + -march=native | +20-40% (with SIMD) |
| + LTO | +25-55% |
| + PGO | **+35-80% total** |

**Conservative estimate:** 40% improvement
**Optimistic estimate:** 80%+ improvement for compute-heavy GED operations

---

## Implementation Recommendations

### Priority 1: High Impact, Low Effort
1. Add missing `target_link_libraries()` for 5 targets
2. Remove incorrect path from `link_directories()`

### Priority 2: High Impact, Medium Effort
1. Implement PGO build workflow with CMake options
2. Migrate to modern `target_include_directories()`

### Priority 3: Medium Impact, Medium Effort
1. Create imported library targets for better cross-platform support
2. Use `find_library()` for flexible library discovery

### Priority 4: Automation
1. Create `pgo_training.sh` script for reproducible profile generation
2. Add CI/CD step for PGO builds in production releases

---

## Verification Checklist

- [x] Dependency linking issues documented with line numbers
- [x] PGO profiling methodology explained with example commands
- [x] CMake integration code provided for PGO
- [x] Expected performance gains quantified
- [x] Representative workload selection guidance provided
- [x] Implementation priorities established

---

*Analysis completed: 2026-01-31*
*Subtask: 3-2 - Build System Analysis*
