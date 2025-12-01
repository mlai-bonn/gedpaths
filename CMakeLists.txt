cmake_minimum_required(VERSION 3.20)
project(GEDPaths)
set(CMAKE_CXX_STANDARD 20)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(GUROBI REQUIRED)
message(STATUS "Found GUROBI: ${GUROBI_LIBRARY}, ${GUROBI_CXX_LIBRARY}, ${GUROBI_INCLUDE_DIRS}")
if (NOT GUROBI_FOUND)
    message(FATAL_ERROR "Could not find GUROBI library. Please set GUROBI_HOME environment variable.")
endif()

# find the libGraph folder
# set LIBGRAPH_ROOT to ../libGraph
# set GEDLIB_ROOT to ${LIBGRAPH_ROOT}/external/gedlib




set(LIBGRAPH_ROOT ../libGraph)
set(GEDLIB_ROOT ${LIBGRAPH_ROOT}/external/gedlib)


include_directories(${LIBGRAPH_ROOT}/include ${GEDLIB_ROOT} ${GEDLIB_ROOT}/ext/boost ${GEDLIB_ROOT}/ext/eigen/Eigen ${GEDLIB_ROOT}/ext/nomad.3.8.1/src ${GEDLIB_ROOT}/ext/nomad.3.8.1/ext/sgtelib/src ${GEDLIB_ROOT}/ext/lsape.5/cpp/include ${GEDLIB_ROOT}/ext/libsvm.3.22 ${GEDLIB_ROOT}/ext/fann.2.2.0/include ${GUROBI_HOME}/include)
link_directories(${LIBGRAPH_ROOT}/include ${GEDLIB_ROOT}/ext/nomad.3.8.1/lib ${GEDLIB_ROOT}/ext/libsvm.3.22 ${GEDLIB_ROOT}/ext/fann.2.2.0/lib)

# adding the Google_Tests_run target
add_executable(CreateMappings create_edit_mappings.cpp ${LIBGRAPH_ROOT}/include/libGraph.h ${GEDLIB_ROOT}/src/env/ged_env.hpp
        src/include.h)
add_executable(CreatePaths create_edit_paths.cpp ${LIBGRAPH_ROOT}/include/libGraph.h
        src/include.h  ${GEDLIB_ROOT}/src/env/ged_env.hpp
        src/include.h)
add_executable(AnalyzePaths analyze_edit_path_graphs.cpp ${LIBGRAPH_ROOT}/include/libGraph.h
        src/include.h)
add_executable(AnalyzeMappings analyze_mappings.cpp ${LIBGRAPH_ROOT}/include/libGraph.h ${GEDLIB_ROOT}/src/env/ged_env.hpp
        src/include.h)
add_executable(Test test.cpp ${LIBGRAPH_ROOT}/include/libGraph.h ${GEDLIB_ROOT}/src/env/ged_env.hpp
        src/include.h)
add_executable(TestLINUX test_linux.cpp)
add_executable(ConvertPrecomputed convert_precomputed.cpp ${LIBGRAPH_ROOT}/include/libGraph.h src/include.h)

target_link_libraries(CreateMappings libsvm.so libnomad.so libdoublefann.so.2 gurobi)
target_link_libraries(CreatePaths libsvm.so libnomad.so libdoublefann.so.2 gurobi)
