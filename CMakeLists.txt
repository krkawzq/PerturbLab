cmake_minimum_required(VERSION 3.18)
project(perturblab_statistics LANGUAGES CXX)

# -----------------------------
# Basic Settings
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------------------
# Highway (as subdirectory)
# -----------------------------
set(HIGHWAY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/lib/highway")
if(NOT EXISTS "${HIGHWAY_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Highway not found at ${HIGHWAY_DIR}. Run scripts/setup_cpp_deps.sh to install.")
endif()

set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
# Force static linking for Highway (especially for sort functions)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
add_subdirectory("${HIGHWAY_DIR}" highway_build EXCLUDE_FROM_ALL)
message(STATUS "Highway library found at ${HIGHWAY_DIR} (static linking)")

# -----------------------------
# OpenMP (optional; cross-platform handling)
# -----------------------------
find_package(OpenMP QUIET)

# macOS: fallback to Homebrew libomp if OpenMP::OpenMP_CXX not found
if(APPLE AND NOT OpenMP_CXX_FOUND)
    foreach(_omp_root "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
        if(EXISTS "${_omp_root}/lib/libomp.dylib")
            message(STATUS "Using fallback libomp at ${_omp_root}")
            add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
            
            # Check compiler flags
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-fopenmp" COMPILER_SUPPORTS_FOPENMP)
            check_cxx_compiler_flag("-Xpreprocessor -fopenmp" COMPILER_SUPPORTS_XPREPROCESSOR_FOPENMP)
            
            if(COMPILER_SUPPORTS_FOPENMP)
                set(_omp_compile_flags "-fopenmp")
                set(_omp_link_flags "-fopenmp")
            elseif(COMPILER_SUPPORTS_XPREPROCESSOR_FOPENMP)
                set(_omp_compile_flags "-Xpreprocessor;-fopenmp")
                set(_omp_link_flags "-lomp")
            else()
                message(WARNING "Compiler does not support OpenMP flags, using library-only approach")
                set(_omp_compile_flags "")
                set(_omp_link_flags "")
            endif()
            
            set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
                IMPORTED_LOCATION "${_omp_root}/lib/libomp.dylib"
                INTERFACE_INCLUDE_DIRECTORIES "${_omp_root}/include"
                INTERFACE_COMPILE_OPTIONS "${_omp_compile_flags}"
                INTERFACE_LINK_OPTIONS "${_omp_link_flags}"
            )
            set(OpenMP_CXX_FOUND TRUE)
            set(_OMP_RPATH "${_omp_root}/lib")
            break()
        endif()
    endforeach()
endif()

# -----------------------------
# Shared Library Target
# -----------------------------
set(CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/perturblab/kernels/statistics/backends/cpp/csrc")

add_library(mwu_kernel SHARED
    ${CSRC_DIR}/mannwhitneyu.cpp
    ${CSRC_DIR}/ttest.cpp
    ${CSRC_DIR}/group_ops.cpp
    ${CSRC_DIR}/sparse_clipped_moments.cpp
    ${CSRC_DIR}/sparse_mean_var.cpp
    ${CSRC_DIR}/clip_matrix.cpp
    ${CSRC_DIR}/polynomial_fit.cpp
    ${CSRC_DIR}/normalization.cpp
    ${CSRC_DIR}/scale.cpp
    ${CSRC_DIR}/capi_wrapper.cpp
)

target_compile_features(mwu_kernel PRIVATE cxx_std_17)

# Platform-specific compile options
if(MSVC)
    target_compile_options(mwu_kernel PRIVATE /O2 /permissive- /EHsc /Zc:preprocessor /bigobj)
    target_compile_definitions(mwu_kernel PRIVATE NOMINMAX)
else()
    target_compile_options(mwu_kernel PRIVATE -O3)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(mwu_kernel PRIVATE -Wno-deprecated-declarations)
    endif()
endif()

# Include directories
# Note: Headers are in csrc/include/, but code uses #include "xxx.hpp" (without include/ prefix)
# So we add csrc/include to the include path
target_include_directories(mwu_kernel PRIVATE
    ${CSRC_DIR}/include
)

# Link Highway (static linking for sort functions)
# hwy_contrib contains sort functions - ensure static linking
target_link_libraries(mwu_kernel PRIVATE hwy hwy_contrib)

# Platform-specific OpenMP handling
if(OpenMP_CXX_FOUND)
    target_link_libraries(mwu_kernel PRIVATE OpenMP::OpenMP_CXX)
    # Note: OpenMP::OpenMP_CXX target already defines _OPENMP, don't redefine it
    # to avoid compiler warnings
    message(STATUS "OpenMP enabled")
else()
    message(WARNING "OpenMP not found, parallel execution will be disabled")
    target_compile_definitions(mwu_kernel PRIVATE NO_OPENMP)
endif()

# macOS: Set rpath for libomp
if(APPLE)
    if(DEFINED _OMP_RPATH)
        set_target_properties(mwu_kernel PROPERTIES
            BUILD_RPATH "${_OMP_RPATH}"
            INSTALL_RPATH "${_OMP_RPATH}"
        )
    endif()
endif()

# -----------------------------
# Installation
# -----------------------------
# Install to perturblab/kernels/statistics/backends/cpp/
set(INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/perturblab/kernels/statistics/backends/cpp")

install(TARGETS mwu_kernel
    LIBRARY DESTINATION ${INSTALL_DIR}   # Unix-like .so
    RUNTIME DESTINATION ${INSTALL_DIR}   # Windows .dll
    ARCHIVE DESTINATION ${INSTALL_DIR}   # Static lib (if any)
)

# -----------------------------
# Build Summary
# -----------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Build Configuration Summary")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Highway: ${HIGHWAY_DIR}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "Install directory: ${INSTALL_DIR}")
message(STATUS "========================================")
message(STATUS "")
