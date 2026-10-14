cmake_minimum_required(VERSION 3.24)

project(SpectralElements
    VERSION 0.1.0
    DESCRIPTION "Spectral element method solver"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SEM_BUILD_TESTS "Build unit tests" ON)

add_library(sem
    src/core/dense_matrix.cpp
    src/core/linear_solve.cpp
    src/basis/legendre.cpp
    src/basis/quadrature.cpp
    src/basis/lagrange_basis.cpp
    src/basis/spectral_basis.cpp
)
add_library(sem::sem ALIAS sem)

target_include_directories(sem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(sem PUBLIC cxx_std_20)

if (MSVC)
    target_compile_options(sem PRIVATE /W4 /permissive-)
else ()
    target_compile_options(sem PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
endif ()

if (SEM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
