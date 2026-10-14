cmake_minimum_required(VERSION 3.24)

project(SpectralElements
    VERSION 0.1.0
    DESCRIPTION "Spectral element method solver"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SEM_BUILD_TESTS "Build unit tests" ON)

# Python нужен только во время работы программы (sem::io::Figure::render),
# не при сборке. Путь можно переопределить: -DSEM_PYTHON_EXECUTABLE=...
find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    set(SEM_DEFAULT_PYTHON "${Python3_EXECUTABLE}")
else ()
    set(SEM_DEFAULT_PYTHON "python")
endif ()
set(SEM_PYTHON_EXECUTABLE "${SEM_DEFAULT_PYTHON}" CACHE STRING
    "Python interpreter with matplotlib used to draw plots")
set(SEM_PLOT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tools/plot.py")
configure_file(src/io/plot_config.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/sem/io/plot_config.hpp @ONLY)

add_library(sem
    src/core/dense_matrix.cpp
    src/core/linear_solve.cpp
    src/basis/legendre.cpp
    src/basis/quadrature.cpp
    src/basis/lagrange_basis.cpp
    src/basis/spectral_basis.cpp
    src/io/plot.cpp
    src/analysis/error_norms.cpp
)
add_library(sem::sem ALIAS sem)

target_include_directories(sem
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
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
