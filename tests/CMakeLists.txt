include(FetchContent)

FetchContent_Declare(Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.8.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(sem_tests
    test_dense_matrix.cpp
    test_linear_solve.cpp
    test_legendre.cpp
    test_quadrature.cpp
    test_lagrange_basis.cpp
    test_spectral_basis.cpp
    test_tensor_basis.cpp
)
target_link_libraries(sem_tests PRIVATE sem::sem Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(sem_tests)
