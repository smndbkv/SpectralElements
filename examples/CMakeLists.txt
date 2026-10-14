add_executable(example_plot_basis plot_basis.cpp)
target_link_libraries(example_plot_basis PRIVATE sem::sem)

# Каждый файл с main — отдельная программа; общий код живёт в библиотеке sem.
add_executable(example_approximation_1d approximation_1d.cpp)
target_link_libraries(example_approximation_1d PRIVATE sem::sem)

add_executable(example_differential_equation_1d diferential_equation_1d.cpp)
target_link_libraries(example_differential_equation_1d PRIVATE sem::sem)
