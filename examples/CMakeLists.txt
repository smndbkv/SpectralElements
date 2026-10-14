add_executable(example_plot_basis plot_basis.cpp)
target_link_libraries(example_plot_basis PRIVATE sem::sem)

# Каждый файл с main — отдельная программа; общий код живёт в библиотеке sem.
add_executable(example_approximation_1d approximation_1d.cpp)
target_link_libraries(example_approximation_1d PRIVATE sem::sem)

add_executable(example_approximation_4d approximation_4d.cpp)
target_link_libraries(example_approximation_4d PRIVATE sem::sem)

# Картинки для README: примеры запускаются без окон прямо в папке docs/images.
#   cmake --build --preset release --target docs_images
set(SEM_DOCS_IMAGES_DIR ${PROJECT_SOURCE_DIR}/docs/images)
file(MAKE_DIRECTORY ${SEM_DOCS_IMAGES_DIR})

# MinGW: программам нужны DLL компилятора (libstdc++ и др.), поэтому при запуске
# из CMake каталог компилятора добавляется в PATH (cmake -E env --modify, CMake >= 3.25).
set(SEM_RUN_ENV)
if (MINGW)
    get_filename_component(SEM_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(SEM_RUN_ENV ${CMAKE_COMMAND} -E env --modify "PATH=path_list_prepend:${SEM_COMPILER_DIR}" --)
endif ()

add_custom_target(docs_images
    COMMAND ${SEM_RUN_ENV} $<TARGET_FILE:example_differential_equation_1d> --no-show
    COMMAND ${SEM_RUN_ENV} $<TARGET_FILE:example_approximation_4d> --no-show
    WORKING_DIRECTORY ${SEM_DOCS_IMAGES_DIR}
    COMMENT "Regenerating README images in docs/images"
    VERBATIM)
add_dependencies(docs_images example_differential_equation_1d example_approximation_4d)

add_executable(example_differential_equation_1d diferential_equation_1d.cpp)
target_link_libraries(example_differential_equation_1d PRIVATE sem::sem)
