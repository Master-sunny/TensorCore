cmake_minimum_required(VERSION 3.20)
project(tensor_core_systemc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SystemCLanguage CONFIG REQUIRED)

add_executable(tensor_dot_tests
    tests/sc_main.cpp
    tests/reference_model.cpp
)
target_include_directories(tensor_dot_tests PRIVATE include tests)
target_link_libraries(tensor_dot_tests PRIVATE SystemC::systemc)
target_compile_options(tensor_dot_tests PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME tensor_dot_random COMMAND tensor_dot_tests)
