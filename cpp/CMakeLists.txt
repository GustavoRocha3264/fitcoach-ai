cmake_minimum_required(VERSION 3.18)
project(fitcoach_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the Python interpreter from the active conda env so pybind11 finds the
# matching ABI. Caller scripts may pass -DPython3_EXECUTABLE for clarity.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fitcoach_cpp angles.cpp)

target_compile_options(fitcoach_cpp PRIVATE -O3 -Wall -Wextra)
