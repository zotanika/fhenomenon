cmake_minimum_required(VERSION 3.19)
project(FhenomenonWorkspace LANGUAGES CXX)
# Keep root-based CMake builds; cpp/ is also a standalone CMake project.
enable_testing()
add_subdirectory(cpp)
