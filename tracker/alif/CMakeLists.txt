cmake_minimum_required(VERSION 3.10)
project(MatrixComputation)

set(CMAKE_CXX_STANDARD 11)

set(EIGEN_INCLUDE_DIR "eigen-3.3.9")
include_directories(${EIGEN_INCLUDE_DIR} include)
file(GLOB SOURCES "src/*.cpp")

add_executable(MatrixComputation ${SOURCES} main.cpp)