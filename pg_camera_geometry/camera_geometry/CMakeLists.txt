cmake_minimum_required(VERSION 3.21.0)

set(CMAKE_CXX_STANDARD 17)

project(CameraGeometry)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_library(CameraGeometry STATIC ${SOURCES})
