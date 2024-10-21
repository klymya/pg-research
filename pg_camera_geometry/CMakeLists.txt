cmake_minimum_required(VERSION 3.21.0)

project(pg_geometry)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add subdirectories
add_subdirectory(camera_geometry)
add_subdirectory(test)
