cmake_minimum_required(VERSION 3.21.0)

project(pg_gateway)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add the CameraGeometry library as a subdirectory
add_subdirectory(../pg_camera_geometry/camera_geometry ${CMAKE_BINARY_DIR}/camera_geometry)

# Include directories for the CameraGeometry library and your project
include_directories(
    ../pg_camera_geometry/camera_geometry/include
    ${CMAKE_SOURCE_DIR}/include
)

# Add all source files from the src directory
file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Create a library target for the shared code
add_library(pg_library ${SRC_FILES})

# Link the CameraGeometry library and SQLite3 to the library target
target_link_libraries(pg_library PRIVATE CameraGeometry sqlite3)

# Add the executable
add_executable(pg_gateway main.cpp)

# Link the library target to the executable
target_link_libraries(pg_gateway PRIVATE pg_library)

# Add the test subdirectory
add_subdirectory(test)