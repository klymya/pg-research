cmake_minimum_required(VERSION 3.21.0)


set(CMAKE_CXX_STANDARD 17)

project(test_me)

# GTest
enable_testing()
find_package(GTest CONFIG REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/camera_geometry/include)  # Add this line

# Unit Tests
# Add test cpp file
add_executable( runUnitTests
    common.hpp
    TestGeometry.cpp
    TestFisheyeCamera.cpp
    TestDetectionArea.cpp
)
target_link_libraries(runUnitTests PRIVATE
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
    CameraGeometry
)
add_test( runUnitTests runUnitTests )