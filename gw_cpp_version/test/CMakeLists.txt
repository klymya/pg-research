cmake_minimum_required(VERSION 3.21.0)

set(CMAKE_CXX_STANDARD 17)

project(test_me)

# GTest
enable_testing()
find_package(GTest CONFIG REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Unit Tests
# Add test cpp file
add_executable(runUnitTests
    Common.hpp
    TestXYStorage.cpp
    TestPreProcessingUtils.cpp
    TestPreProcessor.cpp
    TestAreaOfInterest.cpp
    TestAggregator.cpp
)

# Link the test executable with GTest and the library target
target_link_libraries(runUnitTests PRIVATE
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main pg_library
)

add_test(runUnitTests runUnitTests)