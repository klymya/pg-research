cmake_minimum_required(VERSION 3.10)
project(CLAHE_Test_Project LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(clahe_test
    main.cc
    img_320.cc
    clahe.c
    rgb2lab.c
)

target_include_directories(clahe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(clahe_test m)
