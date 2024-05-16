cmake_minimum_required(VERSION 3.16)
project(main C CXX)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp IModels.h YOLOv8.h YOLOv8.cpp PANMobilenetV2.h PANMobilenetV2.cpp mask2ConvexHul.h)
target_link_libraries(main ${OpenCV_LIBS})

