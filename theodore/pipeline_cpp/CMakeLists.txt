cmake_minimum_required(VERSION 3.16)
project(Yolov8CPPInference C CXX)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(Yolov8CPPInference main.cpp IModels.h YOLOv8.h YOLOv8.cpp PANMobilenetV2.h PANMobilenetV2.cpp)
target_link_libraries(Yolov8CPPInference ${OpenCV_LIBS})
