cmake_minimum_required(VERSION 3.16)
project(main C CXX)

set(OpenCV_DIR "/opt/cp/6.0/sysroots/aarch64-pg-linux/usr/lib/cmake/opencv4/")
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(main yolov8/main_int8.cpp)
target_link_libraries(main ${OpenCV_LIBS} tensorflow-lite pthread)
