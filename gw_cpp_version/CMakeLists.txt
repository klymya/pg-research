cmake_minimum_required(VERSION 3.21.0)

project(pg_gateway)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

find_package(eclipse-paho-mqtt-c CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(unofficial-inih CONFIG REQUIRED)

# Add the CameraGeometry library as a subdirectory
add_subdirectory(../pg_camera_geometry/camera_geometry ${CMAKE_BINARY_DIR}/camera_geometry)

# Include directories for the CameraGeometry library and your project
include_directories(
    ../pg_camera_geometry/camera_geometry/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/nlohmann
)

# Add all source files from the src directory
file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Generate C++ source files from the .proto files
set(PROTO_FILES ${CMAKE_SOURCE_DIR}/proto/sensor_payload.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Include the generated protobuf headers
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Create a library target for the shared code
add_library(pg_library ${SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})

# Link the CameraGeometry library and SQLite3 to the library target
target_link_libraries(pg_library PRIVATE
    CameraGeometry
    sqlite3
    eclipse-paho-mqtt-c::paho-mqtt3a-static
    eclipse-paho-mqtt-c::paho-mqtt3c-static
    eclipse-paho-mqtt-c::paho-mqtt3as-static
    eclipse-paho-mqtt-c::paho-mqtt3cs-static
    protobuf::libprotoc
    protobuf::libprotobuf
    protobuf::libprotobuf-lite
    unofficial::inih::inireader
)

# Add the protobuf include directories to the target
target_include_directories(pg_library PRIVATE ${Protobuf_INCLUDE_DIRS})

# Add the executable
add_executable(pg_gateway gw_entrypoint.cpp)

# Link the library target to the executable
target_link_libraries(pg_gateway PRIVATE pg_library)

# Add the protobuf include directories to the executable
target_include_directories(pg_gateway PRIVATE ${Protobuf_INCLUDE_DIRS})

# Add the test subdirectory
add_subdirectory(test)
