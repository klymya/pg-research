# SPDX-License-Identifier: Apache-2.0
#
# Sign-of-Life pipeline (C++14 reference). Host build with the native GCC/Clang
# toolchain. The algorithm sources use only C++14 + the standard library so they
# port to the Alif ML embedded evaluation kit; the host I/O layer (stb + dirent)
# is isolated in ImageIo.cpp.

cmake_minimum_required(VERSION 3.21)

project(sign_of_life_cpp LANGUAGES CXX)

# --- Alif C++ standard: C++14, strict ISO (no GNU extensions) ----------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(sign_of_life
    src/Main.cpp
    src/Pipeline.cpp
    src/BackgroundModel.cpp
    src/ShadowSuppressor.cpp
    src/NativeOps.cpp
    src/ImageIo.cpp
)

target_include_directories(sign_of_life PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# Vendored single-header libraries: mark SYSTEM so their warnings stay quiet.
target_include_directories(sign_of_life SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Warning set mirrors the eval kit (warnings on; -Werror not enforced).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sign_of_life PRIVATE
        -Wall -Wextra -Wshadow -Wsign-compare
    )
endif()
