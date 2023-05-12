cmake_minimum_required(VERSION 3.1)
project( main )

set(paths "${OPENNI2_DIR}")
list(APPEND paths $ENV{OPENNI2_INCLUDE} $ENV{OPENNI2_LIB} $ENV{OPENNI2_REDIST})

# From SDK
find_path(OPENNI2_INCLUDE "OpenNI.h"
    PATHS ${paths}
    PATH_SUFFIXES "Include"
    NO_DEFAULT_PATH)
find_library(OPENNI2_LIBRARY "OpenNI2"
    PATHS ${paths}
    PATH_SUFFIXES "Redist" "Lib"
    NO_DEFAULT_PATH)

if(OPENNI2_LIBRARY AND OPENNI2_INCLUDE)
    set(HAVE_OPENNI2 TRUE)
    set(OPENNI2_INCLUDE_DIRS "${OPENNI2_INCLUDE}")
    set(OPENNI2_LIBRARIES "${OPENNI2_LIBRARY}")

    file(STRINGS "${OPENNI2_INCLUDE_DIRS}/OniVersion.h" ver_strings REGEX "#define +ONI_VERSION_(MAJOR|MINOR|MAINTENANCE|BUILD).*")
    string(REGEX REPLACE ".*ONI_VERSION_MAJOR[^0-9]+([0-9]+).*" "\\1" ver_major "${ver_strings}")
    string(REGEX REPLACE ".*ONI_VERSION_MINOR[^0-9]+([0-9]+).*" "\\1" ver_minor "${ver_strings}")
    string(REGEX REPLACE ".*ONI_VERSION_MAINTENANCE[^0-9]+([0-9]+).*" "\\1" ver_maint "${ver_strings}")
    set(OPENNI2_VERSION "${ver_major}.${ver_minor}.${ver_maint}")  # informational

    message("Found OpenNI2: " "${OPENNI2_VERSION}")
  endif()

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${OPENNI2_INCLUDE_DIRS} )
include_directories($ENV{INCGL})
include_directories($ENV{INLIBUSB})

add_library(cmd cmd.cpp)

target_include_directories(cmd PUBLIC $ENV{INLIBUSB})

add_executable( main cap.cpp )
target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries(main ${OPENNI2_LIBRARIES})
target_link_libraries(main cmd)
target_link_libraries(main usb-1.0)
