# /* Copyright (C) 2024 PointGrab - All Rights Reserved. */
# /* This file was ported to work on Alif Semiconductor Ensemble family of devices. */
#
#/* Copyright (C) 2022 Alif Semiconductor - All Rights Reserved.
# * Use, distribution and modification of this code is permitted under the
# * terms stated in the Alif Semiconductor Software License Agreement
# *
# * You should have received a copy of the Alif Semiconductor Software
# * License Agreement with this file. If not, please write to:
# * contact@alifsemi.com, or visit: https://alifsemi.com/license
# *
# */

#----------------------------------------------------------------------------
#  Copyright (c) 2022 Arm Limited. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#----------------------------------------------------------------------------

#########################################################
# Environmental sensor BME68x library                                         #
#########################################################

cmake_minimum_required(VERSION 3.16.3)

project(envsns_component
    DESCRIPTION     "Environmental sensor BME68x library"
    LANGUAGES       C CXX ASM)

# Add top level interface library
set(BME68X_IFACE_TARGET envsns_iface)
add_library(${BME68X_IFACE_TARGET} INTERFACE)
target_include_directories(${BME68X_IFACE_TARGET}
    INTERFACE
    include
    source
    source/ensemble
    source/ensemble/include)

# Create static library for Ensemble data
set(BME68X_ENSEMBLE_COMPONENT_TARGET envsns_ensemble)
add_library(${BME68X_ENSEMBLE_COMPONENT_TARGET} STATIC)

## Component sources
target_sources(${BME68X_ENSEMBLE_COMPONENT_TARGET}
    PRIVATE
    source/ensemble/bme68x.c
    source/ensemble/bme68x_common.c)

# Ensemble TARGET_BOARD needs to be set
target_compile_definitions(${BME68X_ENSEMBLE_COMPONENT_TARGET}
    PRIVATE
    TARGET_BOARD=BOARD_${TARGET_BOARD})

## Logging utilities:
if (NOT TARGET log)
    if (NOT DEFINED LOG_PROJECT_DIR)
        message(FATAL_ERROR "LOG_PROJECT_DIR needs to be defined.")
    endif()
    add_subdirectory(${LOG_PROJECT_DIR} ${CMAKE_BINARY_DIR}/log)
endif()

## Add dependencies
target_link_libraries(${BME68X_ENSEMBLE_COMPONENT_TARGET} PUBLIC
    ${BME68X_IFACE_TARGET}
    log
    cmsis_ensemble
    rte_components)

target_link_libraries(${BME68X_ENSEMBLE_COMPONENT_TARGET} PRIVATE
    ${BME68X_IFACE_TARGET}
    arm_math)

# Display status
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "*******************************************************")
message(STATUS "Library                                : " ${BME68X_ENSEMBLE_COMPONENT_TARGET})
message(STATUS "*******************************************************")
