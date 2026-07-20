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
# WDTM library                                          #
#########################################################

cmake_minimum_required(VERSION 3.16.3)

project(wdtm_component
    DESCRIPTION     "WDTM library"
    LANGUAGES       C CXX ASM)

# Add top level interface library
set(WDTM_TPL5010 wdtm_tpl5010)
add_library(${WDTM_TPL5010} INTERFACE)
target_include_directories(${WDTM_TPL5010}
    INTERFACE
    include
    source)

# Create static library for Ensemble data
set(TPL5010_ENSEMBLE_COMPONENT_TARGET wdtm_ensemble)
add_library(${TPL5010_ENSEMBLE_COMPONENT_TARGET} STATIC)

## Component sources
target_sources(${TPL5010_ENSEMBLE_COMPONENT_TARGET}
    PRIVATE
    source/pg_wdtm.c)

# Ensemble TARGET_BOARD needs to be set
target_compile_definitions(${TPL5010_ENSEMBLE_COMPONENT_TARGET}
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
target_link_libraries(${TPL5010_ENSEMBLE_COMPONENT_TARGET} PUBLIC
    ${WDTM_TPL5010}
    log
    cmsis_ensemble
    rte_components)

# Display status
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "*******************************************************")
message(STATUS "Library                                : " ${TPL5010_ENSEMBLE_COMPONENT_TARGET})
message(STATUS "*******************************************************")
