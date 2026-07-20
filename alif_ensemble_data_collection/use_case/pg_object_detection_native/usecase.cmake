#----------------------------------------------------------------------------
#  SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
# Append the API to use for this use case
list(APPEND ${use_case}_API_LIST "pg_object_detection")

set(${use_case}_IMAGE_W 320)
set(${use_case}_IMAGE_H 240)

USER_OPTION(
    ${use_case}_FILE_PATH
    "Directory with custom input files, or path to a single input file, to use in the evaluation application."
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/${use_case}/samples/img_${${use_case}_IMAGE_W}.bmp
    PATH_OR_FILE
)

# Generate input files
generate_images_code("${${use_case}_FILE_PATH}"
                     ${SRC_GEN_DIR}
                     ${INC_GEN_DIR}
                     "${${use_case}_IMAGE_W}"
                     "${${use_case}_IMAGE_H}")


set(DEFAULT_BACKBONE_BUF_SZ 0x0210000) # 2162688 B
set(DEFAULT_HEAD_BUF_SZ 0x0110000) # 1114112 B


USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model backbone"
    ${DEFAULT_BACKBONE_BUF_SZ}
    STRING)

USER_OPTION(${use_case}_HEAD_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model head"
    ${DEFAULT_HEAD_BUF_SZ}
    STRING)

set(${use_case}_COMPILE_DEFS
    "HEAD_ACTIVATION_BUF_SZ=${${use_case}_HEAD_ACTIVATION_BUF_SZ}")


if (ETHOS_U_NPU_ENABLED)
    set(DEFAULT_BACKBONE_PATH      ${DEFAULT_MODEL_DIR}/backbone_${ETHOS_U_NPU_CONFIG_ID}.tflite)
else()
    set(DEFAULT_BACKBONE_PATH      ${DEFAULT_MODEL_DIR}/backbone.tflite)
endif()

if (ETHOS_U_NPU_ENABLED)
    set(DEFAULT_HEAD_MODEL_PATH      ${DEFAULT_MODEL_DIR}/head_${ETHOS_U_NPU_CONFIG_ID}.tflite)
else()
    set(DEFAULT_HEAD_MODEL_PATH      ${DEFAULT_MODEL_DIR}/head.tflite)
endif()

USER_OPTION(${use_case}_BACKBONE_TFLITE_PATH "NN models file to be used in the evaluation application. Model files must be in tflite format."
    ${DEFAULT_BACKBONE_PATH}
    FILEPATH
    )

USER_OPTION(${use_case}_HEAD_TFLITE_PATH "NN models file to be used in the evaluation application. Model files must be in tflite format."
    ${DEFAULT_HEAD_MODEL_PATH}
    FILEPATH
    )

# Generate model file
generate_tflite_code(
    MODEL_PATH ${${use_case}_BACKBONE_TFLITE_PATH}
    DESTINATION ${SRC_GEN_DIR}
    NAMESPACE   "arm" "app" "pg_object_detection")

generate_tflite_code(
    MODEL_PATH ${${use_case}_HEAD_TFLITE_PATH}
    DESTINATION ${SRC_GEN_DIR}
    NAMESPACE   "arm" "app" "pg_object_detection" "detection_head")
