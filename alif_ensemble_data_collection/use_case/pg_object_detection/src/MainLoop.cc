/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "hal.h"                      /* Brings in platform definitions. */
#include "InputFiles.hpp"             /* For input images. */
#include "Yolov8Model.hpp"       /* Model class for running inference. */
#include "DetectionHeadModel.hpp"
#include "UseCaseHandler.hpp"         /* Handlers for different user options. */
#include "UseCaseCommonUtils.hpp"     /* Utils functions. */
#include "log_macros.h"             /* Logging functions */
#include "BufAttributes.hpp"        /* Buffer attributes to be applied */
#include "services.h"
#include "TestEthernet.h"
#include "TestMicrophone.h"
#include "TestEnvSensor.h"
#include "TestEeprom.h"
#include "TestWDTM.h"
#include "image_processing.h"
#include "jpeg.h"
#include "TestPir.h"
#include "pg_wdtm.h"
#include "TestLED.h"

#ifndef HEAD_ACTIVATION_BUF_SZ
    #warning  "HEAD_ACTIVATION_BUF_SZ needs to be defined. Using default value"
    #define HEAD_ACTIVATION_BUF_SZ       0x00C0000
#endif  /* HEAD_ACTIVATION_BUF_SZ */


namespace arm {
namespace app {
    namespace pg_object_detection {
        static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();

        namespace detection_head {
            static uint8_t tensorArena[HEAD_ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
            extern uint8_t* GetModelPointer();
            extern size_t GetModelLen();
        } /* namespace detection_head */
    } /* namespace pg_object_detection */
} /* namespace app */
} /* namespace arm */

static void DisplayDetectionMenu()
{
    printf("\r\n\n");
    printf("User input required\r\n");
    printf("Enter option number from:\r\n\n");
    printf("  %u. Run detection on next ifm\r\n", common::MENU_OPT_RUN_INF_NEXT);
    printf("  %u. Run detection ifm at chosen index\r\n", common::MENU_OPT_RUN_INF_CHOSEN);
    printf("  %u. Run detection on all ifm\r\n", common::MENU_OPT_RUN_INF_ALL);
    printf("  %u. Show NN model info\r\n", common::MENU_OPT_SHOW_MODEL_INFO);
    printf("  %u. List ifm\r\n", common::MENU_OPT_LIST_IFM);
    printf("  %u. Run tracker\r\n", common::MENU_OPT_TRACKER);
    printf("  %u. Go to STOP3 power mode\r\n", common::MENU_OPT_STOP);
    printf("  %u. Test Ethernet\r\n", common::MENU_OPT_TEST_ETHERNET);
    printf("  %u. Test Microphone\r\n", common::MENU_OPT_TEST_MICROPHONE);
    printf("  %u. Test Env Sensor\r\n", common::MENU_OPT_TEST_ENV_SENSOR);
    printf("  %u. Test EEPROM\r\n", common::MENU_OPT_TEST_EEPROM);
    printf("  %u. Test PIR\r\n", common::MENU_OPT_TEST_PIR);
    printf("  %u. Test WDTM\r\n", common::MENU_OPT_TEST_WDTM);
    printf("  %u. Test LED\r\n\n", common::MENU_OPT_TEST_LED);
    printf("  Choice: ");
    fflush(stdout);
}


void ResultCallbackFunction(const uint8_t* imageData, int width, int height, const std::vector<arm::app::pg_object_detection::Detection>& detections) {
    printf("On object detection done\r\n");
    jpeg_send_through_uart(imageData, width, height, &detections);
}


static void main_loop_dev(arm::app::ApplicationContext& caseContext)
{
    /* Loop. */
    bool executionSuccessful = true;
    constexpr bool bUseMenu = NUMBER_OF_FILES >= 1 ? true : false;

    /* Loop. */
    do {
        int menuOption = common::MENU_OPT_TEST_WDTM;
        if (bUseMenu) {
            DisplayDetectionMenu();
            menuOption = arm::app::ReadUserInputAsInt();
            printf("\r\n");
        }

        switch (menuOption) {
            case common::MENU_OPT_RUN_INF_NEXT:
                executionSuccessful = PGObjectDetectionHandler(caseContext, caseContext.Get<uint32_t>("imgIndex"), false);
                break;
            case common::MENU_OPT_RUN_INF_CHOSEN: {
                printf("    Enter the image index [0, %d]: ", NUMBER_OF_FILES-1);
                fflush(stdout);
                auto imgIndex = static_cast<uint32_t>(arm::app::ReadUserInputAsInt());
                executionSuccessful = PGObjectDetectionHandler(caseContext, imgIndex, false);
                break;
            }
            case common::MENU_OPT_RUN_INF_ALL:
                executionSuccessful = PGObjectDetectionHandler(caseContext, caseContext.Get<uint32_t>("imgIndex"), true);
                break;
            case common::MENU_OPT_SHOW_MODEL_INFO: {
                auto& detectionHeadModel = caseContext.Get<arm::app::DetectionHeadModel&>("detectionHeadModel");
                auto& model = caseContext.Get<arm::app::Yolov8Model&>("model");

                executionSuccessful = model.ShowModelInfoHandler() && detectionHeadModel.ShowModelInfoHandler();
                break;
            }
            case common::MENU_OPT_LIST_IFM:
                executionSuccessful = ListFilesHandler(caseContext);
                break;
            case common::MENU_OPT_TRACKER:
                executionSuccessful = arm::app::PGTrackerHandler();
                break;
            case common::MENU_OPT_STOP: {
                const uint32_t wakeup_in_sec = 5;

                printf("Go to STOP mode, and wakeup in %lu sec\r\n", wakeup_in_sec);
                services_execute_aipm_stop_mode(wakeup_in_sec);
                executionSuccessful = true;
                }
                break;
            case common::MENU_OPT_TEST_ETHERNET:
                if (test_ethernet()) {
                    printf("Test FAILED!\r\n");
                } else {
                    printf("Test PASSED!\r\n");
                }
                executionSuccessful = true;
                break;
            case common::MENU_OPT_TEST_MICROPHONE:
                if (test_microphone()) {
                    printf("Test FAILED!\r\n");
                } else {
                    printf("Test PASSED!\r\n");
                }
                executionSuccessful = true;
                break;
            case common::MENU_OPT_TEST_ENV_SENSOR:
                if (test_envsensor()) {
                    printf("Test FAILED!\r\n");
                } else {
                    printf("Test PASSED!\r\n");
                }
                executionSuccessful = true;
                break;
            case common::MENU_OPT_TEST_EEPROM:
                if (test_eeprom()) {
                    printf("Test FAILED!\r\n");
                } else {
                    printf("Test PASSED!\r\n");
                }
                executionSuccessful = true;
                break;
            case common::MENU_OPT_TEST_PIR:
                if (test_pir()) {
                    printf("Test FAILED!\r\n");
                } else {
                    printf("Test PASSED!\r\n");
                }
                executionSuccessful = true;
                break;

            case common::MENU_OPT_TEST_WDTM:
            {
                printf("Testing WDTM... please wait 30 seconds\r\n");
                if(testWDTM() == 0) {
                    printf("Test PASSED!\r\n");
                } else {
                    printf("Test FAILED!\r\n");
                }

                executionSuccessful = true;
                break;
            }

            case common::MENU_OPT_TEST_LED:
            {
                printf("Testing LED... check the board if LEDs light up\r\n");
                testLED(1);
            }

            /* commands to communicate with the test application */
            case common::TEST_CAMERA_CMD:
            {
                int err = hal_image_init();

                if (0 == err) {
                    const uint8_t* currImage = hal_get_image_data(CIMAGE_X, CIMAGE_Y);
                    if (currImage) {
                        jpeg_send_through_uart(currImage, CIMAGE_X, CIMAGE_Y, nullptr);
                    }
                }
            }
            break;
            case common::TEST_MICROPHONE_CMD:
                test_microphone_send();
            break;
            case common::TEST_EEPROM_CMD:
                test_eeprom_send();
            break;
            case common::TEST_ENVSENSOR_CMD:
                test_envsensor_send();
            break;
            case common::TEST_INFERENCE_CMD:
            {
                executionSuccessful = PGObjectDetectionHandler(caseContext, caseContext.Get<uint32_t>("imgIndex"), false, &ResultCallbackFunction);
                break;
            }
            case common::TEST_PIR_CMD:
                test_pir_send();
            case common::TEST_WD_CMD:
                test_wd_send();
            default:
                printf("Incorrect choice, try again.");
                break;
        }
        fflush(stdout);
    } while (executionSuccessful && bUseMenu);
}

static void main_loop_rel(arm::app::ApplicationContext& caseContext)
{
    const uint32_t wakeup_in_sec = 5;
    PGObjectDetectionHandler(caseContext, 0, false);

    printf("Go to STOP mode, and wakeup in %lu sec\r\n", wakeup_in_sec);
    services_execute_aipm_stop_mode(wakeup_in_sec);
}

void main_loop()
{
    if (!arm::app::ObjectDetectionInit()) {
        printf_err("Failed to initialise the camera\r\n");
    }

    arm::app::Yolov8Model model;  /* Model wrapper object. */

    /* Load the model. */
    if (!model.Init(arm::app::pg_object_detection::tensorArena,
                    sizeof(arm::app::pg_object_detection::tensorArena),
                    arm::app::pg_object_detection::GetModelPointer(),
                    arm::app::pg_object_detection::GetModelLen())) {
        printf_err("Failed to initialise model\r\n");
        return;
    }

    arm::app::DetectionHeadModel detectionHeadModel;  /* Model wrapper object. */

    /* Load the model. */
    if (!detectionHeadModel.Init(arm::app::pg_object_detection::detection_head::tensorArena,
                    sizeof(arm::app::pg_object_detection::detection_head::tensorArena),
                    arm::app::pg_object_detection::detection_head::GetModelPointer(),
                    arm::app::pg_object_detection::detection_head::GetModelLen())) {
        printf_err("Failed to initialise detectionHeadModel\r\n");
        return;
    }

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{"pg_object_detection"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::Yolov8Model&>("model", model);
    caseContext.Set<arm::app::DetectionHeadModel&>("detectionHeadModel", detectionHeadModel);
    caseContext.Set<uint32_t>("imgIndex", 0);

#ifdef PROJECT_MENU_CFG
    main_loop_dev(caseContext);
#else
    main_loop_rel(caseContext);
#endif

    printf_err("Main loop terminated.\r\n");
}
