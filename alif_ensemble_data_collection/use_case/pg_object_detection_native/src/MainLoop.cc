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
#include "image_processing.h"
#include "jpeg.h"

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
    printf("  Choice: ");
    fflush(stdout);
}

static void main_loop_dev(arm::app::ApplicationContext& caseContext)
{
    /* Loop. */
    bool executionSuccessful = true;
    constexpr bool bUseMenu = NUMBER_OF_FILES >= 1 ? true : false;

    /* Loop. */
    do {
        int menuOption = common::MENU_OPT_RUN_INF_NEXT;
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
            default:
                printf("Incorrect choice, try again.");
                break;
        }
        fflush(stdout);
    } while (executionSuccessful && bUseMenu);
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

    main_loop_dev(caseContext);

    printf_err("Main loop terminated.\r\n");
}
