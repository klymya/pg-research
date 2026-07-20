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
#include "Yolov8Model.hpp"       /* Model class for running inference. */
#include "DetectionHeadModel.hpp"
#include "UseCaseHandler.hpp"         /* Handlers for different user options. */
#include "UseCaseCommonUtils.hpp"     /* Utils functions. */
#include "log_macros.h"             /* Logging functions */
#include "BufAttributes.hpp"        /* Buffer attributes to be applied */


static void DisplayDetectionMenu()
{
    printf("\n\n");
    printf("User input required\n");
    printf("Enter option number from:\n\n");
    printf("  %u. Run detection on next ifm\n", common::MENU_OPT_RUN_INF_NEXT);
    printf("  %u. Run detection ifm at chosen index\n", common::MENU_OPT_RUN_INF_CHOSEN);
    printf("  %u. Run detection on all ifm\n", common::MENU_OPT_RUN_INF_ALL);
    printf("  %u. Show NN model info\n", common::MENU_OPT_SHOW_MODEL_INFO);
    printf("  %u. List ifm\n\n", common::MENU_OPT_LIST_IFM);
    printf("  Choice: ");
    fflush(stdout);
}

void main_loop()
{
    arm::app::Yolov8Model model;  /* Model wrapper object. */
    arm::app::DetectionHeadModel detectionHeadModel;  /* Model wrapper object. */

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{"pg_object_detection"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::Yolov8Model&>("model", model);
    caseContext.Set<arm::app::DetectionHeadModel&>("detectionHeadModel", detectionHeadModel);
    caseContext.Set<uint32_t>("imgIndex", 0);

    /* Loop. */
    bool executionSuccessful = true;
    constexpr bool bUseMenu = NUMBER_OF_FILES >= 1 ? true : false;

    /* Loop. */
    do {
        int menuOption = common::MENU_OPT_RUN_INF_NEXT;
        if (bUseMenu) {
            DisplayDetectionMenu();
            menuOption = arm::app::ReadUserInputAsInt();
            printf("\n");
        }
        switch (menuOption) {
            case common::MENU_OPT_RUN_INF_NEXT:
                executionSuccessful = PGStubHandler(caseContext, caseContext.Get<uint32_t>("imgIndex"), false);
                break;
            case common::MENU_OPT_RUN_INF_CHOSEN: {
                printf("    Enter the image index [0, %d]: ", NUMBER_OF_FILES-1);
                fflush(stdout);
                auto imgIndex = static_cast<uint32_t>(arm::app::ReadUserInputAsInt());
                executionSuccessful = PGStubHandler(caseContext, imgIndex, false);
                break;
            }
            case common::MENU_OPT_RUN_INF_ALL:
                executionSuccessful = PGStubHandler(caseContext, caseContext.Get<uint32_t>("imgIndex"), true);
                break;
            case common::MENU_OPT_SHOW_MODEL_INFO:
                printf("The case not implemented.");
                break;
            case common::MENU_OPT_LIST_IFM:
                printf("The case not implemented.");
                break;
            default:
                printf("Incorrect choice, try again.");
                break;
        }
    } while (executionSuccessful && bUseMenu);
    info("Main loop terminated.\n");
}
