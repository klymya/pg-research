/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates
 * <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0
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
#include "UseCaseHandler.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "log_macros.h"
#include "Detection.hpp"

using namespace std;

namespace arm {
namespace app {

    bool PresentInferenceResult(const std::vector<pg_object_detection::Detection>& detections)
    {
        printf("Final results:\n");
        printf("Total number of inferences: %d\n", (int)detections.size());

        for (auto detection: detections) {
            printf("Box: [%.2f %.2f %.2f %.2f]. Position: [%.2f %.2f]. Confidence: %.2f\n",
                detection.bbox.x, detection.bbox.y, detection.bbox.w, detection.bbox.h, detection.position.x,
                detection.position.y, detection.confidence
            );
        }

        return true;
    }

    /* Object detection inference handler. */
    bool PGStubHandler(ApplicationContext& ctx, uint32_t imgIndex, bool runAll)
    {
        auto& profiler = ctx.Get<Profiler&>("profiler");
        UNUSED(imgIndex);
        UNUSED(runAll);

        std::vector<pg_object_detection::Detection> detections {
            {
                {0.0f, 0.0f, 50.0f, 100.0f},
                {25.0f, 90.0f},
                0.9,
                0
            }
        };
        PresentInferenceResult(detections);
        profiler.PrintProfilingResult();

        return true;
    }

} /* namespace app */
} /* namespace arm */
