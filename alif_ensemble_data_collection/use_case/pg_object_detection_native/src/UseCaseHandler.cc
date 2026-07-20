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
#include "DetectorPostProcessing.hpp"
#include "DetectorPreProcessing.hpp"
#include "InputFiles.hpp"
#include "UseCaseCommonUtils.hpp"
#include "Yolov8Model.hpp"
#include "hal.h"
#include "log_macros.h"
#include "ModelAdapter.hpp"
#include "Detection.hpp"
#include "BYTETracker.hpp"
#include <iostream>
#include "jpeg.h"

using namespace std;

namespace arm {
namespace app {

    bool ObjectDetectionInit(void)
    {
        /* Initialise the camera */
        int err = hal_image_init();
        if (0 != err) {
            printf_err("hal_image_init failed with error: %d\n", err);
            return false;
        }

        return true;
    }

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
    bool PGObjectDetectionHandler(ApplicationContext& ctx, uint32_t imgIndex, bool runAll)
    {
        auto& profiler = ctx.Get<Profiler&>("profiler");

        auto& model = ctx.Get<Yolov8Model&>("model");
        auto& detectionHeadModel = ctx.Get<DetectionHeadModel&>("detectionHeadModel");

        /* If the request has a valid size, set the image index. */
        if (imgIndex < NUMBER_OF_FILES) {
            if (!SetAppCtxIfmIdx(ctx, imgIndex, "imgIndex")) {
                return false;
            }
        }
        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }
        if (!detectionHeadModel.IsInited()) {
            printf_err("DetectionHeadModel is not initialised! Terminating processing.\n");
            return false;
        }

        auto initialImgIdx = ctx.Get<uint32_t>("imgIndex");

        TfLiteTensor* inputTensor = model.GetInputTensor(0);

        if (!inputTensor->dims) {
            printf_err("Invalid input tensor dims\n");
            return false;
        } else if (inputTensor->dims->size < 3) {
            printf_err("Input tensor dimension should be >= 3\n");
            return false;
        }

        TfLiteIntArray* inputShape = model.GetInputShape(0);

        const int inputImgCols = inputShape->data[2];
        const int inputImgRows = inputShape->data[1];

        /* Set up pre and post-processing. */
        printf("IsDataSigned %s\n", model.IsDataSigned() ? "true": "false");
        DetectorPreProcess preProcess = DetectorPreProcess(inputTensor, model.IsDataSigned());
        pg_object_detection::ModelAdapter adapter = pg_object_detection::ModelAdapter(&model, &detectionHeadModel);

        vector<pg_object_detection::Detection> detections;
        pg_object_detection::PostProcessParams params;
        pg_object_detection::DetetorPostProcess  postProcrss = 
            pg_object_detection::DetetorPostProcess(detectionHeadModel.GetOutputTensor(0), detections, params);

        do {

            /* Strings for presentation/logging. */
            std::string str_inf{"Running inference... "};

            const uint8_t* currImage = GetImgArray(ctx.Get<uint32_t>("imgIndex"));

            const size_t copySz =
                inputTensor->bytes < IMAGE_DATA_SIZE ? inputTensor->bytes : IMAGE_DATA_SIZE;

            /* Run the pre-processing, inference and post-processing. */
            if (!preProcess.DoPreProcess(currImage, copySz)) {
                printf_err("Pre-processing failed.");
                return false;
            }

            /* Run inference over this image. */
            info("Running inference on image %" PRIu32 " => %s\n",
                ctx.Get<uint32_t>("imgIndex"),
                GetFilename(ctx.Get<uint32_t>("imgIndex")));

            if (!RunInference(model, profiler)) {
                printf_err("Model inference failed.");
                return false;
            }

            if (!adapter.DoAdaptation()) {
                printf_err("Model and DetectionHead data adaptation failed.");
                return false;
            }

            if (!RunInference(detectionHeadModel, profiler)) {
                printf_err("DetectionHeadModel inference failed.");
                return false;
            }

            postProcrss.DoPostProcess();
            PresentInferenceResult(detections);
            profiler.PrintProfilingResult();

            IncrementAppCtxIfmIdx(ctx, "imgIndex");
        } while (runAll && ctx.Get<uint32_t>("imgIndex") != initialImgIdx);

        return true;
    }

    bool PGTrackerHandler() {
        printf("tracker stub\n");

        vector<vector<arm::app::pg_object_detection::Detection>> detections {
            {{{0.0f, 0.0f, 25.0f, 50.0f}, {12.0f, 45.0f}, 0.9f, 0}},
            {{{5.0f, 5.0f, 25.0f, 50.0f}, {17.0f, 50.0f}, 0.9f, 0}},
            {{{11.0f, 9.0f, 25.0f, 50.0f}, {23.0f, 54.0f}, 0.9f, 0}}
        };

        BYTETracker tracker;
        for (auto tmp_detections: detections) {
            vector<STrack> output_stracks = tracker.update(tmp_detections);
            std::cout << "\ntracks:\n";
            for (auto track: output_stracks) {
                auto bbox = track.tlwh;
                auto pos = track.pos;
                printf("TrackID: %d. Box: [%.2f %.2f %.2f %.2f]. Pos: [%.2f %.2f]. State: %d\n",
                    track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], pos[0], pos[1], track.state
                );
            }
        }
        return true;
    }

} /* namespace app */
} /* namespace arm */
