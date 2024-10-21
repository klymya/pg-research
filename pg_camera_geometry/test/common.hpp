#pragma once

#include "FisheyeCamera.hpp"
#include "DetectionArea.hpp"

// cam3
const CameraCalibrationParameters DEFAULT_CALIBRATION_PARAMS(
    332.663306034164,
    247.362736041039,
    201.530310535271,
    201.753479085099,
    0.351145527648731,
    -0.0859250956571053,
    -0.0056549593712587,
    0.00646710057070369
);
const int W = 640;
const int H = 480;
const float CAMERA_H = 275.0;
const float TABLE_H = 70.0;

const CameraConfiguration DEFAULT_CAMERA_CONFIG(
    DEFAULT_CALIBRATION_PARAMS,
    135.0,
    109.0,
    CAMERA_H,
    W,
    H
);
