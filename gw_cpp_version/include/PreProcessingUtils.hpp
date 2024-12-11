#pragma once

#include <vector>
#include "Geometry.hpp"
#include "FisheyeCamera.hpp"
#include "DetectionArea.hpp"

struct Detection
{
    Rectangle bbox;
    Point position;
    float confidence;

    Detection(const Rectangle& bbox, const Point& position, float confidence)
        : bbox(bbox), position(position), confidence(confidence) {}
};

std::vector<Detection> filterDetectionsByDa(const std::vector<Detection>& detections, const DetectionArea& da);
std::vector<Point> detection2xy(const std::vector<Detection>& detections, const CameraCalibrationParameters& calibration_params, int cam_h_cm);
std::vector<Point> preProcessing(const std::vector<Detection>& detections, const DetectionArea& da, const CameraConfiguration& cam_config);
