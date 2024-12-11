#include <cmath>

#include "PreProcessingUtils.hpp"

std::vector<Detection> filterDetectionsByDa(const std::vector<Detection>& detections, const DetectionArea& da)
{
    std::vector<Detection> filtered_detections;
    for (const auto& detection : detections)
    {
        if (da.IsRectIntersectPx(detection.bbox))
        {
            filtered_detections.push_back(detection);
        }
    }
    return filtered_detections;
}

std::vector<Point> detection2xy(const std::vector<Detection>& detections, const CameraCalibrationParameters& calibration_params, int cam_h_cm)
{
    std::vector<Point> points;
    for (const auto& detection : detections)
    {
        Point point;
        Pixel2Cm(detection.position, point, cam_h_cm, calibration_params);
        point.x = round(point.x);
        point.y = round(point.y);
        points.push_back(point);
    }
    return points;
}

std::vector<Point> preProcessing(const std::vector<Detection>& detections, const DetectionArea& da, const CameraConfiguration& cam_config)
{
    auto filtered_detections = filterDetectionsByDa(detections, da);
    auto points = detection2xy(filtered_detections, cam_config.calibrationParams, cam_config.heightCm);
    return points;
}
