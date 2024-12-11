#include <gtest/gtest.h>

#include "PreProcessingUtils.hpp"
#include "DetectionArea.hpp"
#include "Geometry.hpp"
#include "FisheyeCamera.hpp"
#include "Common.hpp"

TEST(PreprocessingUtilsTest, FilterDetectionsByDA_AllInside) {
    DetectionArea da(DEFAULT_CAMERA_CONFIG);
    std::vector<Detection> detections = {
        Detection(Rectangle::fromXYWH(320, 240, 50, 50), Point(320, 240), 0.9f),
        Detection(Rectangle::fromXYWH(330, 250, 50, 50), Point(330, 250), 0.8f)
    };
    auto filtered = filterDetectionsByDa(detections, da);
    ASSERT_EQ(filtered.size(), 2);
}

TEST(PreprocessingUtilsTest, FilterDetectionsByDA_AllOutside) {
    DetectionArea da(DEFAULT_CAMERA_CONFIG);
    std::vector<Detection> detections = {
        Detection(Rectangle::fromXYWH(10, 0, 50, 50), Point(10, 0), 0.9f),
        Detection(Rectangle::fromXYWH(20, 10, 50, 50), Point(20, 10), 0.8f)
    };
    auto filtered = filterDetectionsByDa(detections, da);
    ASSERT_EQ(filtered.size(), 0);
}

TEST(PreprocessingUtilsTest, FilterDetectionsByDA_SomeInside) {
    DetectionArea da(DEFAULT_CAMERA_CONFIG);
    std::vector<Detection> detections = {
        Detection(Rectangle::fromXYWH(320, 240, 50, 50), Point(320, 240), 0.9f),
        Detection(Rectangle::fromXYWH(10, 0, 50, 50), Point(10, 0), 0.8f)
    };
    auto filtered = filterDetectionsByDa(detections, da);
    ASSERT_EQ(filtered.size(), 1);
    ASSERT_EQ(filtered[0].position.x, 320);
    ASSERT_EQ(filtered[0].position.y, 240);
}

TEST(PreprocessingUtilsTest, Detection2XY) {
    std::vector<Detection> detections = {
        Detection(Rectangle::fromXYWH(320, 240, 50, 50), Point(320, 240), 0.9f),
    };
    auto point = detection2xy(detections, DEFAULT_CALIBRATION_PARAMS, CAMERA_H)[0];
    ASSERT_EQ(point.x, -17);
    ASSERT_EQ(point.y, 10);
}

TEST(PreprocessingUtilsTest, Preprocessing) {
    DetectionArea da(DEFAULT_CAMERA_CONFIG);
    std::vector<Detection> detections = {
        Detection(Rectangle::fromXYWH(320, 240, 50, 50), Point(320, 240), 0.9f),
        Detection(Rectangle::fromXYWH(10, 0, 50, 50), Point(10, 0), 0.8f)
    };
    auto points = preProcessing(detections, da, DEFAULT_CAMERA_CONFIG);
    ASSERT_EQ(points.size(), 1);
    ASSERT_EQ(points[0].x, -17);
    ASSERT_EQ(points[0].y, 10);
}
