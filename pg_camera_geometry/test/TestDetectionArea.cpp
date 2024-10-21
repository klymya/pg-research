#include <gtest/gtest.h>
#include <iostream>
#include "DetectionArea.hpp"
#include "common.hpp"

using namespace std;

// CameraCalibrationParametersToString
TEST(TestDetectionArea, Deg2RadDefault)
{
    EXPECT_NEAR(DetectionAreaInternal::Deg2Rad(1), 0.018, 1e-3);
    EXPECT_NEAR(DetectionAreaInternal::Deg2Rad(180), 3.142, 1e-3);
}

// DetectionArea::DetectionArea
TEST(TestDetectionArea, DetectionAreaConstructor)
{
    DetectionArea da(DEFAULT_CAMERA_CONFIG);
    Rectangle rectCm = da.GetDARectangleCm();
    EXPECT_EQ(rectCm.w, 845);
    EXPECT_EQ(rectCm.h, 491);
}

// DetectionArea::IsPointInsidePx
TEST(TestDetectionArea, IsPointInsidePxTrue)
{
    // (95, 109) -> (570, 386)
    DetectionArea da(DEFAULT_CAMERA_CONFIG);
    Rectangle rect = da.GetDARectanglePx();
    Point p(320, 240);
    EXPECT_TRUE(da.IsPointInsidePx(p));
}

TEST(TestDetectionArea, IsPointInsidePxFalse)
{
    DetectionArea da(DEFAULT_CAMERA_CONFIG);
    Point p(90, 100);
    EXPECT_FALSE(da.IsPointInsidePx(p));
}

// DetectionArea::IsRectIntersectPx
TEST(TestDetectionArea, IsRectIntersectPxTrue)
{
    DetectionArea da(DEFAULT_CAMERA_CONFIG);
    Rectangle r = Rectangle::fromXYWH(320, 240, 50, 50);
    EXPECT_TRUE(da.IsRectIntersectPx(r));
}

TEST(TestDetectionArea, IsRectIntersectPxFalse)
{
    DetectionArea da(DEFAULT_CAMERA_CONFIG);
    Rectangle r = Rectangle::fromXYWH(10, 0, 50, 50);
    EXPECT_FALSE(da.IsRectIntersectPx(r));
}