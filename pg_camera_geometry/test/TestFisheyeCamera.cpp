#include <gtest/gtest.h>
#include <iostream>
#include "FisheyeCamera.hpp"
#include "common.hpp"

using namespace std;

// CameraCalibrationParametersToString
TEST(TestFisheyeCamera, CameraCalibrationParametersToStringDefault)
{
    CameraCalibrationParameters params(0, 1, 2, 3, 4, 5, 6, 7);
    stringstream expected;
    expected << "\n\tCX: " << 0
        << "\n\tCY: " << 1
        << "\n\tFX: " << 2
        << "\n\tFY: " << 3
        << "\n\tK0: " << 4
        << "\n\tK1: " << 5
        << "\n\tK2: " << 6
        << "\n\tK3: " << 7;
    string result = CameraCalibrationParametersToString(params);
    EXPECT_EQ(result, expected.str());
}

// FisheyeInternal::Normalize
TEST(TestFisheyeCamera, NormalizeCenter)
{
    Point pIn = {float(W) / 2, float(H) / 2};
    Point pOut;
    FisheyeInternal::Normalize(pIn, pOut, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, -0.063, 1e-3);
    EXPECT_NEAR(pOut.y, -0.037, 1e-3);
}

// FisheyeInternal::Denormalize
TEST(TestFisheyeCamera, DenormalizeCenter)
{
    Point pIn = {-0.063, -0.037};
    Point pOut;
    FisheyeInternal::Denormalize(pIn, pOut, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, float(W) / 2, 1);
    EXPECT_NEAR(pOut.y, float(H) / 2, 1);
}

// FisheyeInternal::DistortNormalizedCenter
TEST(TestFisheyeCamera, DistortNormalizedCenter)
{
    Point pIn = {0.0f, 0.0f};
    Point pOut;
    FisheyeInternal::DistortNormalized(pIn, pOut, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, pIn.x, 1e-3);
    EXPECT_NEAR(pOut.y, pIn.y, 1e-3);
}

TEST(TestFisheyeCamera, DistortNormalizedBottomRight)
{
    Point pIn = {1, 1};
    Point pOut;
    FisheyeInternal::DistortNormalized(pIn, pOut, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, 0.844, 1e-3);
    EXPECT_NEAR(pOut.y, 0.844, 1e-3);
}

// Distort
TEST(TestFisheyeCamera, DistortCenter)
{
    Point pIn = {float(W) / 2, float(H) / 2};
    Point pOut;
    Distort(pIn, pOut, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, float(W) / 2, 1);
    EXPECT_NEAR(pOut.y, float(H) / 2, 1);
}

TEST(TestFisheyeCamera, DistortTopLeft)
{
    Point pIn = {0.0f, 0.0f};
    Point pOut;
    Distort(pIn, pOut, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, 96, 1);
    EXPECT_NEAR(pOut.y, 71, 1);
}

// Undistort
TEST(TestFisheyeCamera, UndistortCenter)
{
    Point pIn = {float(W) / 2, float(H) / 2};
    Point pOut;
    Undistort(pIn, pOut, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, float(W) / 2, 1);
    EXPECT_NEAR(pOut.y, float(H) / 2, 1);
}

TEST(TestFisheyeCamera, UndistortTopLeft)
{
    Point pIn = {0, 0};
    Point pOut;
    Undistort(pIn, pOut, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, -461, 1);
    EXPECT_NEAR(pOut.y, -343, 1);
}

// Project
TEST(TestFisheyeCamera, ProjectZeroHeight)
{
    Point pIn = {float(W) / 4 , float(H) / 4};
    Point pOut;
    Project(pIn, pOut, CAMERA_H, 0, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, float(W) / 4, 1);
    EXPECT_NEAR(pOut.y, float(H) / 4, 1);
}

TEST(TestFisheyeCamera, ProjectPositiveHeight)
{
    Point pIn = {float(W) / 4 , float(H) / 4};
    Point pOut;
    Project(pIn, pOut, CAMERA_H, TABLE_H, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, 124, 1);
    EXPECT_NEAR(pOut.y, 93, 1);
}

TEST(TestFisheyeCamera, ProjectNegativeHeight)
{
    Point pIn = {float(W) / 4 , float(H) / 4};
    Point pOut;
    Project(pIn, pOut, CAMERA_H, -TABLE_H, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, 196, 1);
    EXPECT_NEAR(pOut.y, 146, 1);
}

// Pixel2Cm
TEST(TestFisheyeCamera, Pixel2CmCenter)
{
    Point pIn = {float(W) / 2 , float(H) / 2};
    Point pOut;
    Pixel2Cm(pIn, pOut, CAMERA_H, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, -17, 1);
    EXPECT_NEAR(pOut.y, 10, 1);
}

// Cm2Pixel
TEST(TestFisheyeCamera, Cm2PixelGroundLevel)
{
    Point pIn = {200.0f, 200.0f};
    Point pOut;
    Cm2Pixel(pIn, pOut, CAMERA_H, 0, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, 468, 1);
    EXPECT_NEAR(pOut.y, 112, 1);
}

TEST(TestFisheyeCamera, Cm2PixelTableLevel)
{
    Point pIn = {200.0f, 200.0f};
    Point pOut;
    Cm2Pixel(pIn, pOut, CAMERA_H, TABLE_H, DEFAULT_CALIBRATION_PARAMS);
    EXPECT_NEAR(pOut.x, 500, 1);
    EXPECT_NEAR(pOut.y, 80, 1);
}
