/*
Fisheye camera model: https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
 */
#pragma once

#include <iostream>
#include "Geometry.hpp"

using namespace std;

struct CameraCalibrationParameters
{
    CameraCalibrationParameters(float cx, float cy, float fx, float fy, float k0, float k1, float k2, float k3)
        : CX(cx),
          CY(cy),
          FX(fx),
          FY(fy),
          K0(k0),
          K1(k1),
          K2(k2),
          K3(k3),
          FX_INV(1.f/FX),
          FY_INV(1.f/FY)
    {
        EPS2 = FX_INV<FY_INV ? FX_INV*FX_INV : FY_INV*FY_INV;
    }

    CameraCalibrationParameters(const CameraCalibrationParameters& other) = default;

    float CX;
    float CY;
    float FX;
    float FY;

    float K0;
    float K1;
    float K2;
    float K3;

    float FX_INV;
    float FY_INV;

    // Percision is taken to be the minimal size of 1 pixel in normalized units
    float EPS2;
};

namespace FisheyeInternal
{
    // convert to normalized units
    void Normalize(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params);

    // back to un-noramlized units
    void Denormalize(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params);

    // Apply distortion to normalized pixel:
    void DistortNormalized(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params);
}

string CameraCalibrationParametersToString(const CameraCalibrationParameters& params);

// --- Distort
// "flat" (or "corrected") plane --> image-plane:
void Distort(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params);

// --- Undistort
// image-plane --> "flat" (or "corrected") plane:
void Undistort(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params);

// --- Pixel2Cm
// pixel -> cm:
void Pixel2Cm(const Point& pointIn, Point& pointOut, float camHeightCm, const CameraCalibrationParameters& params);

// --- Cm2Pixel
// cm -> pixel. cm is essentially 3d: (x,y,height):
void Cm2Pixel(const Point& pointIn, Point& pointOut, float camHeightCm, float heightCm, const CameraCalibrationParameters& params);

// --- Project
// project pixel to/from a given height.
//
// examples:
// Say we have a person whos feet meet the floor at pixel = (100, 200).
// Assuming the person's height is 175cm, and that the camera is at height
// 250cm, the pixel of the top of the head of the person is given by:
//    float x(100.f), y(200.f);
//  Project(x, y, 175, 250);
//
// The opposite case is if we have the pixel location of the head, and we
// want the pixel of the feet, i.e. a *negative* height difference:
//    Project(x, y, -175, 250);
void Project(const Point& pointIn, Point& pointOut, float camHeightCm, float heightCm, const CameraCalibrationParameters& params);
