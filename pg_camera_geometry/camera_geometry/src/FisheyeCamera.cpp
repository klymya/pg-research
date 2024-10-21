#include <math.h>
#include <iostream>
#include <sstream>
#include "FisheyeCamera.hpp"

using namespace std;

static const int MAX_UNDISTORT_ITRS = 10;

string CameraCalibrationParametersToString(const CameraCalibrationParameters& params)
{
    stringstream ss;
    ss << "\n\tCX: " << params.CX
       << "\n\tCY: " << params.CY
       << "\n\tFX: " << params.FX
       << "\n\tFY: " << params.FY
       << "\n\tK0: " << params.K0
       << "\n\tK1: " << params.K1
       << "\n\tK2: " << params.K2
       << "\n\tK3: " << params.K3;

    return ss.str();
}

// convert to normalized units
void FisheyeInternal::Normalize(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params)
{
    pointOut.x = (pointIn.x - params.CX) * params.FX_INV;
    pointOut.y = (pointIn.y - params.CY) * params.FY_INV;
}

// back to un-noramlized units
void FisheyeInternal::Denormalize(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params)
{
    pointOut.x = pointIn.x * params.FX + params.CX;
    pointOut.y = pointIn.y * params.FY + params.CY;
}

// Apply distortion to normalized pixel:
void FisheyeInternal::DistortNormalized(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params)
{
    float r = pointIn.x*pointIn.x + pointIn.y*pointIn.y;

    if (r < params.EPS2)
    {
        // center pixel - nothing to do here

        pointOut = pointIn;
        return;
    }

    // calc radial scaling factor by distortion parameters:
    r = sqrtf(r);
    float t = atanf(r);
    float t2 = t*t;
    float t3 = t2*t;
    float t5 = t3*t2;
    float t7 = t5*t2;

    float a = (t + params.K0*t3 + params.K1*t5 + params.K2*t7 + params.K3*t7*t2) / r;

    // apply factor:
    pointOut.x = pointIn.x * a;
    pointOut.y = pointIn.y * a;
}

// Distort pixel (=From "real world" (upto pixel-to-cm factor) to image)
void Distort(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params)
{
    FisheyeInternal::Normalize(pointIn, pointOut, params);
    FisheyeInternal::DistortNormalized(pointOut, pointOut, params);
    FisheyeInternal::Denormalize(pointOut, pointOut, params);
}

// Undistort pixel (=From image to "real world" (upto pixel-to-cm factor))
void Undistort(const Point& pointIn, Point& pointOut, const CameraCalibrationParameters& params)
{
    Point p;
    FisheyeInternal::Normalize(pointIn, p, params);

    // undistort by iteratively guessing and correcting:
    Point guess(p);
    Point dstrt;
    for (int itr=0; itr<MAX_UNDISTORT_ITRS; ++itr)
    {
        FisheyeInternal::DistortNormalized(guess, dstrt, params);
        float dx = p.x - dstrt.x;
        float dy = p.y - dstrt.y;
        guess.x += dx;
        guess.y += dy;
        if (dx*dx + dy*dy < params.EPS2)
        {
            // we've reached sub-pixel accuracy
            break;
        }
    }

    FisheyeInternal::Denormalize(guess, pointOut, params);
}

void Project(const Point& pointIn, Point& pointOut, float camHeightCm, float heightCm, const CameraCalibrationParameters& params)
{
    float factor = heightCm > 0 ?
        camHeightCm / (float)(camHeightCm - heightCm) :
        (camHeightCm + heightCm) / (float) camHeightCm ;

    Undistort(pointIn, pointOut, params);
    pointOut.x = factor * (pointOut.x - params.CX) + params.CX;
    pointOut.y = factor * (pointOut.y - params.CY) + params.CY;
    Distort(pointOut, pointOut, params);
}

// Pixel -> Cm
void Pixel2Cm(const Point& pointIn, Point& pointOut, float camHeightCm, const CameraCalibrationParameters& params)
{
    Undistort(pointIn, pointOut, params);
    pointOut.x = (pointOut.x - params.CX) * params.FX_INV * camHeightCm;
    pointOut.y = (params.CY - pointOut.y) * params.FY_INV * camHeightCm;
}

void Cm2Pixel(const Point& pointIn, Point& pointOut, float camHeightCm, float heightCm, const CameraCalibrationParameters& params)
{
    pointOut = pointIn;

    if (heightCm != 0)
    {
        // non-zero height -- project cm "on the floor":
        float factor = camHeightCm / (camHeightCm - heightCm) ;
        pointOut.x *= factor;
        pointOut.y *= factor;
    }
    float camHeightInv = 1.f/camHeightCm;
    pointOut.x = pointOut.x * params.FX * camHeightInv + params.CX;
    pointOut.y = -pointOut.y * params.FY * camHeightInv + params.CY;
    Distort(pointOut, pointOut, params);
}
