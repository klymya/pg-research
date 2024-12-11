#pragma once

#include "FisheyeCamera.hpp"
#include "Geometry.hpp"

struct CameraConfiguration
{
    CameraConfiguration(const CameraCalibrationParameters& calibrationParams, float fovAngleX,  float fovAngleY,
                        float heightCm, int imageW, int imageH): calibrationParams(calibrationParams),
                        fovAngleX(fovAngleX), fovAngleY(fovAngleY), heightCm(heightCm) {};
    CameraConfiguration(const CameraConfiguration& other) = default;

    CameraCalibrationParameters calibrationParams;
    float fovAngleX;
    float fovAngleY;
    float heightCm;
    int imageW;
    int imageH;
    
};

class DetectionArea
{
public:
    DetectionArea(const CameraConfiguration& cam);

    Rectangle GetDARectangleCm();
    Rectangle GetDARectanglePx();

    bool IsPointInsidePx(const Point& p) const;
    bool IsRectIntersectPx(const Rectangle& r) const;

private:
    CameraConfiguration m_cam;

    Rectangle m_rectangleCm;
    Rectangle m_rectanglePx;
};

namespace DetectionAreaInternal
{
    float Deg2Rad(float x);
}
