#include <math.h>
#include "DetectionArea.hpp"
#include "Geometry.hpp"

// ATTENTION! The parameters move from previouse version of the product for backward compability.
// TODO: reseach the papameters influence and update/remove them.

// Parameters for determining the margin we take from the FOV boundary
// for computing the detection area. For a narrow FOV we take a smaller
// margin, to allow a reasonable detection area size.
const float MIN_VISIBLE_HEIGHT_CM = 100;

// These values are based on the previous max DA limit (800x600cm) and SPCA2093 camera's
// actual DA's aspect ratio.
const int MAX_DETECTION_AREA_CM2 = 500000;
// END

const int DA_INTERSECTION_TH = 0;

float DetectionAreaInternal::Deg2Rad(float x)
{
    return x * M_PI / 180.0;
}

DetectionArea::DetectionArea(const CameraConfiguration& cam): m_cam(cam)
{
    float fovXAtFocalLen = 2.f * tanf(DetectionAreaInternal::Deg2Rad(m_cam.fovAngleX / 2.f));
    float fovYAtFocalLen = 2.f * tanf(DetectionAreaInternal::Deg2Rad(m_cam.fovAngleY / 2.f));

    float aspectRatio = fovXAtFocalLen / fovYAtFocalLen;

    float width = round((m_cam.heightCm - MIN_VISIBLE_HEIGHT_CM) * fovXAtFocalLen);
    float height = round((m_cam.heightCm - MIN_VISIBLE_HEIGHT_CM) * fovYAtFocalLen);

    if (width * height > MAX_DETECTION_AREA_CM2)
    {
        width = round(sqrt(MAX_DETECTION_AREA_CM2 * aspectRatio));
        height = round(1 / aspectRatio * width);
    }

    m_rectangleCm = Rectangle::fromXYXY(-width / 2, height / 2, width / 2, -height / 2);

    Point tlCm(m_rectangleCm.x, m_rectangleCm.y);
    Point brCm(m_rectangleCm.x1, m_rectangleCm.y1);
    Point tlPx;
    Point brPx;
    Cm2Pixel(tlCm, tlPx, m_cam.heightCm, 0, m_cam.calibrationParams);
    Cm2Pixel(brCm, brPx, m_cam.heightCm, 0, m_cam.calibrationParams);

    m_rectanglePx = Rectangle::fromXYXY(tlPx.x, tlPx.y, brPx.x, brPx.y);
}

Rectangle DetectionArea::GetDARectangleCm()
{
    return m_rectangleCm;
}

Rectangle DetectionArea::GetDARectanglePx()
{
    return m_rectanglePx;
}

bool DetectionArea::IsPointInsidePx(const Point& p) const
{
    return IsPointInsideRectangle(p, m_rectanglePx);
}

bool DetectionArea::IsRectIntersectPx(const Rectangle& r) const
{
    return CalculateIoU(r, m_rectanglePx) > DA_INTERSECTION_TH;
}