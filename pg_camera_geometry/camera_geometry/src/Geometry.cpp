#include "Geometry.hpp"

bool IsPointInsideRectangle(const Point& p, const Rectangle& r)
{
    if (p.x >= r.x & p.x <= r.x1 & p.y >= r.y & p.y <= r.y1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

float CalculateIoU(const Rectangle& rect1, const Rectangle& rect2)
{
    float xA = max(rect1.x, rect2.x);
    float yA = max(rect1.y, rect2.y);
    float xB = min(rect1.x1, rect2.x1);
    float yB = min(rect1.y1, rect2.y1);

    float intersectionArea = max(0.0f, xB - xA) * max(0.0f, yB - yA);

    float rect1Area = rect1.w * rect1.h;
    float rect2Area = rect2.w * rect2.h;

    float unionArea = rect1Area + rect2Area - intersectionArea;

    return intersectionArea / unionArea;
}