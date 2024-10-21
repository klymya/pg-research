#pragma once

#include <cmath>

using namespace std;

struct Point
{
    float x;
    float y;

    Point(float x, float y): x(x), y(y) {};
    Point() = default;
    Point(const Point& other) = default;
};

struct Rectangle
{
    float x;
    float y;
    float w;
    float h;
    float x1;
    float y1;

    Rectangle(float x, float y, float w, float h, float x1, float y1): x(x), y(y), w(w), h(h), x1(x1), y1(y1) {};
    Rectangle(const Rectangle& other) = default;
    Rectangle() = default;

public:
    static Rectangle fromXYWH(float x, float y, float w, float h)
    {
        return Rectangle(x, y, w, h, x + w, y + h);
    };

    static Rectangle fromXYXY(float x, float y, float x1, float y1)
    {
        return Rectangle(x, y, abs(x1 - x), abs(y1 - y), x1, y1);
    };
};

bool IsPointInsideRectangle(const Point& p, const Rectangle& r);
float CalculateIoU(const Rectangle& a, const Rectangle& b);
