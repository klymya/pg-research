#include <gtest/gtest.h>
#include "Geometry.hpp"

// IsPointInsideRectangle
TEST(TestGeometry, IsPointInsideRectangleTrue)
{
    Rectangle r = Rectangle::fromXYWH(0.0f, 0.0f, 10.0f, 10.0f);
    Point p(1, 1);
    EXPECT_TRUE(IsPointInsideRectangle(p, r));
}

TEST(TestGeometry, IsPointInsideRectangleFalse)
{
    Rectangle r = Rectangle::fromXYWH(0.0f, 0.0f, 10.0f, 10.0f);
    Point p(11, 11);
    EXPECT_FALSE(IsPointInsideRectangle(p, r));
}

// CalculateIoU
TEST(TestGeometry, CalculateIoUDefault)
{
    Rectangle r1 = Rectangle::fromXYWH(0.0f, 0.0f, 10.0f, 10.0f);
    Rectangle r2 = Rectangle::fromXYWH(5.0f, 0.0f, 5.0f, 10.0f);
    EXPECT_EQ(CalculateIoU(r1, r2), 0.5f);
}
