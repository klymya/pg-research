#include "AreaOfInterest.hpp"
#include <gtest/gtest.h>

TEST(AreaOfInterestTests, Initialization) {
    std::vector<Point> polygon = {Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)};
    AreaOfInterest aoi("123", "Test Area", polygon, 60);
    EXPECT_EQ(aoi.getId(), "123");
    EXPECT_EQ(aoi.getName(), "Test Area");
    EXPECT_EQ(aoi.getPolygon(), polygon);
    EXPECT_EQ(aoi.getReportingPeriod(), 60);
}

TEST(AreaOfInterestTests, ReportingPeriodSetter) {
    std::vector<Point> polygon = {Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)};
    AreaOfInterest aoi("123", "Test Area", polygon, 60);
    aoi.setReportingPeriod(120);
    EXPECT_EQ(aoi.getReportingPeriod(), 120);

    EXPECT_THROW(aoi.setReportingPeriod(0), std::invalid_argument);
}

TEST(AreaOfInterestTests, NameSetter) {
    std::vector<Point> polygon = {Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)};
    AreaOfInterest aoi("123", "Test Area", polygon, 60);
    aoi.setName("New Name");
    EXPECT_EQ(aoi.getName(), "New Name");

    EXPECT_THROW(aoi.setName(""), std::invalid_argument);
}

TEST(AreaOfInterestTests, PolygonSetter) {
    std::vector<Point> polygon1 = {Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)};
    std::vector<Point> polygon2 = {Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)};
    AreaOfInterest aoi("123", "Test Area", polygon1, 60);
    aoi.setPolygon(polygon2);
    EXPECT_EQ(aoi.getPolygon(), polygon2);
}

TEST(AreaOfInterestTests, IsPointInside) {
    std::vector<Point> polygon = {Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)};
    AreaOfInterest aoi("123", "Test Area", polygon, 60);

    Point inside_point(0.5, 0.5);
    Point outside_point(1.5, 1.5);
    Point edge_point(1, 0);

    EXPECT_TRUE(aoi.isPointInside(inside_point));
    EXPECT_FALSE(aoi.isPointInside(outside_point));
    EXPECT_TRUE(aoi.isPointInside(edge_point));
}
