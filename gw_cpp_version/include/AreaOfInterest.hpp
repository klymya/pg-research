#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include "Geometry.hpp"

class AreaOfInterest {
private:
    std::string _id;
    std::string _name;
    std::vector<Point> _polygon;
    int _reporting_period;

    static float cross_product(const Point& o, const Point& a, const Point& b);

public:
    AreaOfInterest(const std::string& area_id, const std::string& name, const std::vector<Point>& polygon, int reporting_period_sec);

    std::string getId() const;
    int getReportingPeriod() const;
    void setReportingPeriod(int period_sec);
    std::vector<Point> getPolygon() const;
    void setPolygon(const std::vector<Point>& polygon);
    std::string getName() const;
    void setName(const std::string& name);
    bool isPointInside(const Point& point) const;
};
