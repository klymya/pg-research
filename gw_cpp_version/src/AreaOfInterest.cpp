#include "AreaOfInterest.hpp"

AreaOfInterest::AreaOfInterest(const std::string& area_id, const std::string& name, const std::vector<Point>& polygon, int reporting_period_sec)
    : _id(area_id), _name(name), _polygon(polygon), _reporting_period(reporting_period_sec) {
    if (reporting_period_sec <= 0) {
        throw std::invalid_argument("Reporting period must be positive");
    }
    if (name.empty()) {
        throw std::invalid_argument("Name cannot be empty");
    }
}

std::string AreaOfInterest::getId() const {
    return _id;
}

int AreaOfInterest::getReportingPeriod() const {
    return _reporting_period;
}

void AreaOfInterest::setReportingPeriod(int period_sec) {
    if (period_sec <= 0) {
        throw std::invalid_argument("Reporting period must be positive");
    }
    _reporting_period = period_sec;
}

std::vector<Point> AreaOfInterest::getPolygon() const {
    return _polygon;
}

void AreaOfInterest::setPolygon(const std::vector<Point>& polygon) {
    _polygon = polygon;
}

std::string AreaOfInterest::getName() const {
    return _name;
}

void AreaOfInterest::setName(const std::string& name) {
    if (name.empty()) {
        throw std::invalid_argument("Name cannot be empty");
    }
    _name = name;
}

float AreaOfInterest::cross_product(const Point& o, const Point& a, const Point& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

bool AreaOfInterest::isPointInside(const Point& point) const {
    int n_pos = 0;
    int n_neg = 0;
    for (size_t i = 0; i < _polygon.size(); ++i) {
        const Point& a = _polygon[i];
        const Point& b = _polygon[(i + 1) % _polygon.size()];
        if (cross_product(a, b, point) < 0) {
            n_neg++;
        } else {
            n_pos++;
        }
    }

    return !(n_neg > 0 && n_pos > 0);
}
