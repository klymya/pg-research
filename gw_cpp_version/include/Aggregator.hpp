#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include "XYStorage.hpp"
#include "AreaOfInterest.hpp"
#include "Geometry.hpp"

typedef std::tuple<std::string, std::string, int> AreaCount;
typedef std::tuple<std::string, Point> XY;
typedef std::tuple<int, Point> PointTs;

class Aggregator {
public:
    Aggregator(const std::vector<std::string>& sensorSerials);
    void setAoIs(const std::string& sensorSerialNumber, const std::vector<AreaOfInterest>& aois);
    void getAoIs(const std::string& sensorSerialNumber, std::vector<AreaOfInterest>& aois);
    void checkAoIs(int tsSec, vector<AreaCount>& counts);
    void getXYs(int tsSec, vector<XY>& xys);

private:
    XYStorage m_storage;
    std::vector<std::string> m_sensorSerials;
    std::vector<std::vector<AreaOfInterest>> m_aois;

    void getAoIByReportPeriod(int tsSec, std::vector<std::tuple<std::string, AreaOfInterest>>& aoisByTs);
    void getPoints(const std::string& serialNum, int startTs, int endTs, std::vector<int>& timestamps, std::vector<std::vector<Point>>& points);
    int calculateCount(const AreaOfInterest& aoi, const  std::vector<int>& timestamps, const std::vector<std::vector<Point>>& points, int startTs, int endTs);
    int countNumbersInRange(int anchor, int period, int start, int end);
};
