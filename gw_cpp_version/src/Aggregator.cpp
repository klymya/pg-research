#include <algorithm>

#include "Aggregator.hpp"
#include "Utils.hpp"

#define DEFAULT_SENSOR_REPORTING_PERIOD_SEC 5

Aggregator::Aggregator(const std::vector<std::string>& sensorSerials) : m_sensorSerials(sensorSerials) {
    m_aois.resize(sensorSerials.size());
}

void Aggregator::setAoIs(const std::string& sensorSerialNumber, const std::vector<AreaOfInterest>& aois) {
    int sensorIndex = findIndex(m_sensorSerials, sensorSerialNumber);
    if (sensorIndex == -1) {
        throw std::invalid_argument("Sensor with serial number " + sensorSerialNumber + " does not exist");
    }
    m_aois[sensorIndex] = aois;
}

void Aggregator::getAoIs(const std::string& sensorSerialNumber, std::vector<AreaOfInterest>& aois) {
    int sensorIndex = findIndex(m_sensorSerials, sensorSerialNumber);
    if (sensorIndex == -1) {
        throw std::invalid_argument("Sensor with serial number " + sensorSerialNumber + " does not exist");
    }
    aois = m_aois[sensorIndex];
}

void Aggregator::getAoIByReportPeriod(int tsSec, std::vector<std::tuple<std::string, AreaOfInterest>>& aoisByTs) {
    aoisByTs.clear();
    for (auto i = 0; i < m_sensorSerials.size(); i++) {
        auto sensorSerialNumber = m_sensorSerials[i];
        for (const auto& aoi : m_aois[i]) {
            if (tsSec % aoi.getReportingPeriod() == 0) {
                aoisByTs.push_back({sensorSerialNumber, aoi});
            }
        }
    }
}

int Aggregator::countNumbersInRange(int anchor, int period, int start, int end) {
    if (!(start <= anchor && anchor < end)) {
        return 0;
    }

    int count = 0;
    int current = anchor;

    // Check numbers in the positive direction
    while (current < end) {
        count++;
        current += period;
    }

    // Check numbers in the negative direction
    current = anchor - period;
    while (current >= start) {
        count++;
        current -= period;
    }

    return count;
}

int Aggregator::calculateCount(const AreaOfInterest& aoi, const  std::vector<int>& timestamps,
                               const std::vector<std::vector<Point>>& points, int startTs, int endTs) {
    int anchorTs = timestamps.empty() ? 0 : *std::min_element(timestamps.begin(), timestamps.end());

    int count = 0;
    for (auto pointByTs : points) {
        for (auto point : pointByTs) {
            if (aoi.isPointInside(point)) {
                count++;
            }
        }
    }
    if (count > 0) {
        count = static_cast<int>(std::ceil(count / 
            static_cast<double>(countNumbersInRange(anchorTs, DEFAULT_SENSOR_REPORTING_PERIOD_SEC, startTs, endTs))));
    }
    return count;
}

void Aggregator::getPoints(const std::string& serialNum, int startTs, int endTs, std::vector<int>& timestamps, std::vector<std::vector<Point>>& points) {
    timestamps.clear();
    points.clear();

    std::vector<DataPoint> dataPoints;
    m_storage.queryDataPoints(dataPoints, serialNum, startTs, endTs);
    int i = -1;
    for (const auto& [ts, _, x, y] : dataPoints) {
        i = findIndex(timestamps, ts);
        if (i == -1) {
            timestamps.push_back(ts);
            points.push_back({Point(x, y)});
        } else {
            points[i].push_back(Point(x, y));
        }
    }
}

void Aggregator::checkAoIs(int tsSec, vector<AreaCount>& counts) {
    counts.clear();
    std::vector<std::tuple<std::string, AreaOfInterest>> aois2Check;
    getAoIByReportPeriod(tsSec, aois2Check);

    std::vector<int> timestamps;
    std::vector<std::vector<Point>> points;
    for (const auto& [serialNum, aoi] : aois2Check) {
        getPoints(serialNum, tsSec - aoi.getReportingPeriod(), tsSec, timestamps, points);
        int count = calculateCount(aoi, timestamps, points, tsSec - aoi.getReportingPeriod(), tsSec);
        counts.push_back({serialNum, aoi.getId(), count});
    }

}

void Aggregator::getXYs(int tsSec, vector<XY>& xys) {
    xys.clear();

    std::vector<int> timestamps;
    std::vector<std::vector<Point>> points;
    for (const auto& serialNum : m_sensorSerials) {
        getPoints(serialNum, tsSec - DEFAULT_SENSOR_REPORTING_PERIOD_SEC, tsSec, timestamps, points);
        auto latestPoints = points.empty() ? std::vector<Point>() : points.back();
        for (auto point : latestPoints) {
            xys.emplace_back(serialNum, point);
        }
    }
}
