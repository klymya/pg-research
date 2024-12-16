
#pragma once

#include <sqlite3.h>
#include <string>
#include <vector>
#include <tuple>

typedef std::tuple<int, std::string, int, int> DataPoint;

class XYStorage {
public:
    XYStorage(const std::string& db_name = "gw.db");
    ~XYStorage();

    void appendDataPoint(const std::string& sensorSerialNumber, int x, int y, int timestamp = -1);
    void queryDataPoints(std::vector<DataPoint>& dataPoints, const std::string& sensorSerialNumber, int startTimestamp, int endTimestamp = -1);
    void removeOldDataPoints(int thresholdMin = 30);
    void close();

private:
    void createTable();

    sqlite3* m_conn;
};
