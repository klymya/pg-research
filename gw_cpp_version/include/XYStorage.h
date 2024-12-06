
#pragma once

#include <sqlite3.h>
#include <string>
#include <vector>
#include <tuple>

class XYStorage {
public:
    XYStorage(const std::string& db_name = "gw.db");
    ~XYStorage();

    void appendDataPoint(const std::string& sensorSerialNumber, int x, int y, int timestamp = -1);
    std::vector<std::tuple<int, std::string, int, int>> queryDataPoints(const std::string& sensorSerialNumber, int startTimestamp, int endTimestamp = -1);
    void removeOldDataPoints(int thresholdMin = 30);
    void close();

private:
    void createTable();

    sqlite3* conn;
};
