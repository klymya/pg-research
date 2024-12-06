#include "XYStorage.h"
#include <ctime>
#include <stdexcept>

XYStorage::XYStorage(const std::string& db_name) {
    if (sqlite3_open(db_name.c_str(), &conn) != SQLITE_OK) {
        throw std::runtime_error("Cannot open database");
    }
    createTable();
}

XYStorage::~XYStorage() {
    sqlite3_close(conn);
}

void XYStorage::createTable() {
    const char* createTableSQL = R"(
        CREATE TABLE IF NOT EXISTS xy_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            sensor_serial_number TEXT NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL
        );
    )";

    const char* createIndexSQLTimestamp = R"(
        CREATE INDEX IF NOT EXISTS idx_timestamp ON xy_data (timestamp);
    )";

    const char* createIndexSQLSensorSerialNumber = R"(
        CREATE INDEX IF NOT EXISTS idx_sensor_serial_number ON xy_data (sensor_serial_number);
    )";

    char* errMsg = nullptr;
    if (sqlite3_exec(conn, createTableSQL, nullptr, nullptr, &errMsg) != SQLITE_OK ||
        sqlite3_exec(conn, createIndexSQLTimestamp, nullptr, nullptr, &errMsg) != SQLITE_OK ||
        sqlite3_exec(conn, createIndexSQLSensorSerialNumber, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::string error = "SQL error: ";
        error += errMsg;
        sqlite3_free(errMsg);
        throw std::runtime_error(error);
    }
}

void XYStorage::appendDataPoint(const std::string& sensorSerialNumber, int x, int y, int timestamp) {
    if (timestamp == -1) {
        timestamp = static_cast<int>(std::time(nullptr));
    }

    const char* insertSQL = R"(
        INSERT INTO xy_data (timestamp, sensor_serial_number, x, y)
        VALUES (?, ?, ?, ?);
    )";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(conn, insertSQL, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
    }

    sqlite3_bind_int(stmt, 1, timestamp);
    sqlite3_bind_text(stmt, 2, sensorSerialNumber.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 3, x);
    sqlite3_bind_int(stmt, 4, y);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        throw std::runtime_error("Failed to execute statement");
    }

    sqlite3_finalize(stmt);
}

std::vector<std::tuple<int, std::string, int, int>> XYStorage::queryDataPoints(const std::string& sensorSerialNumber, int startTimestamp, int endTimestamp) {
    if (endTimestamp == -1) {
        endTimestamp = static_cast<int>(std::time(nullptr));
    }

    const char* querySQL = R"(
        SELECT timestamp, sensor_serial_number, x, y
        FROM xy_data
        WHERE timestamp >= ? AND timestamp < ? AND sensor_serial_number = ?
        ORDER BY timestamp;
    )";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(conn, querySQL, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
    }

    sqlite3_bind_int(stmt, 1, startTimestamp);
    sqlite3_bind_int(stmt, 2, endTimestamp);
    sqlite3_bind_text(stmt, 3, sensorSerialNumber.c_str(), -1, SQLITE_STATIC);

    std::vector<std::tuple<int, std::string, int, int>> results;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int timestamp = sqlite3_column_int(stmt, 0);
        std::string serialNumber = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        int x = sqlite3_column_int(stmt, 2);
        int y = sqlite3_column_int(stmt, 3);
        results.emplace_back(timestamp, serialNumber, x, y);
    }

    sqlite3_finalize(stmt);
    return results;
}

void XYStorage::removeOldDataPoints(int thresholdMin) {
    int th = static_cast<int>(std::time(nullptr)) - thresholdMin * 60;

    const char* deleteSQL = R"(
        DELETE FROM xy_data
        WHERE timestamp <= ?;
    )";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(conn, deleteSQL, -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
    }

    sqlite3_bind_int(stmt, 1, th);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        throw std::runtime_error("Failed to execute statement");
    }

    sqlite3_finalize(stmt);
}

void XYStorage::close() {
    sqlite3_close(conn);
    conn = nullptr;
}
