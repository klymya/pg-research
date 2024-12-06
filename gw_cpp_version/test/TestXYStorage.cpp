#include <gtest/gtest.h>
#include <XYStorage.h>
#include <cstdio>
#include <ctime>
#include <unistd.h>

class XYStorageTest : public ::testing::Test {
protected:
    XYStorage* storage;

    void SetUp() override {
        storage = new XYStorage("test_gw.db");
    }

    void TearDown() override {
        storage->close();
        delete storage;
        std::remove("test_gw.db");
    }
};

TEST_F(XYStorageTest, AppendDataPoint) {
    // Test with default timestamp
    int timestamp = static_cast<int>(std::time(nullptr)) - 1000;
    storage->appendDataPoint("123", 10, 20, timestamp);
    usleep(100000); // Sleep for 0.1 seconds
    auto data_points = storage->queryDataPoints("123", 0);
    ASSERT_EQ(data_points.size(), 1);
    EXPECT_EQ(std::get<1>(data_points[0]), "123");
    EXPECT_EQ(std::get<2>(data_points[0]), 10);
    EXPECT_EQ(std::get<3>(data_points[0]), 20);

    // Test with provided timestamp
    timestamp = static_cast<int>(std::time(nullptr)) - 100;
    storage->appendDataPoint("123", 30, 40, timestamp);
    usleep(100000); // Sleep for 0.1 seconds
    data_points = storage->queryDataPoints("123", 0);
    ASSERT_EQ(data_points.size(), 2);
    EXPECT_EQ(data_points[1], std::make_tuple(timestamp, "123", 30, 40));
}

TEST_F(XYStorageTest, QueryDataPoints) {
    int timestamp1 = static_cast<int>(std::time(nullptr)) - 2000;
    int timestamp2 = static_cast<int>(std::time(nullptr)) - 1000;
    storage->appendDataPoint("123", 10, 20, timestamp1);
    storage->appendDataPoint("123", 30, 40, timestamp2);
    usleep(100000); // Sleep for 0.1 seconds

    // Query with a range that includes both data points
    auto data_points = storage->queryDataPoints("123", timestamp1 - 1, timestamp2 + 1);
    ASSERT_EQ(data_points.size(), 2);
    EXPECT_LT(std::get<0>(data_points[0]), std::get<0>(data_points[1]));

    // Query with a range that includes only the second data point
    data_points = storage->queryDataPoints("123", timestamp2 - 1, timestamp2 + 1);
    ASSERT_EQ(data_points.size(), 1);
    EXPECT_EQ(data_points[0], std::make_tuple(timestamp2, "123", 30, 40));
}

TEST_F(XYStorageTest, RemoveOldDataPoints) {
    int timestamp1 = static_cast<int>(std::time(nullptr)) - 3600; // 1 hour ago
    int timestamp2 = static_cast<int>(std::time(nullptr)) - 1800; // 30 minutes ago
    storage->appendDataPoint("123", 10, 20, timestamp1);
    storage->appendDataPoint("123", 30, 40, timestamp2);
    usleep(100000); // Sleep for 0.1 seconds

    // Remove data points older than 45 minutes
    storage->removeOldDataPoints(45);
    auto data_points = storage->queryDataPoints("123", 0);
    ASSERT_EQ(data_points.size(), 1);
    EXPECT_EQ(data_points[0], std::make_tuple(timestamp2, "123", 30, 40));
}

TEST_F(XYStorageTest, MultipleCases) {
    int timestamp1 = static_cast<int>(std::time(nullptr)) - 3600; // 1 hour ago
    int timestamp2 = static_cast<int>(std::time(nullptr)) - 1800; // 30 minutes ago
    int timestamp3 = static_cast<int>(std::time(nullptr)) - 900;  // 15 minutes ago
    storage->appendDataPoint("123", 10, 20, timestamp1);
    storage->appendDataPoint("123", 30, 40, timestamp2);
    storage->appendDataPoint("456", 50, 60, timestamp3);
    usleep(100000); // Sleep for 0.1 seconds

    // Query for sensor "123"
    auto data_points = storage->queryDataPoints("123", 0);
    ASSERT_EQ(data_points.size(), 2);

    // Query for sensor "456"
    data_points = storage->queryDataPoints("456", 0);
    ASSERT_EQ(data_points.size(), 1);
    EXPECT_EQ(data_points[0], std::make_tuple(timestamp3, "456", 50, 60));

    // Remove old data points
    storage->removeOldDataPoints(45);
    data_points = storage->queryDataPoints("123", 0);
    ASSERT_EQ(data_points.size(), 1);
    EXPECT_EQ(data_points[0], std::make_tuple(timestamp2, "123", 30, 40));

    data_points = storage->queryDataPoints("456", 0);
    ASSERT_EQ(data_points.size(), 1);
    EXPECT_EQ(data_points[0], std::make_tuple(timestamp3, "456", 50, 60));
}
