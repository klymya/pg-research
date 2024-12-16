#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "Aggregator.hpp"
#include "XYStorage.hpp"
#include "AreaOfInterest.hpp"
#include "Geometry.hpp"

using ::testing::_;
using ::testing::Return;
using ::testing::MockFunction;

class AggregatorTest : public ::testing::Test {
protected:
    XYStorage* storage;
    Aggregator* aggregator;

    void SetUp() override {
        storage = new XYStorage("gw.db");
        
        aggregator = new Aggregator({"sensor1", "sensor2"});
    }

    void TearDown() override {
        storage->close();
        delete storage;
        std::remove("gw.db");

        delete aggregator;
    }
};

TEST_F(AggregatorTest, SetAndGetAoIs) {
    AreaOfInterest aoi1("1", "tst", {}, 10);
    AreaOfInterest aoi2("2", "tst", {}, 20);
    aggregator->setAoIs("sensor1", {aoi1, aoi2});

    std::vector<AreaOfInterest> aois;
    aggregator->getAoIs("sensor1", aois);
    ASSERT_EQ(aois.size(), 2);
    EXPECT_EQ(aois[0].getId(), "1");
    EXPECT_EQ(aois[1].getId(), "2");

    EXPECT_THROW(aggregator->getAoIs("sensor3", aois), std::invalid_argument);
}

TEST_F(AggregatorTest, CheckAoIs) {
    AreaOfInterest aoi("aoi1", "tst", {Point(0, 0), Point(0, 1.5), Point(1.5, 1.5), Point(1.5, 0)}, 30);
    aggregator->setAoIs("sensor1", {aoi});

    storage->appendDataPoint("sensor1", 1, 1, 10);
    storage->appendDataPoint("sensor1", 1, 1, 15);
    storage->appendDataPoint("sensor1", 2, 2, 20);

    int tsSec = 30;
    std::vector<AreaCount> counts;
    aggregator->checkAoIs(tsSec, counts);

    ASSERT_EQ(counts.size(), 1);
    EXPECT_EQ(std::get<0>(counts[0]), "sensor1");
    EXPECT_EQ(std::get<1>(counts[0]), "aoi1");
    EXPECT_EQ(std::get<2>(counts[0]), 1);
}

TEST_F(AggregatorTest, GetXYs) {
    storage->appendDataPoint("sensor1", 1, 1, 995);
    storage->appendDataPoint("sensor1", 2, 2, 1000);

    storage->appendDataPoint("sensor2", 3, 3, 995);
    storage->appendDataPoint("sensor2", 4, 4, 1000);

    int tsSec = 1001;
    std::vector<XY> xys;
    aggregator->getXYs(tsSec, xys);

    ASSERT_EQ(xys.size(), 2);
    EXPECT_EQ(std::get<0>(xys[0]), "sensor1");
    EXPECT_EQ(std::get<1>(xys[0]).x, 2);
    EXPECT_EQ(std::get<1>(xys[0]).y, 2);
    EXPECT_EQ(std::get<0>(xys[1]), "sensor2");
    EXPECT_EQ(std::get<1>(xys[1]).x, 4);
    EXPECT_EQ(std::get<1>(xys[1]).y, 4);
}
