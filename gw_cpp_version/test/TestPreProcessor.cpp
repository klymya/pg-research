#include <gtest/gtest.h>
#include <fstream>
#include <json.hpp>
#include "PreProcessor.hpp"
#include "PreProcessingUtils.hpp"
#include "Geometry.hpp"
#include "Common.hpp"

using json = nlohmann::json;

class PreProcessorTest : public ::testing::Test {
protected:
    XYStorage* storage;

    std::string tmpJsonFilePath;

    void SetUp() override {
        storage = new XYStorage("gw.db");

        json data = {
            {"sensor_1", {
                {"LENS_PARAMS", {
                    {"CX", DEFAULT_CALIBRATION_PARAMS.CX},
                    {"CY", DEFAULT_CALIBRATION_PARAMS.CY},
                    {"FX", DEFAULT_CALIBRATION_PARAMS.FX},
                    {"FY", DEFAULT_CALIBRATION_PARAMS.FY},
                    {"K0", DEFAULT_CALIBRATION_PARAMS.K0},
                    {"K1", DEFAULT_CALIBRATION_PARAMS.K1},
                    {"K2", DEFAULT_CALIBRATION_PARAMS.K2},
                    {"K3", DEFAULT_CALIBRATION_PARAMS.K3},
                    {"FOV_X", DEFAULT_CAMERA_CONFIG.fovAngleX},
                    {"FOV_Y", DEFAULT_CAMERA_CONFIG.fovAngleY}
                }},
                {"HEIGHT", DEFAULT_CAMERA_CONFIG.heightCm},
                {"IMG_W", DEFAULT_CAMERA_CONFIG.imageW},
                {"IMG_H", DEFAULT_CAMERA_CONFIG.imageH}
            }}
        };

        tmpJsonFilePath = "sensors.json";
        std::ofstream file(tmpJsonFilePath);
        file << data.dump();
        file.close();
    }

    void TearDown() override {
        storage->close();
        delete storage;
        std::remove("gw.db");

        std::remove(tmpJsonFilePath.c_str());
    }
};

TEST_F(PreProcessorTest, PreprocessDetection) {
    PreProcessor preprocessor(tmpJsonFilePath);

    std::vector<Detection> detections = {
        Detection(Rectangle::fromXYWH(320, 240, 50, 50), Point(320, 240), 0.9f),
        Detection(Rectangle::fromXYWH(10, 0, 50, 50), Point(10, 0), 0.8f)
    };

    preprocessor.preProcessDetection(detections, "sensor_1");

    auto data_points = storage->queryDataPoints("sensor_1", 0, preprocessor.getTimestampSec() + 1);
    ASSERT_EQ(data_points.size(), 1);
    ASSERT_EQ(std::get<1>(data_points[0]), "sensor_1");
    ASSERT_EQ(std::get<2>(data_points[0]), -17);
    ASSERT_EQ(std::get<3>(data_points[0]), 10);
}
