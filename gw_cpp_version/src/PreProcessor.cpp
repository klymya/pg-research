#include <fstream>

#include "PreProcessor.hpp"
#include "Utils.hpp"

using json = nlohmann::json;

PreProcessor::PreProcessor(const std::string& dataPath) {
    loadSensorsParams(dataPath);
}

void PreProcessor::loadSensorsParams(const std::string& dataPath) {
    std::ifstream file(dataPath, ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + dataPath);
    }

    json data;
    file >> data;

    for (auto& [serialNum, params] : data.items()) {
        auto lensParams = params["LENS_PARAMS"];
        CameraCalibrationParameters calibParams(
            lensParams["CX"].get<float>(),
            lensParams["CY"].get<float>(),
            lensParams["FX"].get<float>(),
            lensParams["FY"].get<float>(),
            lensParams["K0"].get<float>(),
            lensParams["K1"].get<float>(),
            lensParams["K2"].get<float>(),
            lensParams["K3"].get<float>()
        );

        CameraConfiguration camConfig(
            calibParams,
            lensParams["FOV_X"].get<float>(),
            lensParams["FOV_Y"].get<float>(),
            params["HEIGHT"].get<float>(),
            params["IMG_W"].get<int>(),
            params["IMG_H"].get<int>()
        );

        DetectionArea da(camConfig);

        m_sensorSerialNumbers.push_back(serialNum);
        m_cameraConfigs.push_back(camConfig);
        m_detectionAreas.push_back(da);
    }
}

void PreProcessor::preProcessDetection(const std::vector<Detection>& detections, const std::string& sensorSerialNumber, int ts) {
    int sensorIndex = findIndex(m_sensorSerialNumbers, sensorSerialNumber);
    if (sensorIndex == -1) {
        throw std::invalid_argument("Unknown sensor serial number: " + sensorSerialNumber);
    }

    if (ts == -1) {
        ts = getTimestampSec();
    }

    auto xyPoints = preProcessing(detections, m_detectionAreas[sensorIndex], m_cameraConfigs[sensorIndex]);

    for (const auto& point : xyPoints) {
        m_xyStorage.appendDataPoint(sensorSerialNumber, point.x, point.y, ts);
    }
}
