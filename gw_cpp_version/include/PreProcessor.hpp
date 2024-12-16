#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <ctime>
#include "PreProcessingUtils.hpp"
#include "DetectionArea.hpp"
#include "FisheyeCamera.hpp"
#include "XYStorage.hpp"
#include "json.hpp"

class PreProcessor {
public:
    PreProcessor(const std::string& dataPath);
    void preProcessDetection(const std::vector<Detection>& detections, const std::string& sensorSerialNumber, int ts = -1);

private:
    void loadSensorsParams(const std::string& dataPath);

    XYStorage m_xyStorage;
    std::vector<CameraConfiguration> m_cameraConfigs;
    std::vector<DetectionArea> m_detectionAreas;
    std::vector<std::string> m_sensorSerialNumbers;
};
