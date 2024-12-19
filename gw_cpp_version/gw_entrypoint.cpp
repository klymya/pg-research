#include <INIReader.h>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <fstream>
#include "json.hpp"
#include "sensor_payload.pb.h"
#include "MQTTClientHandler.hpp"
#include "PreProcessor.hpp"
#include "PreProcessingUtils.hpp"
#include "XYStorage.hpp"
#include "Aggregator.hpp"
#include "AreaOfInterest.hpp"
#include "Utils.hpp"

using json = nlohmann::json;

const std::string COFIG_FILE = "config.ini";
const std::string SENSOR_SERIAL = "0";
const int THRESHOLD_MIN =1;

PreProcessor* preprocessor;

int onMessage(void* context, char* topicName, int topicLen, MQTTClient_message* message) {
    std::string payload(static_cast<char*>(message->payload), message->payloadlen);

    telemetrypayload::TelemetryNotification msg;
    if (!msg.ParseFromString(payload)) {
        std::cerr << "Failed to parse payload." << std::endl;
        return 1;
    }

    std::vector<Detection> detections;
    for (int i = 0; i < msg.objects_size(); ++i) {
        const telemetrypayload::ReportingObject& object = msg.objects(i);
        detections.push_back(Detection(
            Rectangle::fromXYWH(object.box_x(), object.box_y(), object.box_w(), object.box_h()),
            Point(object.position_x(), object.position_y()),
            object.confidence()
        ));
    }
    std::cout << "Received message from  topic " << topicName << " " << "with ts: " << msg.timestamp() << std::endl;
    preprocessor->preProcessDetection(detections, "0", msg.timestamp());

    return 1;
}

void removeOutdatedData() {
    XYStorage storage = XYStorage();
    while (true) {
        std::cout << "Removing outdated data" << std::endl;
        storage.removeOldDataPoints(THRESHOLD_MIN);
        usleep(THRESHOLD_MIN * 60 * 1e6);
    }
}

void readAoIs(std::string aoiPath, std::vector<AreaOfInterest>& aois) {
    std::ifstream file(aoiPath, ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + aoiPath);
    }

    json data;
    file >> data;
    for (auto& [areaId, params] : data.items()) {
        std::vector<Point> polygon;
        for (const auto& v : params["polygon"]) {
            polygon.push_back(Point(v[0], v[1]));
        }
        aois.push_back(AreaOfInterest(areaId, params["name"], polygon, params["reporting_period"]));
    }
}

void sendCounts(MQTTClientHandler& mqttClient, const std::string& topic, const int tsSec, const std::vector<AreaCount>& counts) {
    telemetrypayload::AoICountNotification msg;
    msg.set_timestamp(tsSec);
    for (const auto& [serialNum, areaId, count] : counts) {
        telemetrypayload::AoICountObject* object = msg.add_objects();
        object->set_aoi_id(areaId);
        object->set_count(count);
    }

    std::string payload;
    if (!msg.SerializeToString(&payload)) {
        std::cerr << "Failed to serialize message." << std::endl;
        return;
    }

    int status = mqttClient.pushMessage(topic, payload);
    if (status == 0) {
        std::cout << "Message is published to topic " << topic << std::endl;
    } else {
        std::cerr << "Failed to publish message to topic " << topic << std::endl;
    }
}

void sendXYs(MQTTClientHandler& mqttClient, const std::string& topic, const int tsSec, const std::vector<XY>& xys){
    telemetrypayload::XYNotification msg;
    msg.set_timestamp(tsSec);
    for (const auto& [serialNum, point] : xys) {
        telemetrypayload::XYObject* object = msg.add_objects();
        object->set_x(point.x);
        object->set_y(point.y);
    }

    std::string payload;
    if (!msg.SerializeToString(&payload)) {
        std::cerr << "Failed to serialize message." << std::endl;
        return;
    }
    int status = mqttClient.pushMessage(topic, payload);
    if (status == 0) {
        std::cout << "Message is published to topic " << topic << std::endl;
    } else {
        std::cerr << "Failed to publish message to topic " << topic << std::endl;
    }
}

void aggretation(std::string aoiPath, MQTTClientHandler mttqClient, std::string topicCounts, std::string topicXYs) {
    std::vector<AreaOfInterest> aois;
    readAoIs(aoiPath, aois);
    Aggregator aggregarot = Aggregator({SENSOR_SERIAL});
    aggregarot.setAoIs(SENSOR_SERIAL, aois);

    int ts;
    std::vector<AreaCount> areaCounts;
    std::vector<XY> xys;
    while (true) {
        std::cout << "Aggregating start" << std::endl;
        ts = getTimestampSec();
        aggregarot.checkAoIs(ts, areaCounts);
        aggregarot.getXYs(ts, xys);
        std::cout << "Aggregation done " << areaCounts.size() << " " << xys.size() << std::endl;
        if (!areaCounts.empty()) {
            sendCounts(mttqClient, topicCounts, ts, areaCounts);
        }
        if (!xys.empty()) {
            sendXYs(mttqClient, topicXYs, ts, xys);
        }
        usleep(1e6);
    }
}

int main() {
    INIReader reader(COFIG_FILE);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load 'config.ini'\n";
        return 1;
    }

    preprocessor = new PreProcessor(reader.Get("sensor_params", "path", "UNKNOWN"));

    MQTTClientHandler mqttClient = MQTTClientHandler(
        "GWSubscriber",
        reader.Get("mosquitto", "address", "UNKNOWN"),
        {reader.Get("mqtt.topics", "telemetry", "UNKNOWN")},
        onMessage
    );

    std::vector<std::thread> pool;
    pool.push_back(std::thread(
        aggretation,
        reader.Get("aoi", "path", "UNKNOWN"),
        mqttClient,
        reader.Get("mqtt.topics", "count", "UNKNOWN"),
        reader.Get("mqtt.topics", "xy", "UNKNOWN")
    ));
    pool.push_back(std::thread(removeOutdatedData));

    for (auto& t : pool) {
        if (t.joinable()) {
            t.join();
        }
    }

    while (true) {
        usleep(1e6);
    }

    return 0;
}