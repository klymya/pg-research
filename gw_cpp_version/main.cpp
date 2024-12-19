#include <iostream>
#include <fstream>
#include <unistd.h>
#include "sensor_payload.pb.h"
#include "MQTTClientHandler.hpp"

// publisher
// int main() {
//     // Create a SensorPayload message
//     telemetrypayload::XYNotification payload;
//     payload.set_timestamp(1);
//     telemetrypayload::XYObject* object = payload.add_objects();
//     object->set_x(9);
//     object->set_y(9);

//     // Serialize the message to a string
//     std::string serialized_payload;
//     if (!payload.SerializeToString(&serialized_payload)) {
//         std::cerr << "Failed to serialize payload." << std::endl;
//         return -1;
//     }

//     MQTTClientHandler mqttClient = MQTTClientHandler("cppPub", "tcp://localhost:1883");
//     mqttClient.pushMessage("xy/0", serialized_payload);
//     usleep(3 * 1e6);

//     return 0;
// }



// subscriber

int onMessage(void* context, char* topicName, int topicLen, MQTTClient_message* message) {
    std::string payload(static_cast<char*>(message->payload), message->payloadlen);

    telemetrypayload::TelemetryNotification msg;
    if (!msg.ParseFromString(payload)) {
        std::cerr << "Failed to parse payload." << std::endl;
        return 1;
    }

    std::cout << "Received message on topic " << topicName << std::endl;
    std::cout << "Timestamp: " << msg.timestamp() << std::endl;
    for (int i = 0; i < msg.objects_size(); ++i) {
        const telemetrypayload::ReportingObject& object = msg.objects(i);
        std::cout << "Object " << i << ": (" <<  object.confidence() << ", " << object.position_x() << ", " << object.position_y() << ")" << std::endl;
    }

    return 1;
}


int main() {
    MQTTClientHandler mqttClient = MQTTClientHandler("cppSub", "tcp://localhost:1883", {"telemetry/0"}, onMessage);
    usleep(10 * 1e6);

    return 0;
}
 