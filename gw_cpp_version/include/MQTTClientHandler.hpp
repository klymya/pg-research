#include <string>
#include <vector>
#include <utility>
#include "MQTTClient.h"

class MQTTClientHandler {
public:
    using MessageCallback = int(*)(void*, char*, int, MQTTClient_message*);

    MQTTClientHandler(const std::string& clientId, const std::string& serverAdress,
                      const std::vector<std::string>& topics = {}, MessageCallback onMessage = nullptr);
    ~MQTTClientHandler();

    bool isConnected() const;
    int pushMessage(const std::string& topic, const std::string& message);
    void setDeliveryToken(MQTTClient_deliveryToken token);
    void close();

private:
    static void onConnect(void* context, MQTTClient_deliveryToken dt);
    static void onConnectionLost(void* context, char* cause);
    static void onMessageDelivered(void* context, MQTTClient_deliveryToken dt);
    static int onMessageArrivedStub(void* context, char* topicName, int topicLen, MQTTClient_message* message);

    std::string m_clientId;
    std::string m_serverAddress;
    MQTTClient m_client;
    bool m_connected;
    MQTTClient_deliveryToken m_deliveredToken;
};
