#include <iostream>
#include <unistd.h>
#include "MQTTClientHandler.hpp"

MQTTClientHandler::MQTTClientHandler(const std::string& clientId, const std::string& serverAdress,
                                     const std::vector<std::string>& topics,
                                     MessageCallback onMessage)
    : m_serverAddress(serverAdress), m_clientId(clientId), m_connected(false)
{
    int rc;

    if ((rc = MQTTClient_create(&m_client, m_serverAddress.c_str(), m_clientId.c_str(), MQTTCLIENT_PERSISTENCE_NONE, nullptr)) != MQTTCLIENT_SUCCESS) {
        std::cerr << "Failed to create client, return code " << rc << std::endl;
        return;
    }

    MessageCallback onMsgCallack = nullptr;
    if (onMessage == nullptr && topics.empty()) {
        onMsgCallack = onMessageArrivedStub;
    } else {
        onMsgCallack = onMessage;
    }
    if ((rc = MQTTClient_setCallbacks(m_client, this, onConnectionLost, onMsgCallack, onMessageDelivered)) != MQTTCLIENT_SUCCESS) {
        std::cerr << "Failed to set callbacks, return code " << rc << std::endl;
        return;
    }

    MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
    conn_opts.keepAliveInterval = 20;
    conn_opts.cleansession = 1;

    if ((rc = MQTTClient_connect(m_client, &conn_opts)) != MQTTCLIENT_SUCCESS) {
        std::cerr << "Failed to connect, return code " << rc << std::endl;
        return;
    }

    m_connected = true;

    if (onMessage != nullptr && !topics.empty()) {
        std::vector<int> qos(topics.size(), 0);

        // Create a vector of char* to hold the C-style string pointers
        std::vector<char*> cstr_topics;
        cstr_topics.reserve(topics.size());

        // Populate the vector with pointers to the C-style strings
        for (const auto& topic : topics) {
            cstr_topics.push_back(const_cast<char*>(topic.c_str()));
        }

        if (rc = MQTTClient_subscribeMany(m_client, topics.size(), cstr_topics.data(), qos.data()) != MQTTCLIENT_SUCCESS) {
            std::cerr << "Failed to subscribe to topics, return code " << rc << std::endl;
        }
    }
}

MQTTClientHandler::~MQTTClientHandler() {
    close();
}

bool MQTTClientHandler::isConnected() const {
    return m_connected;
}

int MQTTClientHandler::pushMessage(const std::string& topic, const std::string& message) {
    MQTTClient_message pubmsg = MQTTClient_message_initializer;
    pubmsg.payload = (void*)message.c_str();
    pubmsg.payloadlen = (int)message.size();
    pubmsg.qos = 1;
    pubmsg.retained = 0;
    m_deliveredToken = 0;
    MQTTClient_deliveryToken token;
    int rc;

    if ((rc = MQTTClient_publishMessage(m_client, topic.c_str(), &pubmsg, &token)) != MQTTCLIENT_SUCCESS) {
        std::cerr << "Failed to publish message, return code " << rc << std::endl;
        return rc;
    }

    while (token != m_deliveredToken) {
        usleep(10000L);
    }

    return rc;
}

void MQTTClientHandler::close() {
    if (m_connected) {
        MQTTClient_disconnect(m_client, 10000);
        MQTTClient_destroy(&m_client);
        m_connected = false;
    }
}

void MQTTClientHandler::setDeliveryToken(MQTTClient_deliveryToken token) {
    m_deliveredToken = token;
}

void MQTTClientHandler::onConnect(void* context, MQTTClient_deliveryToken dt) {
    std::cout << "Message with token value " << dt << " delivery confirmed" << std::endl;
}

void MQTTClientHandler::onConnectionLost(void* context, char* cause) {
    std::cerr << "Connection lost, cause: " << cause << std::endl;
}

void MQTTClientHandler::onMessageDelivered(void* context, MQTTClient_deliveryToken dt) {
    std::cout << "Message with token value " << dt << " delivery confirmed" << std::endl;

    MQTTClientHandler* handler = static_cast<MQTTClientHandler*>(context);
    handler->setDeliveryToken(dt);
}

int MQTTClientHandler::onMessageArrivedStub(void* context, char* topicName, int topicLen, MQTTClient_message* message) {
    std::cout << "Message arrived" << std::endl;
    return 1;
}