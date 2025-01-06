import time
import logging
from random import randrange

from proto.sensor_payload_pb2 import TelemetryNotification, Notification
from mqtt_client import MQTTClient
from config import get_config
from logconfig import configure_logging


configure_logging()

def get_notification():
    telemetry_notification = TelemetryNotification()

    telemetry_notification.timestamp = int(time.time())

    obj = telemetry_notification.objects.add()
    obj.confidence = 0.9
    obj.position_x = 480 + randrange(0, 5)
    obj.position_y = 200 + randrange(0, 5)
    obj.box_x = 470
    obj.box_y = 190
    obj.box_w = 20
    obj.box_h = 20

    obj = telemetry_notification.objects.add()
    obj.confidence = 0.8
    obj.position_x = 190 + randrange(0, 5)
    obj.position_y = 260 + randrange(0, 5)
    obj.box_x = 180
    obj.box_y = 250
    obj.box_w = 20
    obj.box_h = 20

    obj = telemetry_notification.objects.add()
    obj.confidence = 0.8
    obj.position_x = 220 + randrange(0, 5)
    obj.position_y = 260 + randrange(0, 5)
    obj.box_x = 210
    obj.box_y = 250
    obj.box_w = 20
    obj.box_h = 20

    notification = Notification(telemetry=telemetry_notification)
    return notification


if __name__ == "__main__":
    config = get_config()

    client = MQTTClient(
        "SensorStub",
        config["mosquitto"]["host"],
        config.getint("mosquitto", "port"),
    )

    try:
        while True:
            time.sleep(5)
            notification = get_notification()
            proto_msg = notification.SerializeToString()
            result = client.push_message(config["mqtt.topics"]["telemetry"], proto_msg)
            status = result[0]
            if status == 0:
                logging.info("Message is published to topic " + config["mqtt.topics"]["telemetry"])
            else:
                logging.warning("Failed to send message to topic " + config["mqtt.topics"]["telemetry"])
                if not client.is_connected():
                    logging.info("Client not connected, exiting...")
                    break
    finally:
        client.close()
