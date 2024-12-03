import time
import logging
from random import randrange

from proto.sensor_payload_pb2 import TelemetryNotification
from mqtt_client import MQTTClient
from config import get_config
from logconfig import configure_logging


configure_logging()

def get_notification():
    notification = TelemetryNotification()

    notification.timestamp = int(time.time())

    obj = notification.objects.add()
    obj.confidence = 0.9
    obj.position_x = 480 + randrange(0, 5)
    obj.position_y = 200 + randrange(0, 5)
    obj.box_x = 470
    obj.box_y = 190
    obj.box_w = 20
    obj.box_h = 20

    obj = notification.objects.add()
    obj.confidence = 0.8
    obj.position_x = 190 + randrange(0, 5)
    obj.position_y = 260 + randrange(0, 5)
    obj.box_x = 180
    obj.box_y = 250
    obj.box_w = 20
    obj.box_h = 20

    obj = notification.objects.add()
    obj.confidence = 0.8
    obj.position_x = 220 + randrange(0, 5)
    obj.position_y = 260 + randrange(0, 5)
    obj.box_x = 210
    obj.box_y = 250
    obj.box_w = 20
    obj.box_h = 20

    return notification


def telemetry_notification_to_str(notification: TelemetryNotification) -> str:
    output = ""
    output += f"ts: {notification.timestamp}. "
    for obj in notification.objects:
        output += f"conf: {obj.confidence}, position: ({obj.position_x}, {obj.position_y}), box: ({obj.box_x}, {obj.box_y}, {obj.box_w}, {obj.box_h}). "
    return output

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
