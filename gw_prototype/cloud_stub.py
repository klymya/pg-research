import time
import logging

from proto.sensor_payload_pb2 import AoICountNotification, XYNotification
from mqtt_client import MQTTClient
from config import get_config
from logconfig import configure_logging


configure_logging()

CONFIG = get_config()

def count_notification_to_str(notification: AoICountNotification) -> str:
    output = ""
    output += f"ts: {notification.timestamp}. "
    for obj in notification.objects:
        output += f"aoi_id: {obj.aoi_id}, count: {obj.count}. "
    return output

def xy_notification_to_str(notification: XYNotification) -> str:
    output = ""
    output += f"ts: {notification.timestamp}. "
    for obj in notification.objects:
        output += f"x: {obj.x}, y: {obj.y}. "
    return output

def on_message(client, userdata, message):
    if message.topic == CONFIG["mqtt.topics"]["count"]:
        notification = AoICountNotification()
        notification.ParseFromString(message.payload)
        logging.info(f"Received message from topic {message.topic}: {count_notification_to_str(notification)}")
    elif message.topic == CONFIG["mqtt.topics"]["xy"]:
        notification = XYNotification()
        notification.ParseFromString(message.payload)
        logging.info(f"Received message from topic {message.topic}: {xy_notification_to_str(notification)}")

if __name__ == "__main__":
    client = MQTTClient(
        "CloudStub",
        CONFIG["mosquitto"]["host"],
        CONFIG.getint("mosquitto", "port"),
        [(CONFIG["mqtt.topics"]["count"], 0), (CONFIG["mqtt.topics"]["xy"], 0)],
        on_message
    )
    time.sleep(1)
    try:
        while client.is_connected() == True:
            time.sleep(1)
        if client.is_connected() == False:
            logging.warning('Connection failed, exiting...')

    finally:
        logging.info("Closing connection")
        client.close()
