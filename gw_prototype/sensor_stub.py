import time
import configparser

from proto.sensor_payload_pb2 import Notification
from mqtt_client import MQTTClient


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    notification = Notification()
    obj = notification.objects.add()
    obj.confidence = 0.9
    obj.position_x = 480
    obj.position_y = 200
    obj.box_x = 470
    obj.box_y = 190
    obj.box_w = 20
    obj.box_h = 20

    obj = notification.objects.add()
    obj.confidence = 0.8
    obj.position_x = 190
    obj.position_y = 260
    obj.box_x = 180
    obj.box_y = 250
    obj.box_w = 20
    obj.box_h = 20

    obj = notification.objects.add()
    obj.confidence = 0.8
    obj.position_x = 220
    obj.position_y = 260
    obj.box_x = 210
    obj.box_y = 250
    obj.box_w = 20
    obj.box_h = 20

    client = MQTTClient(
        "SensorStub",
        config["mosquitto"]["host"],
        config.getint("mosquitto", "port"),
    )

    try:
        while True:
            time.sleep(5)
            notification.timestamp = int(time.time())
            proto_msg = notification.SerializeToString()
            result = client.push_message(config["mqtt.topics"]["telemetry"], proto_msg)
            status = result[0]
            if status == 0:
                print("Message "+ str(proto_msg.hex()) + " is published to topic " + config["mqtt.topics"]["telemetry"])
            else:
                print("Failed to send message to topic " + config["mqtt.topics"]["telemetry"])
                if not client.is_connected():
                    print("Client not connected, exiting...")
                    break
    finally:
        client.close()
