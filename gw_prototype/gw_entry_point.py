import json
import time
from multiprocessing import Process
import logging

from proto.sensor_payload_pb2 import TelemetryNotification, AoICountNotification, XYNotification
from mqtt_client import MQTTClient
from preprocessor import Preprocessor
from preprocessing_utils import Detection, Rectangle, Point
from aoi import AreaOfInterest
from aggregator import Aggregator
from xy_storage import XYStorage
from logconfig import configure_logging
from config import get_config

configure_logging()


class PreprocessorWrapper:
    def __init__(self, data_pth: str) -> None:
        self._preprocessor = Preprocessor(data_pth)

    def on_message(self, client, userdata, message):
        notification = TelemetryNotification()
        notification.ParseFromString(message.payload)

        detections = []
        for obj in notification.objects:
            detections.append(Detection(
                Rectangle.from_xywh(obj.box_x, obj.box_y, obj.box_w, obj.box_h),
                Point(obj.position_x, obj.position_y),
                obj.confidence,
            ))
        logging.debug(f"Received message from  topic {message.topic} with ts: {notification.timestamp}")
        self._preprocessor.preprocess_detection(detections, "0", notification.timestamp)


def preprocessing_worker():
    config = get_config()

    preprocessor = PreprocessorWrapper(config["sensor_params"]["path"])
    client = MQTTClient(
        "GWPublisher",
        config["mosquitto"]["host"],
        config.getint("mosquitto", "port"),
        [(config["mqtt.topics"]["telemetry"], 0)],
        preprocessor.on_message
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


def get_aois(pth) -> list[AreaOfInterest]:
    with open(pth, 'r') as f:
       data = json.load(f)

    aois = []
    for k, raw_aoi in data.items():
        aoi = AreaOfInterest(
            k,
            raw_aoi["name"],
            [Point(*p) for p in raw_aoi["polygon"]],
            raw_aoi["reporting_period"],
        )
        aois.append(aoi)
    return aois


def send_counts(client: MQTTClient, topic: str, ts: int, counts: list[tuple[str, str, int]]):
    notification = AoICountNotification()
    notification.timestamp = ts
    for _, aoi_id, count in counts:
        object = notification.objects.add()
        object.aoi_id = aoi_id
        object.count = count
    proto_msg = notification.SerializeToString()
    result = client.push_message(topic, proto_msg)
    status = result[0]
    if status == 0:
        logging.debug(f"Message is published to topic {topic}")
    else:
        logging.warning(f"Failed to send message to topic {topic}")


def send_xys(client: MQTTClient, topic: str, ts: int, xys: list[tuple[str, int, int]]):
    notification = XYNotification()
    notification.timestamp = ts
    for _, x, y in xys:
        object = notification.objects.add()
        object.x = x
        object.y = y
    proto_msg = notification.SerializeToString()
    result = client.push_message(topic, proto_msg)
    status = result[0]
    if status == 0:
        logging.debug(f"Message is published to topic {topic}")
    else:
        logging.warning(f"Failed to send message to topic {topic}")


def aggregator():
    config = get_config()

    client = MQTTClient(
        "GWSubscriber",
        config["mosquitto"]["host"],
        config.getint("mosquitto", "port"),
    )

    aois = get_aois(config["aoi"]["path"])
    aggregator = Aggregator(["0"])
    aggregator.set_aois("0", aois)

    while True:
        ts, counts = aggregator.check_aois()
        if len(counts) > 0:
            send_counts(client, config["mqtt.topics"]["count"], ts, counts)
        ts, xys = aggregator.get_xys()
        if len(xys) > 0:
            send_xys(client, config["mqtt.topics"]["xy"], ts, xys)
        time.sleep(1)

def remove_outdated_data():
    storage = XYStorage()
    th_min = 1
    while True:
        logging.debug("Removing outdated data points")
        storage.remove_old_data_points(th_min)
        time.sleep(th_min * 60)


if __name__ == "__main__":
    pool = [
        Process(target=preprocessing_worker),
        Process(target=aggregator),
        Process(target=remove_outdated_data),
    ]
    for p in pool:
        p.start()
    for p in pool:
        p.join()
