import json
import time
import configparser
from multiprocessing import Process

from proto.sensor_payload_pb2 import Notification
from mqtt_client import MQTTClient
from preprocessor import Preprocessor
from preprocessing_utils import Detection, Rectangle, Point
from aoi import AreaOfInterest
from aggregator import Aggregator
from xy_storage import XYStorage


class PreprocessorWrapper:
    def __init__(self, data_pth: str) -> None:
        self._preprocessor = Preprocessor(data_pth)

    def on_message(self, client, userdata, message):
        notification = Notification()
        notification.ParseFromString(message.payload)

        detections = []
        for obj in notification.objects:
            detections.append(Detection(
                Rectangle.from_xywh(obj.box_x, obj.box_y, obj.box_w, obj.box_h),
                Point(obj.position_x, obj.position_y),
                obj.confidence,
            ))
        print("Received message: ", notification.timestamp)
        self._preprocessor.preprocess_detection(detections, "0", notification.timestamp)


def preprocessing_worker():
    config = configparser.ConfigParser()
    config.read('config.ini')

    preprocessor = PreprocessorWrapper(config["sensor_params"]["path"])
    client = MQTTClient(
        "GW",
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
            print('Connection failed, exiting...')

    finally:
        print("Closing connection")
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

def aggregator():
    config = configparser.ConfigParser()
    config.read('config.ini')

    aois = get_aois(config["aoi"]["path"])
    aggregator = Aggregator()
    aggregator.set_aoi("0", aois)

    while True:
        counts = aggregator.check_aois()
        if len(counts) > 0:
            print(counts)
        xys = aggregator.get_xys()
        print(xys)
        time.sleep(1)

def remove_outdated_data():
    storage = XYStorage()
    th_min = 1
    while True:
        print("Removing outdated data points", int(time.time()))
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
