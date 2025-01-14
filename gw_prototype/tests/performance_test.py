from functools import partial
import os
import time
import json
from typing import Callable
import pathlib
import shutil

from geometry import Rectangle, Point
from preprocessing_utils import Detection
from preprocessor import Preprocessor
from aggregator import Aggregator
from geometry import Point
from aoi import AreaOfInterest
from common import DEFAULT_CAMERA_CONFIG, DEFAULT_CALIBRATION_PARAMS


N = 100


def tmp_json_file(tmp_path):
    data = {
        "sensor_1": {
            "LENS_PARAMS": {
                "CX": DEFAULT_CALIBRATION_PARAMS.CX,
                "CY": DEFAULT_CALIBRATION_PARAMS.CY,
                "FX": DEFAULT_CALIBRATION_PARAMS.FX,
                "FY": DEFAULT_CALIBRATION_PARAMS.FY,
                "K0": DEFAULT_CALIBRATION_PARAMS.K0,
                "K1": DEFAULT_CALIBRATION_PARAMS.K1,
                "K2": DEFAULT_CALIBRATION_PARAMS.K2,
                "K3": DEFAULT_CALIBRATION_PARAMS.K3,
                "FOV_X": DEFAULT_CAMERA_CONFIG.fov_angle_x,
                "FOV_Y": DEFAULT_CAMERA_CONFIG.fov_angle_x
            },
            "HEIGHT": DEFAULT_CAMERA_CONFIG.height_cm,
            "IMG_W": DEFAULT_CAMERA_CONFIG.image_w,
            "IMG_H": DEFAULT_CAMERA_CONFIG.image_h,
        }
    }
    file_path = tmp_path / "sensors.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    return file_path


def measure_time_consuming(f: Callable, n: int = N) -> float:
    total_consumed_time = 0
    start = time.time()
    for _ in range(n):
        f()
    end = time.time()
    total_consumed_time += end - start
    return total_consumed_time / n


def preprocessing() -> None:
    tmp_db = "tmp.db"
    tmp_folder = pathlib.Path("tmp")
    os.makedirs(tmp_folder, exist_ok=True)
    data_pth = tmp_json_file(tmp_folder)

    preprocessor = Preprocessor(data_pth, tmp_db)

    n = 1
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9),
    ]
    f = partial(preprocessor.preprocess_detection, detections, "sensor_1")

    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.3f} sec")

    n = 10
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
        for _ in range(n)
    ]
    f = partial(preprocessor.preprocess_detection, detections, "sensor_1")

    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.3f} sec")
    
    n = 100
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
        for _ in range(n)
    ]
    f = partial(preprocessor.preprocess_detection, detections, "sensor_1")

    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.3f} sec")

    shutil.rmtree(tmp_folder)
    os.remove(tmp_db)

def xy() -> None:
    tmp_db = "tmp.db"
    tmp_folder = pathlib.Path("tmp")
    os.makedirs(tmp_folder, exist_ok=True)
    data_pth = tmp_json_file(tmp_folder)

    aggregator = Aggregator(["sensor_1"], tmp_db)
    preprocessor = Preprocessor(data_pth, tmp_db)

    n = 1
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
    ]
    preprocessor.preprocess_detection(detections, "sensor_1", ts=59)

    f = partial(aggregator.get_xys, 60)
    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.6f} sec")

    n = 10
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
        for _ in range(n)
    ]
    preprocessor.preprocess_detection(detections, "sensor_1", ts=59)

    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.6f} sec")

    n = 100
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
        for _ in range(n)
    ]
    preprocessor.preprocess_detection(detections, "sensor_1", ts=59)

    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.6f} sec")

    n = 1000
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
        for _ in range(n)
    ]
    preprocessor.preprocess_detection(detections, "sensor_1", ts=59)

    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.6f} sec")

    shutil.rmtree(tmp_folder)
    os.remove(tmp_db)


def count_one_aoi() -> None:
    tmp_db = "tmp.db"
    tmp_folder = pathlib.Path("tmp")
    os.makedirs(tmp_folder, exist_ok=True)
    data_pth = tmp_json_file(tmp_folder)

    aggregator = Aggregator(["sensor_1"], tmp_db)
    aois = [
        AreaOfInterest(
            area_id="1",
            name="tst",
            polygon=[Point(-100, 100), Point(0, 100), Point(0, 0), Point(-100, 0)],
            reporting_period_sec=60
        ),
    ]
    aggregator.set_aois("sensor_1", aois)
    preprocessor = Preprocessor(data_pth, tmp_db)

    n = 1
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
    ]
    preprocessor.preprocess_detection(detections, "sensor_1", ts=50)

    f = partial(aggregator.check_aois, 60)
    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.6f} sec")

    n = 10
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
        for _ in range(n)
    ]
    preprocessor.preprocess_detection(detections, "sensor_1", ts=50)

    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.6f} sec")

    n = 100
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
        for _ in range(n)
    ]
    preprocessor.preprocess_detection(detections, "sensor_1", ts=50)

    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.6f} sec")

    n = 1000
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
        for _ in range(n)
    ]
    preprocessor.preprocess_detection(detections, "sensor_1", ts=50)

    t = measure_time_consuming(f)
    print(f"Detections: {n:>4}. Elapsed time: {t:.6f} sec")

    shutil.rmtree(tmp_folder)
    os.remove(tmp_db)


def count_many_aoi() -> None:
    tmp_db = "tmp.db"
    tmp_folder = pathlib.Path("tmp")
    os.makedirs(tmp_folder, exist_ok=True)
    data_pth = tmp_json_file(tmp_folder)

    aggregator = Aggregator(["sensor_1"], tmp_db)
    preprocessor = Preprocessor(data_pth, tmp_db)
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9)
        for _ in range(10)
    ]
    preprocessor.preprocess_detection(detections, "sensor_1", ts=50)

    n = 1
    aois = [
        AreaOfInterest(
            area_id="1",
            name="tst",
            polygon=[Point(-100, 100), Point(0, 100), Point(0, 0), Point(-100, 0)],
            reporting_period_sec=60
        ) for _ in range(n)
    ]
    aggregator.set_aois("sensor_1", aois)

    f = partial(aggregator.check_aois, 60)
    t = measure_time_consuming(f)
    print(f"AoIs: {n:>4}. Elapsed time: {t:.6f} sec")

    n = 10
    aois = [
        AreaOfInterest(
            area_id="1",
            name="tst",
            polygon=[Point(-100, 100), Point(0, 100), Point(0, 0), Point(-100, 0)],
            reporting_period_sec=60
        ) for _ in range(n)
    ]
    aggregator.set_aois("sensor_1", aois)

    f = partial(aggregator.check_aois, 60)
    t = measure_time_consuming(f)
    print(f"AoIs: {n:>4}. Elapsed time: {t:.6f} sec")

    n = 100
    aois = [
        AreaOfInterest(
            area_id="1",
            name="tst",
            polygon=[Point(-100, 100), Point(0, 100), Point(0, 0), Point(-100, 0)],
            reporting_period_sec=60
        ) for _ in range(n)
    ]
    aggregator.set_aois("sensor_1", aois)

    f = partial(aggregator.check_aois, 60)
    t = measure_time_consuming(f)
    print(f"AoIs: {n:>4}. Elapsed time: {t:.6f} sec")

    shutil.rmtree(tmp_folder)
    os.remove(tmp_db)


if __name__ == "__main__":
    print("PREPROCESSING PERFORMANCE TEST")
    preprocessing()
    print("====================================")
    print("XY PERFORMANCE TEST")
    xy()
    print("====================================")
    print("COUNT ONE AOI PERFORMANCE TEST")
    count_one_aoi()
    print("====================================")
    print("COUNT MANY AOI PERFORMANCE TEST")
    count_many_aoi()
    print("====================================")
