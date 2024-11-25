from dataclasses import dataclass

from detection_area import DetectionArea, CameraConfiguration
from geometry import Point, Rectangle
from fisheye_camera import CameraCalibrationParameters, pixel2cm


@dataclass
class Detection:
    bbox: Rectangle
    position: Point
    confidence: float


def filter_detections_by_da(detections: list[Detection], da: DetectionArea) -> list[Detection]:
    filtered_detections = []
    for detection in detections:
        if da.is_rect_intersect_px(detection.bbox):
            filtered_detections.append(detection)

    return filtered_detections


def detection2xy(
    detections: list[Detection], calibration_params: CameraCalibrationParameters, cam_h_cm: int
) -> list[Point]:
    points = [pixel2cm(detection.position, cam_h_cm, calibration_params) for detection in detections]
    for p in points:
        p.x = round(p.x)
        p.y = round(p.y)

    return points


def preprocessing(detections: list[Detection], da: DetectionArea, cam_config: CameraConfiguration) -> list[Point]:
    fildeterd_detections = filter_detections_by_da(detections, da)
    points = detection2xy(fildeterd_detections, cam_config.calibration_params, cam_config.height_cm)
    return points
