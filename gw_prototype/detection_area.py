import numpy as np
from geometry import Point, Rectangle, is_point_inside_rectangle, calculate_iou
from fisheye_camera import cm2pixel, CameraCalibrationParameters

MIN_VISIBLE_HEIGHT_CM = 100
MAX_DETECTION_AREA_CM2 = 500000
DA_INTERSECTION_TH = 0


class DetectionAreaInternal:
    @staticmethod
    def deg2rad(x):
        return x * np.pi / 180.0


class CameraConfiguration:
    def __init__(self, calibration_params: CameraCalibrationParameters, fov_angle_x, fov_angle_y, height_cm, image_w, image_h):
        self.calibration_params: CameraCalibrationParameters = calibration_params
        self.fov_angle_x = fov_angle_x
        self.fov_angle_y = fov_angle_y
        self.height_cm = height_cm
        self.image_w = image_w
        self.image_h = image_h


class DetectionArea:
    def __init__(self, cam: CameraConfiguration):
        self.m_cam = cam
        fov_x_at_focal_len = 2.0 * np.tan(DetectionAreaInternal.deg2rad(self.m_cam.fov_angle_x / 2.0))
        fov_y_at_focal_len = 2.0 * np.tan(DetectionAreaInternal.deg2rad(self.m_cam.fov_angle_y / 2.0))

        aspect_ratio = fov_x_at_focal_len / fov_y_at_focal_len

        width = round((self.m_cam.height_cm - MIN_VISIBLE_HEIGHT_CM) * fov_x_at_focal_len)
        height = round((self.m_cam.height_cm - MIN_VISIBLE_HEIGHT_CM) * fov_y_at_focal_len)

        if width * height > MAX_DETECTION_AREA_CM2:
            width = round(np.sqrt(MAX_DETECTION_AREA_CM2 * aspect_ratio))
            height = round(1 / aspect_ratio * height)

        self.m_rectangle_cm = Rectangle.from_xyxy(-width / 2, height / 2, width / 2, -height / 2)

        tl_cm = Point(self.m_rectangle_cm.x, self.m_rectangle_cm.y)
        br_cm = Point(self.m_rectangle_cm.x1, self.m_rectangle_cm.y1)

        tl_px = cm2pixel(tl_cm, self.m_cam.height_cm, 0, self.m_cam.calibration_params)
        br_px = cm2pixel(br_cm, self.m_cam.height_cm, 0, self.m_cam.calibration_params)

        self.m_rectangle_px = Rectangle.from_xyxy(tl_px.x, tl_px.y, br_px.x, br_px.y)

    def get_da_rectangle_cm(self):
        return self.m_rectangle_cm

    def get_da_rectangle_px(self):
        return self.m_rectangle_px

    def is_point_inside_px(self, p):
        return is_point_inside_rectangle(p, self.m_rectangle_px)

    def is_rect_intersect_px(self, r):
        return calculate_iou(r, self.m_rectangle_px) > DA_INTERSECTION_TH
