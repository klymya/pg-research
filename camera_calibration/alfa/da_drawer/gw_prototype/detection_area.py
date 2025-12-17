import numpy as np
from .geometry import Point, Rectangle, is_point_inside_rectangle, calculate_iou
from .fisheye_camera import cm2pixel, CameraCalibrationParameters

# Parameters for determining the margin we take from the FOV boundary
# for computing the detection area. For a narrow FOV we take a smaller
# margin, to allow a reasonable detection area size.
MIN_VISIBLE_HEIGHT_FOR_DA_NARROW_VFOV = 30 # (cm)
MIN_VISIBLE_HEIGHT_FOR_DA_WIDE_VFOV = 100 # (cm)
VFOV_WIDE_THRESH = 100 # (deg). FOV is considered 'narrow' if lower than this value.

MAX_DETECTION_AREA_CM2 = 500000
DA_INTERSECTION_TH = 0


class DetectionAreaInternal:
    @staticmethod
    def deg2rad(x):
        return x * np.pi / 180.0


class CameraConfiguration:
    def __init__(
        self, calibration_params: CameraCalibrationParameters, fov_angle_x, fov_angle_y, height_cm, image_w, image_h
    ) -> None:
        self.calibration_params: CameraCalibrationParameters = calibration_params
        self.fov_angle_x = fov_angle_x
        self.fov_angle_y = fov_angle_y
        self.height_cm = height_cm
        self.image_w = image_w
        self.image_h = image_h


class DetectionArea:
    def __init__(self, cam: CameraConfiguration, use_da_limit: bool = False):
        self.m_cam = cam
        fov_x_at_focal_len = 2.0 * np.tan(DetectionAreaInternal.deg2rad(self.m_cam.fov_angle_x / 2.0))
        fov_y_at_focal_len = 2.0 * np.tan(DetectionAreaInternal.deg2rad(self.m_cam.fov_angle_y / 2.0))

        aspect_ratio = fov_x_at_focal_len / fov_y_at_focal_len

        min_visible_height_cm = MIN_VISIBLE_HEIGHT_FOR_DA_NARROW_VFOV \
            if self.m_cam.fov_angle_y < VFOV_WIDE_THRESH else MIN_VISIBLE_HEIGHT_FOR_DA_WIDE_VFOV

        width = round((self.m_cam.height_cm - min_visible_height_cm) * fov_x_at_focal_len)
        height = round((self.m_cam.height_cm - min_visible_height_cm) * fov_y_at_focal_len)

        if width * height > MAX_DETECTION_AREA_CM2 and use_da_limit:
            width = round(np.sqrt(MAX_DETECTION_AREA_CM2 * aspect_ratio))
            height = round(width / aspect_ratio)

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

    def get_cm_polygon(self, closed=True):

        l = self.m_rectangle_cm.x
        r = self.m_rectangle_cm.x1
        t = self.m_rectangle_cm.y
        b = self.m_rectangle_cm.y1
        vertices = [[l, t], [r, t], [r, b], [l, b]]
        if closed:
            vertices += [vertices[0]]

        return vertices

    @staticmethod
    def _resample_polygon(vertices):

        STEP_SIZE_PX = 50.0  # resampling step

        num_of_vertices = len(vertices)
        resampled_vertices = []
        for i in range(num_of_vertices):
            v_curr = vertices[i]
            v_next = vertices[(i + 1) % num_of_vertices]
            diff = [v_next[0] - v_curr[0], v_next[1] - v_curr[1]]
            dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
            number_of_segments_in_edge = max(1, int(dist / STEP_SIZE_PX))
            for i_segment in range(number_of_segments_in_edge):
                segment_length = float(i_segment) / number_of_segments_in_edge
                resampled_vertices += [[v_curr[0] + segment_length * diff[0],
                                        v_curr[1] + segment_length * diff[1]]]

        return resampled_vertices

    def polygon_in_pixels(self):
        vertices = self.get_cm_polygon()
        resampled_vertices = self._resample_polygon(vertices)
        vertices_px = [cm2pixel(Point(*v), self.m_cam.height_cm, 0, self.m_cam.calibration_params) for v in resampled_vertices]
        vertices_px = [[int(v.x), int(v.y)] for v in vertices_px]

        return vertices_px
