import math

from .geometry import Point

MAX_UNDISTORT_ITRS = 10

class CameraCalibrationParameters:
    def __init__(self, cx, cy, fx, fy, k0, k1, k2, k3):
        self.CX = cx
        self.CY = cy
        self.FX = fx
        self.FY = fy
        self.K0 = k0
        self.K1 = k1
        self.K2 = k2
        self.K3 = k3
        self.FX_INV = 1.0 / fx
        self.FY_INV = 1.0 / fy
        self.EPS2 = min(self.FX_INV, self.FY_INV) ** 2

def camera_calibration_parameters_to_string(params):
    return f"""
    CX: {params.CX}
    CY: {params.CY}
    FX: {params.FX}
    FY: {params.FY}
    K0: {params.K0}
    K1: {params.K1}
    K2: {params.K2}
    K3: {params.K3}
    """

class FisheyeInternal:
    @staticmethod
    def normalize(point_in, params):
        return Point((point_in.x - params.CX) * params.FX_INV, (point_in.y - params.CY) * params.FY_INV)

    @staticmethod
    def denormalize(point_in, params):
        return Point(point_in.x * params.FX + params.CX, point_in.y * params.FY + params.CY)

    @staticmethod
    def distort_normalized(point_in, params):
        r = point_in.x ** 2 + point_in.y ** 2
        if r < params.EPS2:
            return point_in

        r = math.sqrt(r)
        t = math.atan(r)
        t2 = t ** 2
        t3 = t2 * t
        t5 = t3 * t2
        t7 = t5 * t2

        a = (t + params.K0 * t3 + params.K1 * t5 + params.K2 * t7 + params.K3 * t7 * t2) / r
        return Point(point_in.x * a, point_in.y * a)

def distort(point_in, params):
    point_out = FisheyeInternal.normalize(point_in, params)
    point_out = FisheyeInternal.distort_normalized(point_out, params)
    return FisheyeInternal.denormalize(point_out, params)

def undistort(point_in, params):
    p = FisheyeInternal.normalize(point_in, params)
    guess = Point(p.x, p.y)
    for _ in range(MAX_UNDISTORT_ITRS):
        dstrt = FisheyeInternal.distort_normalized(guess, params)
        dx = p.x - dstrt.x
        dy = p.y - dstrt.y
        guess.x += dx
        guess.y += dy
        if dx ** 2 + dy ** 2 < params.EPS2:
            break
    return FisheyeInternal.denormalize(guess, params)

def project(point_in, cam_height_cm, height_cm, params):
    factor = cam_height_cm / (cam_height_cm - height_cm) if height_cm > 0 else (cam_height_cm + height_cm) / cam_height_cm
    point_out = undistort(point_in, params)
    point_out.x = factor * (point_out.x - params.CX) + params.CX
    point_out.y = factor * (point_out.y - params.CY) + params.CY
    return distort(point_out, params)

def pixel2cm(point_in, cam_height_cm, params):
    point_out = undistort(point_in, params)
    point_out.x = (point_out.x - params.CX) * params.FX_INV * cam_height_cm
    point_out.y = (params.CY - point_out.y) * params.FY_INV * cam_height_cm
    return point_out

def cm2pixel(point_in, cam_height_cm, height_cm, params):
    point_out = Point(point_in.x, point_in.y)
    if height_cm != 0:
        factor = cam_height_cm / (cam_height_cm - height_cm)
        point_out.x *= factor
        point_out.y *= factor
    cam_height_inv = 1.0 / cam_height_cm
    point_out.x = point_out.x * params.FX * cam_height_inv + params.CX
    point_out.y = -point_out.y * params.FY * cam_height_inv + params.CY
    return distort(point_out, params)
