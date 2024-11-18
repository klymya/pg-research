from fisheye_camera import CameraCalibrationParameters
from detection_area import CameraConfiguration

# cam3
DEFAULT_CALIBRATION_PARAMS = CameraCalibrationParameters(
    332.663306034164,
    247.362736041039,
    201.530310535271,
    201.753479085099,
    0.351145527648731,
    -0.0859250956571053,
    -0.0056549593712587,
    0.00646710057070369
)
W = 640
H = 480
CAMERA_H = 275.0
TABLE_H = 70.0

DEFAULT_CAMERA_CONFIG = CameraConfiguration(
    DEFAULT_CALIBRATION_PARAMS,
    135.0,
    109.0,
    CAMERA_H,
    W,
    H
)
