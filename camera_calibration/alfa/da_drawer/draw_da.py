import json
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gw_prototype.fisheye_camera import CameraCalibrationParameters, cm2pixel
from gw_prototype.detection_area import CameraConfiguration, DetectionArea
from gw_prototype.geometry import Point


parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--height_cm", type=int, required=True)
args = parser.parse_args()

img = cv2.imread(args.img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 480))

with open("cam4_2_rev_b_main.json", "rt") as f:
    data = json.load(f)


calibration_params = CameraCalibrationParameters(
    data["LENS_PARAMS"]["CX"],
    data["LENS_PARAMS"]["CY"],
    data["LENS_PARAMS"]["FX"],
    data["LENS_PARAMS"]["FY"],
    data["LENS_PARAMS"]["K0"],
    data["LENS_PARAMS"]["K1"],
    data["LENS_PARAMS"]["K2"],
    data["LENS_PARAMS"]["K3"],
)

cam_config = CameraConfiguration(
    calibration_params,
    data["LENS_PARAMS"]["FOV_X"],
    data["LENS_PARAMS"]["FOV_Y"],
    args.height_cm,
    data["IMG_W"],
    data["IMG_H"]
)

da_cam3 = DetectionArea(cam_config)

poly = np.array(da_cam3.polygon_in_pixels())


center_p = Point(0, 0)
center_p_px = cm2pixel(center_p, args.height_cm, 0, calibration_params)


da_cam3 = DetectionArea(cam_config, True)
poly = np.array(da_cam3.polygon_in_pixels())
plt.imshow(img)
plt.plot(poly[:, 0], poly[:, 1], 'b-', label=f"DA size = {round(da_cam3.m_rectangle_cm.h * da_cam3.m_rectangle_cm.w / 10000)} m^2")
plt.plot(center_p_px.x, center_p_px.y, 'bo', label="camera center")
plt.ylim(480, 0)
plt.title("Detection area")
plt.legend(loc='upper left')
plt.savefig("da_viz.png")
plt.close()
