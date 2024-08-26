import json
import math

import numpy as np
import cv2


DEFAULT_ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE = 0.275

# https://learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        print("non singular")
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


# def rotationMatrixToEulerAngles(R):
#     sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
#     singular = sy < 1e-6

#     if not singular:
#         x = np.arctan2(R[2, 1], R[2, 2])  # Roll
#         y = np.arctan2(-R[2, 0], sy)      # Pitch
#         z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
#     else:
#         x = np.arctan2(-R[1, 2], R[1, 1]) # Roll
#         y = np.arctan2(-R[2, 0], sy)      # Pitch
#         z = 0                             # Yaw

#     return np.array([x, y, z])


def get_camera_mtxs(name="Cam3"):
    with open("cam_params.json", "rt") as f:
        params = json.load(f)

    cam = params[name]["LENS_PARAMS"]

    K = np.array([
        [cam["FX"], 0, cam["CX"]],
        [0, cam["FY"], cam["CY"]],
        [0, 0, 1],
    ])
    D = np.array([cam["K0"], cam["K1"], cam["K2"], cam["K3"]])[:, None]

    return K, D


class ArucoDetector:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(DEFAULT_ARUCO_DICT)
        self.parameters = cv2.aruco.DetectorParameters()

        self.K, self.D = get_camera_mtxs()

    def detect(self, img_gray):
        corners, ids, _ = cv2.aruco.detectMarkers(img_gray, dictionary=self.dictionary, parameters=self.parameters)

        idxs = np.argsort(ids.flatten())
        corners = np.array(corners)[idxs]
        ids = ids[idxs]

        return corners, ids

    def estimate(self, img_gray):
        corners, ids = self.detect(img_gray)

        # If markers are detected
        if np.all(ids is not None):
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, self.K, self.D)
            euler_rvecs = []

            for i in range(len(ids)):
                # R = rodrigues(rvecs[i].flatten())
                R = cv2.Rodrigues(rvecs[i].flatten())[0]
                euler_angles = rotationMatrixToEulerAngles(R)

                euler_angles_degrees = np.degrees(euler_angles)
                euler_rvecs.append(euler_angles_degrees)

                # print(f"Marker ID: {ids[i]} - Yaw: {euler_angles_degrees[2]:.2f}, Pitch: {euler_angles_degrees[1]:.2f}, Roll: {euler_angles_degrees[0]:.2f}")

        euler_angles_degrees = np.array(euler_angles_degrees)

        return ids, corners, tvecs, rvecs, euler_rvecs
