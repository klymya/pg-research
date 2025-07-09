import cv2
import os
import numpy as np

import clahe
import clahe_pure

TMP_DIR = os.path.dirname(os.path.abspath(__file__))


img = cv2.imread(os.path.join(TMP_DIR, "lenna.png"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lab = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

clahe_img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l_gt = clahe_img.apply(l)

l_pred = clahe.apply_clahe_channel(l, clip_limit=3.0, tile_grid_size=(8, 8), nbins=256)

diff = np.abs(l_pred.astype(int) - l_gt.astype(int))
assert np.mean(diff) < 20, "CLAHE channel mismatch"
assert np.median(diff) < 15, "CLAHE channel mismatch"

l_pred = clahe_pure.apply_clahe_channel(l, clip_limit=3.0, tile_grid_size=(8, 8), nbins=256)
l_pred = np.array(l_pred)

diff = np.abs(l_pred.astype(int) - l_gt.astype(int))
assert np.mean(diff) < 20, "CLAHE channel mismatch"
assert np.median(diff) < 15, "CLAHE channel mismatch"
