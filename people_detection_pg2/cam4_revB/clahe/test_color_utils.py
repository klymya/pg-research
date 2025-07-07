import cv2
import os
import numpy as np

import color_utils
import color_utils_pure

TMP_DIR = os.path.dirname(os.path.abspath(__file__))


img = cv2.imread(os.path.join(TMP_DIR, "lenna.png"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_lab_ref = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img_rgb_ref = cv2.cvtColor(img_lab_ref, cv2.COLOR_LAB2RGB)

img_lab = color_utils.rgb2lab(img)
img_rgb = color_utils.lab2rgb(img_lab)

assert np.allclose(img_lab, img_lab_ref, 1e-1), "LAB conversion mismatch"
assert np.allclose(img_rgb.astype(int), img_rgb_ref.astype(int), 5), "RGB conversion mismatch"

img_lab = color_utils_pure.rgb_image_to_lab(img.tolist())
img_rgb = color_utils_pure.lab_image_to_rgb(img_lab)

img_lab = np.array(img_lab, dtype=np.float32)
img_rgb = np.array(img_rgb, dtype=np.uint8)

assert np.allclose(img_lab, img_lab_ref, 1e-1), "LAB conversion mismatch (pure)"
assert np.allclose(img_rgb.astype(int), img_rgb_ref.astype(int), 5), "RGB conversion mismatch (pure)"
