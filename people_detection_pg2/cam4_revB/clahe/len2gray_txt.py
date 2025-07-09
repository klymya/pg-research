import cv2

img = cv2.imread('lenna.png', cv2.IMREAD_UNCHANGED)
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = img_gray.flatten()

with open('lenna_gray.txt', 'w') as f:
    for value in img_gray:
        f.write(f"{value} ")


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = img_rgb.flatten()

with open('lenna_rgb.txt', 'w') as f:
    for value in img_rgb:
        f.write(f"{value} ")
