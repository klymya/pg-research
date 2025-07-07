import numpy as np

# Constants for the reference white point (D65)
REF_X =  95.047
REF_Y = 100.000
REF_Z = 108.883

def rgb_to_xyz(rgb):
    rgb = rgb / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] /= 12.92

    rgb *= 100

    # sRGB to XYZ matrix (D65)
    x = rgb[..., 0]*0.4124 + rgb[..., 1]*0.3576 + rgb[..., 2]*0.1805
    y = rgb[..., 0]*0.2126 + rgb[..., 1]*0.7152 + rgb[..., 2]*0.0722
    z = rgb[..., 0]*0.0193 + rgb[..., 1]*0.1192 + rgb[..., 2]*0.9505

    return np.stack([x, y, z], axis=-1)

def xyz_to_lab(xyz):
    xyz[..., 0] /= REF_X
    xyz[..., 1] /= REF_Y
    xyz[..., 2] /= REF_Z

    epsilon = 0.008856
    kappa = 903.3

    mask = xyz > epsilon
    xyz[mask] = np.cbrt(xyz[mask])
    xyz[~mask] = (kappa * xyz[~mask] + 16) / 116

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    L = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    return np.stack([L, a, b], axis=-1)

def lab_to_xyz(lab):
    y = (lab[..., 0] + 16) / 116
    x = lab[..., 1] / 500 + y
    z = y - lab[..., 2] / 200

    xyz = np.stack([x, y, z], axis=-1)

    xyz = np.where(xyz ** 3 > 0.008856, xyz ** 3, (xyz - 16/116) / 7.787)

    xyz[..., 0] *= REF_X
    xyz[..., 1] *= REF_Y
    xyz[..., 2] *= REF_Z
    return xyz

def xyz_to_rgb(xyz):
    xyz /= 100  # normalize

    r = xyz[..., 0]*3.2406 + xyz[..., 1]*-1.5372 + xyz[..., 2]*-0.4986
    g = xyz[..., 0]*-0.9689 + xyz[..., 1]*1.8758 + xyz[..., 2]*0.0415
    b = xyz[..., 0]*0.0557 + xyz[..., 1]*-0.2040 + xyz[..., 2]*1.0570

    rgb = np.stack([r, g, b], axis=-1)

    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1/2.4)) - 0.055
    rgb[~mask] *= 12.92

    rgb = np.clip(rgb * 255, 0, 255)
    return rgb.astype(np.uint8)

# Interface functions
def rgb2lab_standard(img_rgb):
    return xyz_to_lab(rgb_to_xyz(img_rgb.astype(np.float32)))

def lab2rgb_standard(img_lab):
    return xyz_to_rgb(lab_to_xyz(img_lab.astype(np.float32)))


def rgb2lab(img_rgb):
    lab = xyz_to_lab(rgb_to_xyz(img_rgb.astype(np.float32)))
    lab[..., 0] = lab[..., 0] * 255 / 100          # L from [0,100] → [0,255]
    lab[..., 1:] = lab[..., 1:] + 128              # a,b from [-128,127] → [0,255]
    return lab


def lab2rgb(img_lab):
    lab = np.empty_like(img_lab)
    lab[..., 0] = img_lab[..., 0] * 100 / 255       # L from [0,255] → [0,100]
    lab[..., 1:] = img_lab[..., 1:] - 128           # a,b from [0,255] → [-128,127]
    return xyz_to_rgb(lab_to_xyz(lab.astype(np.float32)))
