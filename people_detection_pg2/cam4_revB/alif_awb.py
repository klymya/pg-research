import numpy as np


# === Shared Constants ===
CORRECTION_MATRIX = np.array([
    [1.86,  0.00, -0.05],  # R
    [-0.16, 0.95, -0.53],  # G
    [-0.53, 0.07,  2.27]   # B
], dtype=np.float32)

# === Utilities ===

def srgb_oetf(lum: np.ndarray) -> np.ndarray:
    """sRGB Opto-Electronic Transfer Function (OETF)"""
    out = np.empty_like(lum)
    mask = lum <= 0.0031308
    out[mask] = 12.92 * lum[mask]
    out[~mask] = 1.055 * np.power(lum[~mask], 1 / 2.4) - 0.055
    return out

def prepare_gamma_lut() -> np.ndarray:
    """Create a 256-entry gamma LUT using sRGB OETF"""
    lut = np.empty(256, dtype=np.uint8)
    for i in range(256):
        lum = i / 255.0
        gamma = srgb_oetf(np.array([lum], dtype=np.float32))[0]
        lut[i] = np.clip(np.round(gamma * 255), 0, 255).astype(np.uint8)
    return lut


# === Vectorized White Balance ===
def white_balance_vectorized(image: np.ndarray) -> np.ndarray:
    """
    Vectorized white balance method.
    This method combines color correction and gray-world white balancing.
    """
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3

    img_float = image.astype(np.float32)
    corrected = np.tensordot(img_float, CORRECTION_MATRIX.T, axes=([2], [1]))
    corrected = np.clip(corrected, 0, 255)

    lut = prepare_gamma_lut()
    corrected_gamma = lut[corrected.astype(np.uint8)]

    mean_rgb = corrected_gamma.mean(axis=(0, 1), dtype=np.float32)
    mean_gray = mean_rgb.mean()
    scale = mean_gray / mean_rgb
    balanced = corrected_gamma.astype(np.float32) * scale

    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced
