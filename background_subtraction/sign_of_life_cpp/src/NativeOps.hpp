/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Native (pure-C++, OpenCV-free) image operators, ported from the OpenCV
 * algorithms and the NumPy reference in ``scr/native_ops.py``:
 *   - RgbToGray  : BT.601 fixed-point luma (matches cv2 RGB2GRAY)
 *   - RgbToHsv   : 8-bit fixed-point HSV (matches cv2 RGB2HSV)
 *   - GaussianBlur : separable Gaussian, reflect-101 border (~ cv2.GaussianBlur)
 *   - Morphology : elliptical structuring element, erode/dilate/open/close
 *   - ConnectedComponents : two-pass union-find labelling + per-blob stats
 */
#ifndef SOL_NATIVE_OPS_HPP
#define SOL_NATIVE_OPS_HPP

#include "Types.hpp"

namespace sol {
namespace ops {

    /** @brief RGB (HWC, 3ch) -> single-channel grayscale (BT.601). */
    Image RgbToGray(const Image& rgb);

    /** @brief RGB (HWC, 3ch) -> HSV (HWC, 3ch), H in [0,180). */
    Image RgbToHsv(const Image& rgb);

    /**
     * @brief Convert one RGB triple to 8-bit HSV.
     *
     * Same implementation RgbToHsv() is built on. Lets the shadow veto convert
     * only the pixels it actually tests, instead of materialising two
     * full-frame HSV images per frame.
     */
    void RgbToHsvPixel(int r, int g, int b, uint8_t& h, uint8_t& s, uint8_t& v);

    /** @brief Separable Gaussian blur of an odd @c kernelSize; sigma from cv2. */
    Image GaussianBlur(const Image& src, int kernelSize);

    /** @brief Compound morphology on a 0/255 mask with an elliptical kernel. */
    enum class MorphOp
    {
        Open,
        Close
    };
    Image MorphologyEx(const Image& mask, MorphOp op, int kernelSize);

    /**
     * @brief Label the connected foreground regions of a 0/255 mask.
     * @param mask         Single-channel 0/255 image.
     * @param connectivity 4 or 8.
     * @param labels       Optional out: per-pixel label image (0 = background,
     *                     component @c k in the result carries label @c k+1).
     * @return All components (any non-zero area), with bbox/area/centroid.
     */
    std::vector<Blob> ConnectedComponents(const Image& mask, int connectivity,
                                          std::vector<int>* labels = nullptr);

} /* namespace ops */
} /* namespace sol */

#endif /* SOL_NATIVE_OPS_HPP */
