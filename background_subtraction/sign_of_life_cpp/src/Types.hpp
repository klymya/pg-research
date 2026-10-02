/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Sign-of-Life pipeline (C++14 reference) -- shared data types and the pipeline
 * configuration. A faithful port of the pure-NumPy reference in
 * ``background_subtraction/scr`` (single-model, native ops), targeting the Alif
 * ML embedded evaluation kit C++ standards.
 */
#ifndef SOL_TYPES_HPP
#define SOL_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sol {

    /**
     * @brief Interleaved 8-bit image, row-major HWC. @c channels is 3 for RGB
     *        and 1 for a mask (values 0/255).
     */
    struct Image
    {
        int width  = 0;
        int height = 0;
        int channels = 0;
        std::vector<uint8_t> data;

        Image() = default;
        Image(int w, int h, int c) : width{w}, height{h}, channels{c},
                                     data(static_cast<size_t>(w) * h * c, 0)
        {
        }

        bool Empty() const { return width <= 0 || height <= 0 || data.empty(); }
        size_t PixelCount() const { return static_cast<size_t>(width) * height; }
    };

    /**
     * @brief A connected foreground region surviving the area gate.
     */
    struct Blob
    {
        int x      = 0;   /* bounding box, top-left */
        int y      = 0;
        int width  = 0;
        int height = 0;
        int area   = 0;   /* pixel count */
        float centroidX = 0.0f;
        float centroidY = 0.0f;
    };

    /**
     * @brief Everything the pipeline produces for one frame.
     */
    struct FrameResult
    {
        Image rawMask;              /* foreground after the model + shadow veto */
        Image filteredMask;         /* foreground after morphology + area gate  */
        std::vector<Blob> blobs;    /* area-gated foreground objects            */
        bool signOfLife = false;    /* any blob area >= area_threshold          */
    };

    /**
     * @brief Pipeline parameters. Defaults mirror the Python reference
     *        (`PipelineConfig` with the Phase-2 single fast model).
     */
    struct PipelineConfig
    {
        /* Layer 0 -- denoise */
        int gaussianKernel = 5;             /* odd; matches cv2.GaussianBlur (5,5) */

        /* Layer 1 -- running Gaussian model (on the grayscale feature) */
        float modelAlpha        = 0.1f;     /* learning rate                        */
        float modelThreshold    = 12.25f;   /* = 3.5^2 (bgslibrary WrenGA default)  */
        float modelInitVariance = 36.0f;
        int   modelLearnFrames  = 30;       /* every pixel updates for this many    */

        /* Layer 2 -- HSV shadow veto */
        bool  shadowEnabled       = false;  /* off by default, as on target       */
        float shadowAlpha         = 0.1f;   /* EW background-reference rate          */
        float valueRatioMin       = 0.40f;
        float valueRatioMax       = 0.95f;
        float saturationDeltaMax  = 40.0f;
        float hueDeltaMax         = 12.0f;

        /* Layer 3 -- structural filter + Sign-of-Life */
        int openKernel   = 3;
        int closeKernel  = 7;
        int connectivity = 8;               /* 4 or 8 */
        int minArea      = 150;             /* smallest kept foreground object (px)  */
        int areaThreshold = 500;            /* Sign-of-Life trigger (px)             */
    };

} /* namespace sol */

#endif /* SOL_TYPES_HPP */
