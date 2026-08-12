/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Running Gaussian-average background model (Wren et al., Pfinder), on the
 * single-channel grayscale feature. Port of
 * ``scr/models/running_gaussian.py``.
 */
#ifndef SOL_BACKGROUND_MODEL_HPP
#define SOL_BACKGROUND_MODEL_HPP

#include "Types.hpp"

namespace sol {

    /**
     * @brief Single Gaussian per pixel (running mean + shared variance). A pixel
     *        is foreground when its squared distance to the mean exceeds
     *        @c threshold * variance.
     *
     * The state is @c uint8 -- 2 bytes per pixel, 150 KB at QVGA against the
     * 600 KB a float mean and variance would cost. @c PipelineConfig keeps
     * @c alpha and @c threshold as floats; they are quantised to Q8 fixed point
     * once in the constructor so the per-pixel update is integer-only and
     * bit-identical to the Python reference.
     *
     * @note Quantising the mean introduces an update dead zone of
     *       @c 1/(2*alpha) grey levels (5 at @c alpha=0.1). It is harmless only
     *       while that stays below the smallest detectable distance,
     *       @c sqrt(threshold * kVarianceFloor) = 7 at the defaults. Lowering
     *       @c modelAlpha below ~0.071 freezes the mean; see the dead-zone note
     *       in the Python module docstring.
     */
    class BackgroundModel
    {
    public:
        explicit BackgroundModel(const PipelineConfig& config);

        /** @brief Classify @c gray (1-channel) and update the model in place. */
        Image Apply(const Image& gray);

        /** @brief Discard the learned Gaussian parameters. */
        void Reset();

    private:
        int32_t m_alphaQ8;       /* alpha     * 256, in [0, 256]              */
        int32_t m_thresholdQ8;   /* threshold * 256                           */
        int32_t m_initVariance;  /* quantised initial variance                */
        int32_t m_varianceCeil;  /* min(round(5 * initVariance), 255)         */
        int     m_learnFrames;
        int     m_frameIndex = 0;
        int     m_width      = 0;
        int     m_height     = 0;
        std::vector<uint8_t> m_mean;
        std::vector<uint8_t> m_variance;
    };

} /* namespace sol */

#endif /* SOL_BACKGROUND_MODEL_HPP */
