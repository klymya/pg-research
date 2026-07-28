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
        float m_alpha;
        float m_threshold;
        float m_initVariance;
        float m_varianceCeil;
        int   m_learnFrames;
        int   m_frameIndex = 0;
        int   m_width      = 0;
        int   m_height     = 0;
        std::vector<float> m_mean;
        std::vector<float> m_variance;
    };

} /* namespace sol */

#endif /* SOL_BACKGROUND_MODEL_HPP */
