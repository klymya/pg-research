/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Implementation of the running Gaussian-average model. See BackgroundModel.hpp.
 */
#include "BackgroundModel.hpp"

#include <algorithm>

namespace sol {

    namespace {
        constexpr float kVarianceFloor = 4.0f; /* matches bgslibrary / the Python port */
    }

    BackgroundModel::BackgroundModel(const PipelineConfig& config)
        : m_alpha{config.modelAlpha},
          m_threshold{config.modelThreshold},
          m_initVariance{config.modelInitVariance},
          m_varianceCeil{5.0f * config.modelInitVariance},
          m_learnFrames{config.modelLearnFrames}
    {
    }

    void BackgroundModel::Reset()
    {
        m_mean.clear();
        m_variance.clear();
        m_frameIndex = 0;
        m_width      = 0;
        m_height     = 0;
    }

    Image BackgroundModel::Apply(const Image& gray)
    {
        const int w = gray.width;
        const int h = gray.height;
        const size_t n = gray.PixelCount();

        Image mask(w, h, 1);

        /* Cold start: copy the first frame into the mean, return an empty mask. */
        if (m_mean.empty() || m_width != w || m_height != h)
        {
            m_width  = w;
            m_height = h;
            m_mean.assign(n, 0.0f);
            m_variance.assign(n, m_initVariance);
            for (size_t i = 0; i < n; ++i)
            {
                m_mean[i] = static_cast<float>(gray.data[i]);
            }
            m_frameIndex = 1;
            return mask; /* all zeros */
        }

        ++m_frameIndex;
        const bool learning = m_frameIndex < m_learnFrames;

        for (size_t i = 0; i < n; ++i)
        {
            const float current = static_cast<float>(gray.data[i]);
            const float delta   = current - m_mean[i];
            const float sqDist  = delta * delta;
            const bool foreground = sqDist > m_threshold * m_variance[i];
            mask.data[i] = foreground ? 255 : 0;

            const bool updatable = (!foreground) || learning;
            if (updatable)
            {
                m_mean[i] += m_alpha * delta;
                float newVar = m_variance[i] + m_alpha * (sqDist - m_variance[i]);
                newVar = std::min(std::max(newVar, kVarianceFloor), m_varianceCeil);
                m_variance[i] = newVar;
            }
        }
        return mask;
    }

} /* namespace sol */
