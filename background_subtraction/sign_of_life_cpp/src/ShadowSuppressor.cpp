/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Implementation of the HSV shadow veto. See ShadowSuppressor.hpp.
 */
#include "ShadowSuppressor.hpp"

#include "NativeOps.hpp"

#include <algorithm>
#include <cmath>

namespace sol {

    namespace {
        constexpr float kEpsilon = 1e-6f;
    }

    ShadowSuppressor::ShadowSuppressor(const PipelineConfig& config) : m_config{config}
    {
    }

    void ShadowSuppressor::Reset()
    {
        m_reference.clear();
        m_width  = 0;
        m_height = 0;
    }

    void ShadowSuppressor::Suppress(Image& mask, const Image& frameRgb) const
    {
        if (!m_config.shadowEnabled || m_reference.empty())
        {
            return;
        }

        /* The veto works on the uint8-clipped reference, as the Python reference
         * does (``background_bgr`` is the clipped EW estimate). */
        const size_t n = frameRgb.PixelCount();
        Image referenceRgb(frameRgb.width, frameRgb.height, 3);
        for (size_t i = 0; i < n * 3; ++i)
        {
            const float v = m_reference[i];
            referenceRgb.data[i] =
                static_cast<uint8_t>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
        }

        const Image frameHsv = ops::RgbToHsv(frameRgb);
        const Image bgHsv    = ops::RgbToHsv(referenceRgb);

        for (size_t i = 0; i < n; ++i)
        {
            if (mask.data[i] == 0)
            {
                continue;
            }
            const float hf = frameHsv.data[i * 3 + 0];
            const float sf = frameHsv.data[i * 3 + 1];
            const float vf = frameHsv.data[i * 3 + 2];
            const float hb = bgHsv.data[i * 3 + 0];
            const float sb = bgHsv.data[i * 3 + 1];
            const float vb = bgHsv.data[i * 3 + 2];

            const float valueRatio = vf / (vb + kEpsilon);
            const float satDelta   = std::fabs(sf - sb);
            float hueDelta         = std::fabs(hf - hb);
            hueDelta               = std::min(hueDelta, 180.0f - hueDelta);

            const bool isShadow = valueRatio >= m_config.valueRatioMin &&
                                  valueRatio <= m_config.valueRatioMax &&
                                  satDelta <= m_config.saturationDeltaMax &&
                                  hueDelta <= m_config.hueDeltaMax;
            if (isShadow)
            {
                mask.data[i] = 0;
            }
        }
    }

    void ShadowSuppressor::UpdateReference(const Image& frameRgb)
    {
        /* Nothing reads the reference when the veto is off, so it must never be
         * allocated -- it is a full-frame float RGB buffer (921,600 B at QVGA).
         * The guard lives here rather than only at the call site so the class is
         * correct for any caller. */
        if (!m_config.shadowEnabled)
        {
            return;
        }

        const int w = frameRgb.width;
        const int h = frameRgb.height;
        const size_t count = frameRgb.PixelCount() * 3;

        if (m_reference.empty() || m_width != w || m_height != h)
        {
            m_width  = w;
            m_height = h;
            m_reference.assign(count, 0.0f);
            for (size_t i = 0; i < count; ++i)
            {
                m_reference[i] = static_cast<float>(frameRgb.data[i]);
            }
            return;
        }

        const float a = m_config.shadowAlpha;
        for (size_t i = 0; i < count; ++i)
        {
            m_reference[i] = (1.0f - a) * m_reference[i] + a * frameRgb.data[i];
        }
    }

} /* namespace sol */
