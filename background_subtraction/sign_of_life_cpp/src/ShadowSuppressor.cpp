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
        constexpr int32_t kQBits = 8;              /* reference absorption      */
        constexpr int32_t kQOne  = 1 << kQBits;
        constexpr int32_t kQHalf = kQOne >> 1;
        constexpr int32_t kUint8Max = 255;

        /* The value-ratio bounds carry more fractional bits than the Q8 used
         * for the absorption rate. In Q8, 0.40 rounds to 102/256 = 0.3984 -- a
         * 0.4 % shift of the threshold, which disagrees with the float
         * reference on a measurable number of pixels. Q16 puts that error at
         * 6e-6 for free: 255 << 16 = 16.7 M keeps a 128x margin under
         * INT32_MAX. */
        constexpr int32_t kRatioBits = 16;

        /** @brief Divide by 2^kQBits, rounding half away from zero. */
        int32_t RoundShift(int32_t value)
        {
            return value >= 0 ? ((value + kQHalf) >> kQBits)
                              : -((-value + kQHalf) >> kQBits);
        }

        int32_t Clamp(int32_t value, int32_t low, int32_t high)
        {
            return value < low ? low : (value > high ? high : value);
        }
    }

    ShadowSuppressor::ShadowSuppressor(const PipelineConfig& config)
        : m_config{config},
          m_alphaQ8{Clamp(static_cast<int32_t>(
                              std::lround(static_cast<double>(config.shadowAlpha) * kQOne)),
                          0, kQOne)},
          m_ratioMinQ16{static_cast<int32_t>(std::lround(
              static_cast<double>(config.valueRatioMin) * (1 << kRatioBits)))},
          m_ratioMaxQ16{static_cast<int32_t>(std::lround(
              static_cast<double>(config.valueRatioMax) * (1 << kRatioBits)))},
          m_satDeltaMax{static_cast<int32_t>(
              std::lround(static_cast<double>(config.saturationDeltaMax)))},
          m_hueDeltaMax{static_cast<int32_t>(
              std::lround(static_cast<double>(config.hueDeltaMax)))}
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

        /* Both HSV conversions happen per pixel, inside this loop, rather than
         * into full-frame images -- same arithmetic, far less scratch, and only
         * foreground pixels are converted at all. The reference needs no
         * clipping now that it is stored uint8. */
        const size_t n = frameRgb.PixelCount();

        for (size_t i = 0; i < n; ++i)
        {
            if (mask.data[i] == 0)
            {
                continue;
            }

            uint8_t hf8, sf8, vf8;
            ops::RgbToHsvPixel(frameRgb.data[i * 3 + 0], frameRgb.data[i * 3 + 1],
                               frameRgb.data[i * 3 + 2], hf8, sf8, vf8);

            uint8_t hb8, sb8, vb8;
            ops::RgbToHsvPixel(m_reference[i * 3 + 0], m_reference[i * 3 + 1],
                               m_reference[i * 3 + 2], hb8, sb8, vb8);

            const int32_t valueFrame      = static_cast<int32_t>(vf8);
            const int32_t valueBackground = static_cast<int32_t>(vb8);

            /* The float reference divides by valueBackground + 1e-6, so a black
             * background pixel yields a ratio that is either enormous or zero
             * and falls outside the band either way -- never shadow. Cross
             * multiplied, 0 >= 0 and 0 <= 0 would both hold at
             * valueFrame == valueBackground == 0 and invert that, hence the
             * explicit guard. */
            if (valueBackground == 0)
            {
                continue;
            }

            const int32_t scaled = valueFrame << kRatioBits;
            if (scaled < m_ratioMinQ16 * valueBackground ||
                scaled > m_ratioMaxQ16 * valueBackground)
            {
                continue;
            }

            const int32_t satDelta = std::abs(static_cast<int32_t>(sf8) -
                                              static_cast<int32_t>(sb8));
            if (satDelta > m_satDeltaMax)
            {
                continue;
            }

            int32_t hueDelta = std::abs(static_cast<int32_t>(hf8) -
                                        static_cast<int32_t>(hb8));
            hueDelta = std::min(hueDelta, 180 - hueDelta);
            if (hueDelta > m_hueDeltaMax)
            {
                continue;
            }

            mask.data[i] = 0;
        }
    }

    void ShadowSuppressor::UpdateReference(const Image& frameRgb)
    {
        /* Nothing reads the reference when the veto is off, so it must never be
         * allocated -- at QVGA it is 230,400 B that the disabled build has not
         * reserved. The guard lives here rather than only at the call site so
         * the class is correct for any caller. */
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
            /* uint8 -> uint8 seeding is lossless, unlike the float state it
             * replaces, which then had to be clipped back on every read. */
            m_reference.assign(frameRgb.data.begin(),
                               frameRgb.data.begin() + static_cast<long>(count));
            return;
        }

        for (size_t i = 0; i < count; ++i)
        {
            /* int32: the signed difference wraps in uint8. Widest product is
             * 256 * 255 = 65,280. */
            const int32_t reference = static_cast<int32_t>(m_reference[i]);
            const int32_t current   = static_cast<int32_t>(frameRgb.data[i]);
            m_reference[i] = static_cast<uint8_t>(Clamp(
                reference + RoundShift(m_alphaQ8 * (current - reference)),
                0, kUint8Max));
        }
    }

} /* namespace sol */
