/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Implementation of the running Gaussian-average model. See BackgroundModel.hpp.
 */
#include "BackgroundModel.hpp"

#include <cmath>

namespace sol {

    namespace {
        constexpr int32_t kQBits = 8;             /* fractional bits of the Q8 form */
        constexpr int32_t kQOne  = 1 << kQBits;   /* fixed-point 1.0                */
        constexpr int32_t kQHalf = kQOne >> 1;    /* rounding offset                */
        constexpr int32_t kVarianceFloor = 4;     /* matches bgslibrary / Python    */
        constexpr int32_t kUint8Max = 255;
        constexpr int32_t kMaxThresholdQ = 1 << 23; /* keeps thr*variance in int32  */

        /**
         * @brief Divide by 2^kQBits, rounding half away from zero.
         *
         * A right shift of a negative value rounds towards negative infinity
         * (and was implementation-defined before C++20), which biases the mean
         * downwards; truncating towards zero widens the dead zone instead. Both
         * cost accuracy, so the magnitude is rounded and the sign reapplied.
         *
         * @param value Q8 product; |value| stays under 50 M here, so negating it
         *              cannot overflow.
         */
        int32_t RoundShift(int32_t value)
        {
            return value >= 0 ? ((value + kQHalf) >> kQBits)
                              : -((-value + kQHalf) >> kQBits);
        }

        int32_t Clamp(int32_t value, int32_t low, int32_t high)
        {
            return value < low ? low : (value > high ? high : value);
        }

        int32_t Quantise(float value)
        {
            return static_cast<int32_t>(std::lround(static_cast<double>(value) * kQOne));
        }
    }

    BackgroundModel::BackgroundModel(const PipelineConfig& config)
        : m_alphaQ8{Clamp(Quantise(config.modelAlpha), 0, kQOne)},
          m_thresholdQ8{Clamp(Quantise(config.modelThreshold), 1, kMaxThresholdQ)},
          m_initVariance{kVarianceFloor},
          m_varianceCeil{kVarianceFloor},
          m_learnFrames{config.modelLearnFrames}
    {
        /* 5 * initVariance is 180 at the default but overflows a uint8 for any
         * initVariance > 51, so the ceiling is capped at 255 and the initial
         * value clamped into the resulting range. */
        const int32_t ceiling =
            static_cast<int32_t>(std::lround(5.0 * static_cast<double>(config.modelInitVariance)));
        m_varianceCeil = Clamp(ceiling, kVarianceFloor, kUint8Max);
        m_initVariance = Clamp(static_cast<int32_t>(std::lround(
                                   static_cast<double>(config.modelInitVariance))),
                               kVarianceFloor, m_varianceCeil);
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

        /* Cold start: copy the first frame into the mean, return an empty mask.
         * uint8 -> uint8 is lossless, unlike the float state it replaces. */
        if (m_mean.empty() || m_width != w || m_height != h)
        {
            m_width  = w;
            m_height = h;
            m_mean.assign(gray.data.begin(), gray.data.begin() + static_cast<long>(n));
            m_variance.assign(n, static_cast<uint8_t>(m_initVariance));
            m_frameIndex = 1;
            return mask; /* all zeros */
        }

        ++m_frameIndex;
        const bool learning = m_frameIndex < m_learnFrames;

        for (size_t i = 0; i < n; ++i)
        {
            /* int32 throughout: uint8 arithmetic wraps on the signed difference,
             * and the squared distance alone reaches 65,025. The widest product
             * below is alphaQ8 * (sqDist - variance) at 16.6 M -- a 128x margin
             * under INT32_MAX. */
            const int32_t current  = static_cast<int32_t>(gray.data[i]);
            const int32_t mean     = static_cast<int32_t>(m_mean[i]);
            const int32_t variance = static_cast<int32_t>(m_variance[i]);

            const int32_t delta  = current - mean;  /* [-255, 255] */
            const int32_t sqDist = delta * delta;   /* <= 65,025   */

            /* Both sides scaled by 2^kQBits to compare against the Q8 threshold:
             * sqDist << 8 <= 16.6 M, thresholdQ8 * variance <= 2.139 G. */
            const bool foreground = (sqDist << kQBits) > m_thresholdQ8 * variance;
            mask.data[i] = foreground ? 255 : 0;

            if (foreground && !learning)
            {
                continue;
            }

            /* mean + alpha * delta cannot leave [0, 255] for alpha <= 1; the
             * clamp also covers an alphaQ8 rounded up to exactly kQOne. */
            m_mean[i] = static_cast<uint8_t>(
                Clamp(mean + RoundShift(m_alphaQ8 * delta), 0, kUint8Max));

            /* Clamped before the uint8 store: the raw update reaches 16.6 M. */
            m_variance[i] = static_cast<uint8_t>(
                Clamp(variance + RoundShift(m_alphaQ8 * (sqDist - variance)),
                      kVarianceFloor, m_varianceCeil));
        }
        return mask;
    }

} /* namespace sol */
