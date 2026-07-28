/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * HSV brightness-ratio shadow veto plus the exponentially-weighted RGB
 * background reference it needs. Port of the shadow logic in
 * ``scr/features.py`` and the reference update in ``scr/bgs_model.py``.
 */
#ifndef SOL_SHADOW_SUPPRESSOR_HPP
#define SOL_SHADOW_SUPPRESSOR_HPP

#include "Types.hpp"

namespace sol {

    /**
     * @brief Removes foreground pixels whose HSV signature matches a cast shadow
     *        (brightness dropped by a bounded factor, hue and saturation kept),
     *        and maintains the RGB reference the veto compares against.
     */
    class ShadowSuppressor
    {
    public:
        explicit ShadowSuppressor(const PipelineConfig& config);

        /**
         * @brief Remove shadow pixels from @c mask using the current reference.
         *        A no-op before the reference is initialised (first frame).
         * @param mask     Foreground mask (0/255), modified in place.
         * @param frameRgb The current (denoised) RGB frame.
         */
        void Suppress(Image& mask, const Image& frameRgb) const;

        /** @brief Absorb @c frameRgb into the EW reference (call after Suppress). */
        void UpdateReference(const Image& frameRgb);

        void Reset();

    private:
        PipelineConfig m_config;
        int m_width  = 0;
        int m_height = 0;
        std::vector<float> m_reference; /* EW RGB reference, HWC float */
    };

} /* namespace sol */

#endif /* SOL_SHADOW_SUPPRESSOR_HPP */
