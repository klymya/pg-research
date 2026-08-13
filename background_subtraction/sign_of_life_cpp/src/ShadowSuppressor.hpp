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
     *
     * The reference is @c uint8 -- 3 bytes per pixel, 230,400 B at QVGA against
     * the 921,600 B a float reference costs -- absorbed with the same Q8
     * integer update as the background model, and the veto's comparisons are
     * integer too. The float reference is not implemented here: it is 12 B/px
     * of persistent state on a target that has none to spare, and the Python
     * reference keeps it selectable (``ShadowConfig.reference_dtype``) for
     * exactly this comparison.
     *
     * @note The quantised reference carries the same @c 1/(2*shadowAlpha) dead
     *       zone as the model mean. Unlike the model there is no closed-form
     *       bound on the error the veto tolerates -- it depends on the value
     *       ratio band and hue stability, both of which vary with scene
     *       brightness -- so this mode rests on measurement, not derivation.
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
        int32_t m_alphaQ8;      /* shadowAlpha * 256, in [0, 256]            */
        int32_t m_ratioMinQ16;  /* valueRatioMin * 65536 (see ShadowSuppressor.cpp) */
        int32_t m_ratioMaxQ16;  /* valueRatioMax * 65536                     */
        int32_t m_satDeltaMax;  /* rounded to whole saturation levels        */
        int32_t m_hueDeltaMax;  /* rounded to whole hue levels               */
        int m_width  = 0;
        int m_height = 0;
        std::vector<uint8_t> m_reference; /* EW RGB reference, HWC uint8 */
    };

} /* namespace sol */

#endif /* SOL_SHADOW_SUPPRESSOR_HPP */
