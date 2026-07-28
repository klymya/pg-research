/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Sign-of-Life pipeline: chains denoise -> grayscale -> running-Gaussian
 * model -> HSV shadow veto -> morphology -> connected components -> area-based
 * Sign-of-Life. Port of ``scr/pipeline.py`` (single-model, area-based flag).
 */
#ifndef SOL_PIPELINE_HPP
#define SOL_PIPELINE_HPP

#include "BackgroundModel.hpp"
#include "ShadowSuppressor.hpp"
#include "Types.hpp"

namespace sol {

    /**
     * @brief Stateful per-frame pipeline. Feed frames in temporal order; the
     *        background model and shadow reference persist across calls.
     */
    class Pipeline
    {
    public:
        explicit Pipeline(const PipelineConfig& config = PipelineConfig{});

        /** @brief Run one RGB frame through every stage. */
        FrameResult Process(const Image& rgb);

        /** @brief Return every stage to its initial state. */
        void Reset();

        const PipelineConfig& Config() const { return m_config; }

    private:
        PipelineConfig m_config;
        BackgroundModel m_model;
        ShadowSuppressor m_shadow;
    };

} /* namespace sol */

#endif /* SOL_PIPELINE_HPP */
