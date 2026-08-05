/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Implementation of the Sign-of-Life pipeline. See Pipeline.hpp.
 */
#include "Pipeline.hpp"

#include "NativeOps.hpp"

#include <algorithm>

namespace sol {

    Pipeline::Pipeline(const PipelineConfig& config)
        : m_config{config}, m_model{config}, m_shadow{config}
    {
    }

    void Pipeline::Reset()
    {
        m_model.Reset();
        m_shadow.Reset();
    }

    FrameResult Pipeline::Process(const Image& rgb)
    {
        FrameResult result;

        /* Layer 0 -- denoise. */
        const Image denoised = ops::GaussianBlur(rgb, m_config.gaussianKernel);

        /* Layer 1 -- grayscale feature -> running-Gaussian model. */
        const Image gray = ops::RgbToGray(denoised);
        Image foreground = m_model.Apply(gray);

        /* Layer 2 -- HSV shadow veto (uses the previous frame's reference), then
         * absorb this frame into the reference. Both are skipped when the veto
         * is off, so the EW reference -- a full-frame float RGB buffer nothing
         * else reads -- is never allocated. Matches scr/bgs_model.py and the
         * Alif port's SolPipeline.cc. */
        if (m_config.shadowEnabled)
        {
            m_shadow.Suppress(foreground, denoised);
            m_shadow.UpdateReference(denoised);
        }
        result.rawMask = foreground;

        /* Layer 3 -- morphology (open then close). */
        Image morphed = foreground;
        if (m_config.openKernel > 0)
        {
            morphed = ops::MorphologyEx(morphed, ops::MorphOp::Open, m_config.openKernel);
        }
        if (m_config.closeKernel > 0)
        {
            morphed = ops::MorphologyEx(morphed, ops::MorphOp::Close, m_config.closeKernel);
        }

        /* Layer 3 -- connected components + area gate. The filtered mask keeps
         * only the surviving (area-gated) regions, matching the Python
         * ``filtered_mask``. */
        std::vector<int> labels;
        const std::vector<Blob> regions =
            ops::ConnectedComponents(morphed, m_config.connectivity, &labels);

        std::vector<char> keepLabel(regions.size() + 1, 0);
        for (size_t k = 0; k < regions.size(); ++k)
        {
            if (regions[k].area >= m_config.minArea)
            {
                keepLabel[k + 1] = 1;
                result.blobs.push_back(regions[k]);
            }
        }

        result.filteredMask = Image(morphed.width, morphed.height, 1);
        const size_t n = morphed.PixelCount();
        for (size_t i = 0; i < n; ++i)
        {
            const int lbl = labels[i];
            if (lbl > 0 && keepLabel[lbl])
            {
                result.filteredMask.data[i] = 255;
            }
        }

        /* Largest object first, mirroring the Python blob ordering. */
        std::sort(result.blobs.begin(), result.blobs.end(),
                  [](const Blob& a, const Blob& b) { return a.area > b.area; });

        /* Sign-of-Life: any surviving region large enough. */
        for (const Blob& blob : result.blobs)
        {
            if (blob.area >= m_config.areaThreshold)
            {
                result.signOfLife = true;
                break;
            }
        }
        return result;
    }

} /* namespace sol */
