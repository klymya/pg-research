/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Implementation of the native image operators. See NativeOps.hpp.
 */
#include "NativeOps.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sol {
namespace ops {

    namespace {

        /* --- HSV fixed-point tables (built once, mirror scr/native_ops.py) --- */

        constexpr int kHsvShift  = 12;
        constexpr int kHueRange  = 180;

        struct HsvTables
        {
            int sdiv[256];
            int hdiv[256];

            HsvTables()
            {
                sdiv[0] = hdiv[0] = 0;
                for (int i = 1; i < 256; ++i)
                {
                    sdiv[i] = static_cast<int>(
                        std::rint((255 << kHsvShift) / static_cast<double>(i)));
                    hdiv[i] = static_cast<int>(
                        std::rint((kHueRange << kHsvShift) / (6.0 * i)));
                }
            }
        };

        const HsvTables& Tables()
        {
            static const HsvTables tables;
            return tables;
        }

        /* Reflect-101 index (cv2 BORDER_REFLECT_101: no edge repeat). */
        int Reflect101(int p, int n)
        {
            if (n == 1)
            {
                return 0;
            }
            while (p < 0 || p >= n)
            {
                if (p < 0)
                {
                    p = -p;
                }
                else
                {
                    p = 2 * (n - 1) - p;
                }
            }
            return p;
        }

        uint8_t SaturateU8(float v)
        {
            const int r = static_cast<int>(std::rint(v));
            return static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
        }

    } /* namespace */

    Image RgbToGray(const Image& rgb)
    {
        /* BT.601 fixed-point, identical to cv2 RGB2GRAY: R2Y=4899, G2Y=9617,
         * B2Y=1868, yuv_shift=14. */
        Image gray(rgb.width, rgb.height, 1);
        const size_t n = rgb.PixelCount();
        for (size_t i = 0; i < n; ++i)
        {
            const int r = rgb.data[i * 3 + 0];
            const int g = rgb.data[i * 3 + 1];
            const int b = rgb.data[i * 3 + 2];
            gray.data[i] =
                static_cast<uint8_t>((4899 * r + 9617 * g + 1868 * b + 8192) >> 14);
        }
        return gray;
    }

    Image RgbToHsv(const Image& rgb)
    {
        const HsvTables& t = Tables();
        Image hsv(rgb.width, rgb.height, 3);
        const size_t n = rgb.PixelCount();
        const int rounding = 1 << (kHsvShift - 1);

        for (size_t i = 0; i < n; ++i)
        {
            const int r = rgb.data[i * 3 + 0];
            const int g = rgb.data[i * 3 + 1];
            const int b = rgb.data[i * 3 + 2];

            const int v    = std::max(std::max(r, g), b);
            const int vmin = std::min(std::min(r, g), b);
            const int diff = v - vmin;

            const int vr = (v == r) ? -1 : 0;
            const int vg = (v == g) ? -1 : 0;

            const int s = (diff * t.sdiv[v] + rounding) >> kHsvShift;

            int h = (vr & (g - b)) +
                    (~vr & ((vg & (b - r + 2 * diff)) + (~vg & (r - g + 4 * diff))));
            h = (h * t.hdiv[diff] + rounding) >> kHsvShift;
            h += (h < 0) ? kHueRange : 0;

            hsv.data[i * 3 + 0] = static_cast<uint8_t>(h);
            hsv.data[i * 3 + 1] = static_cast<uint8_t>(s);
            hsv.data[i * 3 + 2] = static_cast<uint8_t>(v);
        }
        return hsv;
    }

    Image GaussianBlur(const Image& src, int kernelSize)
    {
        if (kernelSize <= 1)
        {
            return src;
        }
        /* sigma derived from ksize exactly as cv2 does when sigma <= 0. */
        const double sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
        const int radius   = kernelSize / 2;

        std::vector<float> kernel(kernelSize);
        double sum = 0.0;
        for (int i = 0; i < kernelSize; ++i)
        {
            const double x = i - radius;
            kernel[i]      = static_cast<float>(std::exp(-(x * x) / (2 * sigma * sigma)));
            sum += kernel[i];
        }
        for (int i = 0; i < kernelSize; ++i)
        {
            kernel[i] = static_cast<float>(kernel[i] / sum);
        }

        const int w = src.width;
        const int h = src.height;
        const int c = src.channels;

        /* Horizontal pass into a float buffer, then vertical pass to uint8. */
        std::vector<float> tmp(static_cast<size_t>(w) * h * c, 0.0f);
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                for (int ch = 0; ch < c; ++ch)
                {
                    float acc = 0.0f;
                    for (int k = -radius; k <= radius; ++k)
                    {
                        const int xx = Reflect101(x + k, w);
                        acc += kernel[k + radius] *
                               src.data[(static_cast<size_t>(y) * w + xx) * c + ch];
                    }
                    tmp[(static_cast<size_t>(y) * w + x) * c + ch] = acc;
                }
            }
        }

        Image dst(w, h, c);
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                for (int ch = 0; ch < c; ++ch)
                {
                    float acc = 0.0f;
                    for (int k = -radius; k <= radius; ++k)
                    {
                        const int yy = Reflect101(y + k, h);
                        acc += kernel[k + radius] *
                               tmp[(static_cast<size_t>(yy) * w + x) * c + ch];
                    }
                    dst.data[(static_cast<size_t>(y) * w + x) * c + ch] = SaturateU8(acc);
                }
            }
        }
        return dst;
    }

    namespace {

        /* Elliptical structuring element as a list of (dy,dx) offsets, matching
         * cv2.getStructuringElement(MORPH_ELLIPSE). */
        std::vector<std::pair<int, int>> EllipseOffsets(int kernelSize)
        {
            std::vector<std::pair<int, int>> offsets;
            const int radius = kernelSize / 2;
            const int anchor = kernelSize / 2;
            const double invR2 = radius ? 1.0 / (static_cast<double>(radius) * radius) : 0.0;
            for (int i = 0; i < kernelSize; ++i)
            {
                const int dy = i - radius;
                if (std::abs(dy) > radius)
                {
                    continue;
                }
                const int dx = radius ? static_cast<int>(std::lround(
                                            anchor * std::sqrt((radius * radius - dy * dy) * invR2)))
                                      : 0;
                const int j1 = std::max(anchor - dx, 0);
                const int j2 = std::min(anchor + dx + 1, kernelSize);
                for (int j = j1; j < j2; ++j)
                {
                    offsets.emplace_back(i - anchor, j - anchor);
                }
            }
            return offsets;
        }

        Image RankFilter(const Image& mask,
                         const std::vector<std::pair<int, int>>& offsets, bool dilate)
        {
            const int w = mask.width;
            const int h = mask.height;
            const uint8_t border = dilate ? 0 : 255;
            Image out(w, h, 1);
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    int value = dilate ? 0 : 255;
                    for (const auto& off : offsets)
                    {
                        const int yy = y + off.first;
                        const int xx = x + off.second;
                        int sample;
                        if (yy < 0 || yy >= h || xx < 0 || xx >= w)
                        {
                            sample = border;
                        }
                        else
                        {
                            sample = mask.data[static_cast<size_t>(yy) * w + xx];
                        }
                        value = dilate ? std::max(value, sample) : std::min(value, sample);
                    }
                    out.data[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(value);
                }
            }
            return out;
        }

    } /* namespace */

    Image MorphologyEx(const Image& mask, MorphOp op, int kernelSize)
    {
        const auto offsets = EllipseOffsets(kernelSize);
        if (op == MorphOp::Open)
        {
            return RankFilter(RankFilter(mask, offsets, false), offsets, true);
        }
        return RankFilter(RankFilter(mask, offsets, true), offsets, false);
    }

    std::vector<Blob> ConnectedComponents(const Image& mask, int connectivity,
                                          std::vector<int>* outLabels)
    {
        const int w = mask.width;
        const int h = mask.height;
        std::vector<int> labels(static_cast<size_t>(w) * h, 0);

        static const int n8dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        static const int n8dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        static const int n4dy[4] = {-1, 1, 0, 0};
        static const int n4dx[4] = {0, 0, -1, 1};
        const int* dy = (connectivity == 4) ? n4dy : n8dy;
        const int* dx = (connectivity == 4) ? n4dx : n8dx;
        const int nn  = (connectivity == 4) ? 4 : 8;

        std::vector<Blob> blobs;
        std::vector<int> stack;
        int label = 0;

        for (int y0 = 0; y0 < h; ++y0)
        {
            for (int x0 = 0; x0 < w; ++x0)
            {
                const size_t seed = static_cast<size_t>(y0) * w + x0;
                if (mask.data[seed] == 0 || labels[seed] != 0)
                {
                    continue;
                }
                ++label;
                Blob blob;
                blob.x = x0;
                blob.y = y0;
                int maxX = x0;
                int maxY = y0;
                long sumX = 0;
                long sumY = 0;
                int area  = 0;

                labels[seed] = label;
                stack.clear();
                stack.push_back(static_cast<int>(seed));
                while (!stack.empty())
                {
                    const int p = stack.back();
                    stack.pop_back();
                    const int py = p / w;
                    const int px = p % w;
                    ++area;
                    sumX += px;
                    sumY += py;
                    blob.x = std::min(blob.x, px);
                    blob.y = std::min(blob.y, py);
                    maxX   = std::max(maxX, px);
                    maxY   = std::max(maxY, py);
                    for (int k = 0; k < nn; ++k)
                    {
                        const int ny = py + dy[k];
                        const int nx = px + dx[k];
                        if (ny < 0 || ny >= h || nx < 0 || nx >= w)
                        {
                            continue;
                        }
                        const size_t np = static_cast<size_t>(ny) * w + nx;
                        if (mask.data[np] != 0 && labels[np] == 0)
                        {
                            labels[np] = label;
                            stack.push_back(static_cast<int>(np));
                        }
                    }
                }

                blob.width     = maxX - blob.x + 1;
                blob.height    = maxY - blob.y + 1;
                blob.area      = area;
                blob.centroidX = static_cast<float>(static_cast<double>(sumX) / area);
                blob.centroidY = static_cast<float>(static_cast<double>(sumY) / area);
                blobs.push_back(blob);
            }
        }
        if (outLabels != nullptr)
        {
            *outLabels = std::move(labels);
        }
        return blobs;
    }

} /* namespace ops */
} /* namespace sol */
