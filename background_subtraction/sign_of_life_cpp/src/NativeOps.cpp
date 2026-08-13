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

    void RgbToHsvPixel(int r, int g, int b, uint8_t& h8, uint8_t& s8, uint8_t& v8)
    {
        const HsvTables& t = Tables();
        const int rounding = 1 << (kHsvShift - 1);

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

        h8 = static_cast<uint8_t>(h);
        s8 = static_cast<uint8_t>(s);
        v8 = static_cast<uint8_t>(v);
    }

    Image RgbToHsv(const Image& rgb)
    {
        Image hsv(rgb.width, rgb.height, 3);
        const size_t n = rgb.PixelCount();

        for (size_t i = 0; i < n; ++i)
        {
            RgbToHsvPixel(rgb.data[i * 3 + 0], rgb.data[i * 3 + 1], rgb.data[i * 3 + 2],
                          hsv.data[i * 3 + 0], hsv.data[i * 3 + 1], hsv.data[i * 3 + 2]);
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

    namespace {

        /** @brief Union-find root of a provisional label, with path compression. */
        int FindRoot(std::vector<int>& parent, int label)
        {
            int root = label;
            while (parent[root] != root)
            {
                root = parent[root];
            }
            while (parent[label] != root) /* compress, iteratively */
            {
                const int next = parent[label];
                parent[label]  = root;
                label          = next;
            }
            return root;
        }

        /**
         * @brief Merge two provisional labels' components.
         *
         * The smaller root wins -- the classic Rosenfeld choice, which biases a
         * component towards the label it was first given in the raster scan.
         */
        void UnionLabels(std::vector<int>& parent, int a, int b)
        {
            int rootA = FindRoot(parent, a);
            int rootB = FindRoot(parent, b);
            if (rootA == rootB)
            {
                return;
            }
            if (rootB < rootA)
            {
                std::swap(rootA, rootB);
            }
            parent[rootB] = rootA;
        }

    } /* namespace */

    std::vector<Blob> ConnectedComponents(const Image& mask, int connectivity,
                                          std::vector<int>* outLabels)
    {
        const int w = mask.width;
        const int h = mask.height;
        std::vector<int> labels(static_cast<size_t>(w) * h, 0);

        /* Two-pass union-find (Rosenfeld & Pfaltz; He et al.). Pass 1 scans
         * once, giving each foreground pixel the smallest label among the
         * neighbours the scan has already visited -- or a fresh provisional
         * label -- and recording the equivalences it finds. Pass 2 resolves
         * each provisional label to its root, renumbers the roots in raster
         * order of first appearance and accumulates the per-blob statistics.
         *
         * Only backwards neighbours are consulted, so no pixel stack is needed:
         * north-west, north, north-east and west for 8-connectivity, north and
         * west for 4-connectivity. Mirrors scr/native_ops.py exactly. */
        static const int nbDy8[4] = {-1, -1, -1, 0};
        static const int nbDx8[4] = {-1, 0, 1, -1};
        static const int nbDy4[2] = {-1, 0};
        static const int nbDx4[2] = {0, -1};
        const int* nbDy = (connectivity == 4) ? nbDy4 : nbDy8;
        const int* nbDx = (connectivity == 4) ? nbDx4 : nbDx8;
        const int nn    = (connectivity == 4) ? 2 : 4;

        /* --- Pass 1: provisional labels + equivalences --------------------- */
        std::vector<int> parent(1, 0); /* index 0 is the background */
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const size_t here = static_cast<size_t>(y) * w + x;
                if (mask.data[here] == 0)
                {
                    continue;
                }

                int smallest = 0;
                for (int k = 0; k < nn; ++k)
                {
                    const int ny = y + nbDy[k];
                    const int nx = x + nbDx[k];
                    if (ny < 0 || ny >= h || nx < 0 || nx >= w)
                    {
                        continue;
                    }
                    const int neighbour = labels[static_cast<size_t>(ny) * w + nx];
                    if (neighbour != 0 && (smallest == 0 || neighbour < smallest))
                    {
                        smallest = neighbour;
                    }
                }

                if (smallest == 0)
                {
                    parent.push_back(static_cast<int>(parent.size()));
                    labels[here] = static_cast<int>(parent.size()) - 1;
                    continue;
                }

                labels[here] = smallest;
                for (int k = 0; k < nn; ++k) /* every neighbour joins this one */
                {
                    const int ny = y + nbDy[k];
                    const int nx = x + nbDx[k];
                    if (ny < 0 || ny >= h || nx < 0 || nx >= w)
                    {
                        continue;
                    }
                    const int neighbour = labels[static_cast<size_t>(ny) * w + nx];
                    if (neighbour != 0)
                    {
                        UnionLabels(parent, smallest, neighbour);
                    }
                }
            }
        }

        /* --- Pass 2: flatten the table, renumber, resolve, measure -------- */
        /* UnionLabels always points the larger root at the smaller one, so
         * parent[l] <= l always holds and one *ascending* sweep both flattens
         * the table and renumbers the roots in place: by the time l is reached,
         * whatever it points at is already a final label.
         *
         * Ascending root order is exactly raster order of first appearance -- a
         * component's root is the smallest provisional label in it, which is the
         * one created at its topmost-leftmost pixel -- so this reproduces the
         * flood fill's numbering with no first-seen bookkeeping. */
        int count = 0;
        for (size_t label = 1; label < parent.size(); ++label)
        {
            if (parent[label] == static_cast<int>(label))
            {
                parent[label] = ++count; /* a root: take the next final label */
            }
            else
            {
                parent[label] = parent[parent[label]];
            }
        }

        std::vector<Blob> blobs(static_cast<size_t>(count));
        std::vector<long> sumX(static_cast<size_t>(count), 0);
        std::vector<long> sumY(static_cast<size_t>(count), 0);
        std::vector<bool> seen(static_cast<size_t>(count), false);

        const size_t n = static_cast<size_t>(w) * h;
        for (size_t i = 0; i < n; ++i)
        {
            const int provisional = labels[i];
            if (provisional == 0)
            {
                continue;
            }
            const int final = parent[provisional];
            const size_t at = static_cast<size_t>(final) - 1u;
            const int x     = static_cast<int>(i % w);
            const int y     = static_cast<int>(i / w);

            labels[i]  = final;
            Blob& blob = blobs[at];
            if (!seen[at]) /* first pixel of this component in raster order */
            {
                seen[at]   = true;
                blob.x     = x;
                blob.y     = y;
                blob.width = x;  /* max-x so far; turned into a width below */
                blob.height = y; /* max-y so far                            */
            }
            ++blob.area;
            sumX[at] += x;
            sumY[at] += y;
            blob.x      = std::min(blob.x, x);
            blob.y      = std::min(blob.y, y);
            blob.width  = std::max(blob.width, x);
            blob.height = std::max(blob.height, y);
        }

        for (size_t k = 0; k < blobs.size(); ++k)
        {
            Blob& blob     = blobs[k];
            blob.width     = blob.width - blob.x + 1;
            blob.height    = blob.height - blob.y + 1;
            blob.centroidX = static_cast<float>(static_cast<double>(sumX[k]) / blob.area);
            blob.centroidY = static_cast<float>(static_cast<double>(sumY[k]) / blob.area);
        }

        if (outLabels != nullptr)
        {
            *outLabels = std::move(labels);
        }
        return blobs;
    }

} /* namespace ops */
} /* namespace sol */
