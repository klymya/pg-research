/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Implementation of the native image operators. See NativeOps.hpp.
 */
#include "NativeOps.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

        /* Separable blur. The horizontal pass feeds a rolling ring of
         * (2*radius + 1) rows instead of a whole-image float buffer -- at QVGA
         * RGB that is 19 KB rather than 900 KB. Reflect-101 near the top and
         * bottom edges only ever maps to rows within radius of that edge, so
         * every row the vertical pass needs for output row y lies in
         * [y - radius, y + radius] -- exactly what the ring holds.
         *
         * Both passes keep the original arithmetic exactly -- acc starts at
         * 0.0f and adds kernel[k] * sample for k = -radius..radius in that
         * order -- so the output is bit-identical to the straightforward form.
         * What changed is only addressing: interior columns skip Reflect101,
         * and the vertical pass resolves its tap rows once per output row
         * instead of once per tap. Same as the target NativeOps.cc. */
        const int ringRows     = 2 * radius + 1;
        const size_t rowStride = static_cast<size_t>(w) * c;
        std::vector<float> ring(rowStride * ringRows, 0.0f);

        auto horizontalPass = [&](int row) {
            float* dstRow         = &ring[static_cast<size_t>(row % ringRows) * rowStride];
            const uint8_t* srcRow = src.data.data() + static_cast<size_t>(row) * rowStride;
            for (int x = 0; x < w; ++x)
            {
                const bool interior = (x >= radius) && (x < w - radius);
                for (int ch = 0; ch < c; ++ch)
                {
                    float acc = 0.0f;
                    if (interior)
                    {
                        const uint8_t* s = srcRow + static_cast<size_t>(x - radius) * c + ch;
                        for (int k = 0; k < kernelSize; ++k)
                        {
                            acc += kernel[k] * static_cast<float>(s[static_cast<size_t>(k) * c]);
                        }
                    }
                    else
                    {
                        for (int k = -radius; k <= radius; ++k)
                        {
                            const int xx = Reflect101(x + k, w);
                            acc += kernel[k + radius] *
                                   static_cast<float>(srcRow[static_cast<size_t>(xx) * c + ch]);
                        }
                    }
                    dstRow[static_cast<size_t>(x) * c + ch] = acc;
                }
            }
        };

        /* Prime the ring with rows [0, radius]; row y then only needs y+radius. */
        int filledTo = -1;
        for (int row = 0; row <= radius && row < h; ++row)
        {
            horizontalPass(row);
            filledTo = row;
        }

        Image dst(w, h, c);
        std::vector<const float*> taps(static_cast<size_t>(kernelSize));
        for (int y = 0; y < h; ++y)
        {
            for (int row = filledTo + 1; row <= y + radius && row < h; ++row)
            {
                horizontalPass(row);
                filledTo = row;
            }

            for (int k = -radius; k <= radius; ++k)
            {
                const int yy = Reflect101(y + k, h);
                taps[static_cast<size_t>(k + radius)] =
                    &ring[static_cast<size_t>(yy % ringRows) * rowStride];
            }

            uint8_t* dstRow = &dst.data[static_cast<size_t>(y) * rowStride];
            for (size_t i = 0; i < rowStride; ++i)
            {
                float acc = 0.0f;
                for (int k = 0; k < kernelSize; ++k)
                {
                    acc += kernel[k] * taps[static_cast<size_t>(k)][i];
                }
                dstRow[i] = SaturateU8(acc);
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

        /* ---- Binary fast path --------------------------------------------
         *
         * Every mask the pipeline morphs is strictly 0/255 (model output, shadow
         * veto, and morphology of those). On such a mask the rank filter above
         * reduces to set logic:
         *   dilate -> 255 iff any in-range sample is 255 (border counts as 0)
         *   erode  -> 0   iff any in-range sample is 0   (border counts as 255)
         * so out-of-range samples never decide the result and can be skipped.
         * Each structuring-element row is one contiguous run [dy, -half..+half],
         * and "any hit in the run" is a difference of two per-row prefix counts:
         * O(rows) per pixel instead of O(offsets). Bit-identical to RankFilter
         * for binary input; anything else takes RankFilter itself. Same as the
         * target NativeOps.cc. */

        constexpr int kMaxKernelRows = 31;

        struct KernelRow
        {
            int dy;
            int half;
        };

        /** @brief Collapse @c offsets into rows; false unless every row is one
         *         contiguous run centred on dx = 0 (the ellipse always is). */
        bool ToKernelRows(const std::vector<std::pair<int, int>>& offsets,
                          KernelRow (&rows)[kMaxKernelRows], int& count)
        {
            count = 0;
            size_t i = 0;
            while (i < offsets.size())
            {
                const int dy = offsets[i].first;
                int lo = offsets[i].second;
                int hi = lo;
                int n  = 0;
                for (; i < offsets.size() && offsets[i].first == dy; ++i, ++n)
                {
                    lo = std::min(lo, offsets[i].second);
                    hi = std::max(hi, offsets[i].second);
                }
                if (count == kMaxKernelRows || lo != -hi || n != hi - lo + 1)
                {
                    return false;
                }
                rows[count].dy   = dy;
                rows[count].half = hi;
                ++count;
            }
            return true;
        }

        bool IsBinary(const Image& mask)
        {
            for (const uint8_t v : mask.data)
            {
                if (v != 0u && v != 255u)
                {
                    return false;
                }
            }
            return true;
        }

        Image BinaryRankFilter(const Image& mask, const KernelRow (&rows)[kMaxKernelRows],
                               int rowCount, bool dilate)
        {
            const int w = mask.width;
            const int h = mask.height;
            const size_t stride = static_cast<size_t>(w) + 1u;
            Image out(w, h, 1);

            /* prefix[y][x] = hits in row y before column x. A "hit" is the value
             * that decides the result: 255 for dilate, 0 for erode. uint16_t
             * holds any row up to 65,535 px. */
            std::vector<uint16_t> prefix(stride * static_cast<size_t>(h));
            for (int y = 0; y < h; ++y)
            {
                const uint8_t* src = &mask.data[static_cast<size_t>(y) * w];
                uint16_t* pre      = &prefix[static_cast<size_t>(y) * stride];
                uint16_t run       = 0u;
                pre[0]             = 0u;
                for (int x = 0; x < w; ++x)
                {
                    run = static_cast<uint16_t>(run + ((src[x] != 0u) == dilate ? 1u : 0u));
                    pre[x + 1] = run;
                }
            }

            const uint8_t onHit  = dilate ? 255u : 0u;
            const uint8_t onMiss = dilate ? 0u : 255u;
            for (int y = 0; y < h; ++y)
            {
                uint8_t* dst = &out.data[static_cast<size_t>(y) * w];
                for (int x = 0; x < w; ++x)
                {
                    bool hit = false;
                    for (int r = 0; r < rowCount && !hit; ++r)
                    {
                        const int yy = y + rows[r].dy;
                        if (yy < 0 || yy >= h)
                        {
                            continue; /* border row: never decides the result */
                        }
                        const int lo = std::max(x - rows[r].half, 0);
                        const int hi = std::min(x + rows[r].half, w - 1);
                        const uint16_t* pre = &prefix[static_cast<size_t>(yy) * stride];
                        hit = pre[hi + 1] != pre[lo];
                    }
                    dst[x] = hit ? onHit : onMiss;
                }
            }
            return out;
        }

    } /* namespace */

    Image MorphologyEx(const Image& mask, MorphOp op, int kernelSize)
    {
        const auto offsets = EllipseOffsets(kernelSize);

        KernelRow rows[kMaxKernelRows];
        int rowCount = 0;
        if (IsBinary(mask) && ToKernelRows(offsets, rows, rowCount))
        {
            /* A binary mask stays binary, so the second pass is safe too. */
            if (op == MorphOp::Open)
            {
                return BinaryRankFilter(BinaryRankFilter(mask, rows, rowCount, false),
                                        rows, rowCount, true);
            }
            return BinaryRankFilter(BinaryRankFilter(mask, rows, rowCount, true),
                                    rows, rowCount, false);
        }

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
         * flood fill's numbering with no first-seen bookkeeping.
         *
         * Components beyond kMaxConnectedComponents are dropped (their pixels
         * become background), exactly as on target, so both give the same
         * blobs even for degenerate masks. */
        int count = 0;
        bool truncated = false;
        for (size_t label = 1; label < parent.size(); ++label)
        {
            if (parent[label] != static_cast<int>(label))
            {
                parent[label] = parent[parent[label]];
                continue;
            }
            if (static_cast<size_t>(count) >= kMaxConnectedComponents)
            {
                if (!truncated)
                {
                    std::fprintf(stderr,
                                 "SoL: component count hit the %u cap; mask is degenerate\n",
                                 static_cast<unsigned>(kMaxConnectedComponents));
                    truncated = true;
                }
                parent[label] = 0;
                continue;
            }
            parent[label] = ++count; /* a root: take the next final label */
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
            labels[i]       = final;
            if (final == 0) /* beyond the cap */
            {
                continue;
            }

            const size_t at = static_cast<size_t>(final) - 1u;
            const int x     = static_cast<int>(i % w);
            const int y     = static_cast<int>(i / w);
            Blob& blob      = blobs[at];
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
