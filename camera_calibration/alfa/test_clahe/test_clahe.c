#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "clahe.h"

// ============================================================================
// Configuration: Maximum image dimensions for static scratch buffer
// ============================================================================
#define MAX_WIDTH  320
#define MAX_HEIGHT 240
#define MAX_PIXELS (MAX_WIDTH * MAX_HEIGHT)

з

// ============================================================================
// CLAHE application using fast Y' (luma) pipeline
// ============================================================================
// This implementation:
// 1. Extracts luma Y' from RGB using integer BT.601 formula (no FP, no pow)
// 2. Runs CLAHE on the luma channel
// 3. Reconstructs RGB by scaling each channel with ratio = y_new / y_old
//    using a reciprocal LUT to avoid per-pixel division
//
// Quality tradeoff:
// - Lab L* is perceptually uniform but requires expensive RGB↔Lab conversions
//   (double precision, pow() calls) which are slow on Cortex-M.
// - Y' (luma) is a good approximation of perceived brightness and can be
//   computed with integer math. CLAHE on Y' enhances local contrast effectively.
// - RGB reconstruction via ratio scaling preserves hue reasonably well.
// - Expected speedup: 10-50x on Cortex-M compared to Lab pipeline.
//
// Memory usage:
// - Static scratch buffer: MAX_PIXELS bytes (76800 bytes for 320×240)
// - Reciprocal LUT: 512 bytes (one-time initialization)
// - No per-frame allocations (malloc/free)
//
int apply_clahe(const uint8_t *__restrict input_buffer,
                uint8_t *__restrict output_buffer,
                size_t width, size_t height)
{
    const size_t num_pixels = width * height;

    // Sanity check: ensure image fits within our static buffer
    if (num_pixels > MAX_PIXELS) {
        fprintf(stderr, "Error: Image size %zux%zu exceeds MAX_PIXELS %d\n",
                width, height, MAX_PIXELS);
        return -1;
    }

    // Initialize reciprocal LUT (one-time, thread-safe for single-threaded use)
    init_reciprocal_lut();

    // Static scratch buffer for luma channel (no per-frame allocation)
    static uint8_t l_channel[MAX_PIXELS];

    // ========================================================================
    // Stage 1: Extract luma Y' from RGB
    // ========================================================================
    const uint8_t *in_ptr = input_buffer;

    for (size_t i = 0; i < num_pixels; ++i) {
        uint8_t r = *in_ptr++;
        uint8_t g = *in_ptr++;
        uint8_t b = *in_ptr++;

        l_channel[i] = luma_u8(r, g, b);
    }

    // ========================================================================
    // Stage 2: Apply CLAHE to luma channel
    // ========================================================================
    unsigned int uiXRes = (unsigned int)width;
    unsigned int uiYRes = (unsigned int)height;

    // Safety check for overflow (unlikely for 320×240, but good practice)
    if ((size_t)uiXRes != width || (size_t)uiYRes != height) {
        fprintf(stderr, "Error: Image dimensions overflow unsigned int\n");
        return -1;
    }

    int status = CLAHE(
        l_channel,
        uiXRes,
        uiYRes,
        0,          // Min
        255,        // Max
        8,          // uiNrX (number of contextual regions in X)
        8,          // uiNrY (number of contextual regions in Y)
        128,        // uiNrBins (histogram bins)
        3.0f        // fCliplimit (contrast limit)
    );

    if (status != 0) {
        fprintf(stderr, "Error: CLAHE processing failed with status %d\n", status);
        return -1;
    }

    // ========================================================================
    // Stage 3: Reconstruct RGB using ratio scaling
    // ========================================================================
    // For each pixel:
    //   y_old = luma(R, G, B) from input
    //   y_new = l_channel[i] after CLAHE
    //   ratio = y_new / y_old  (using reciprocal LUT)
    //   R' = clamp((R * ratio_q12) >> 12)
    //   G' = clamp((G * ratio_q12) >> 12)
    //   B' = clamp((B * ratio_q12) >> 12)
    //
    in_ptr = input_buffer;
    uint8_t *out_ptr = output_buffer;

    for (size_t i = 0; i < num_pixels; ++i) {
        uint8_t r = *in_ptr++;
        uint8_t g = *in_ptr++;
        uint8_t b = *in_ptr++;

        uint8_t y_old = luma_u8(r, g, b);
        uint8_t y_new = l_channel[i];

        // Handle black pixels (y_old == 0) to avoid division by zero
        if (y_old == 0) {
            // Option A: Set output to black (safe)
            *out_ptr++ = 0;
            *out_ptr++ = 0;
            *out_ptr++ = 0;
            continue;

            // Option B: Copy input unchanged (preserves original black pixels)
            // *out_ptr++ = r;
            // *out_ptr++ = g;
            // *out_ptr++ = b;
            // continue;
        }

        // Compute ratio in Q12 fixed-point: ratio_q12 = y_new * (1/y_old)
        uint32_t ratio_q12 = (uint32_t)y_new * inv_q12[y_old];

        // Scale each channel and clamp to [0..255]
        *out_ptr++ = clamp_u8((r * ratio_q12) >> 12);
        *out_ptr++ = clamp_u8((g * ratio_q12) >> 12);
        *out_ptr++ = clamp_u8((b * ratio_q12) >> 12);
    }

    return 0;
}
