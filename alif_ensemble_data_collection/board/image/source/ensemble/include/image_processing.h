/* Copyright (C) 2023 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include <stdint.h>

#define RGB_BYTES 		3
#define RGBA_BYTES 		4
#define RGB565_BYTES	2
#define PIXEL_BYTES 	1

// Camera dimensions
#define CIMAGE_X		RTE_HM0360_CAMERA_SENSOR_FRAME_WIDTH
#define CIMAGE_Y		RTE_HM0360_CAMERA_SENSOR_FRAME_HEIGHT

// Image dimensions for processing
#define IMAGE_X         320
#define IMAGE_Y         240

/*error status*/
#define FRAME_FORMAT_NOT_SUPPORTED   -1
#define FRAME_OUT_OF_RANGE           -2


extern uint32_t exposure_under_count, exposure_low_count, exposure_high_count, exposure_over_count;

int frame_crop(const void *input_fb, uint32_t ip_row_size, uint32_t ip_col_size, uint32_t row_start, uint32_t col_start, void *output_fb, uint32_t op_row_size, uint32_t op_col_size, uint32_t bpp);
int crop_and_interpolate(uint8_t *image, uint32_t srcWidth, uint32_t srcHeight, uint32_t dstWidth, uint32_t dstHeight, uint32_t bpp);
int interpolate(uint8_t *image, uint32_t srcWidth, uint32_t srcHeight, uint32_t dstWidth, uint32_t dstHeight, uint32_t bpp);
void white_balance(int width, int height, const uint8_t *sp, uint8_t *dp);
int bayer_to_RGB(uint8_t *src, uint8_t *dest);

#endif /* IMAGE_PROCESSING_H_ */
