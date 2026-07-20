/**************************************************************************//**
 * @file     jpeg.c
 * @version  V1.00
 * @brief    JPEG driver
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2024 PointGrab. All rights reserved.
 * @copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <stdio.h>
#include <string.h>
//#include <vector>
#include "jpeglib.h"
#include "image_processing.h"
//#include "uart_tracelib.h"
#include "system_utils.h"
//#include "Detection.hpp"

static uint8_t jpeg_buf[CIMAGE_X * CIMAGE_Y] __attribute__((aligned(32),section(".bss.jpeg_buf")));   // 640x480 = 307,200
//
// Encodes a 256 Greyscale image to JPEG directly to a memory buffer
// libJEPG will malloc() the buffer so the caller must free() it when
// they are finished with it.
//
// image    - the input greyscale image, 1 byte is 1 pixel.
// width    - the width of the input image
// height   - the height of the input image
// quality  - target JPEG 'quality' factor (max 100)
// comment  - optional JPEG NULL-termoinated comment, pass NULL for no comment.
// jpegSize - output, the number of bytes in the output JPEG buffer
// jpegBuf  - output, a pointer to the output JPEG buffer, must call free() when finished with it.
//
static void jpeg_encode_to_memory(unsigned char *image, int width, int height, int quality,
                           const char *comment, size_t *jpegSize, unsigned char **jpegBuf)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    JSAMPROW row_pointer[1];
    int row_stride;

    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_compress(&cinfo);
    cinfo.image_width = width;
    cinfo.image_height = height;

#if (TEST_GRAYSCALE == 1)
    // Input is GRAYSCALE, 1 byte per pixel
    cinfo.input_components = 1;
    cinfo.in_color_space   = JCS_GRAYSCALE;
#else
    // Input is RGB888, 3 byte per pixel
    cinfo.input_components = 3;
    cinfo.in_color_space   = JCS_RGB;
#endif

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, (boolean)TRUE);

    // jpegBuf in this sample is declared in main.c
    // Config libjpeg to encode to specified memory buffer and its buffer size
    jpeg_mem_dest(&cinfo, jpegBuf, jpegSize);

    jpeg_start_compress(&cinfo, (boolean)TRUE);

    // Add comment section if any..
    if (comment)
    {
        jpeg_write_marker(&cinfo, JPEG_COM, (const JOCTET *)comment, strlen(comment));
    }

    // raw stride = width * byte per pixel
    row_stride = width * cinfo.input_components;

    // Encode
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = &image[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
}

static void jpeg_encode(unsigned char *image, unsigned char *jBuf, size_t *jSize, int width, int height)
{
    jpeg_encode_to_memory(image, width, height, 85, "PointGrab", jSize, &jBuf);

    printf("JPEG image buffer range: 0x%08x to 0x%08x, JPEG size (bytes): %ld\n",
           (unsigned int)jBuf,
           (unsigned int)jBuf + (unsigned int)*jSize,
           *jSize);
}

static const uint8_t UART_START_JPEG[] = {0xAB, 0xBA};
static const uint8_t UART_START_BBOX[] = {0xAB, 0xBB};
#if 0
void jpeg_send_through_uart(const uint8_t *image_data,
        const int ml_width,
        const int ml_height,
        const std::vector<arm::app::pg_object_detection::Detection>* detections)
{
    size_t jpegSize;

    /* Encode JPEG */
    jpegSize = sizeof(jpeg_buf);
    jpeg_encode(const_cast<unsigned char *>(image_data), jpeg_buf, &jpegSize, ml_width, ml_height);
    /* Clean jpeg buffer to dump memory with debugger */
    RTSS_CleanDCache_by_Addr(jpeg_buf, sizeof(jpeg_buf));

    printf("dump binary memory /tmp/cam_image0_Bayer_to_RGB_320p.jpg 0x%X 0x%X \r\n",
                (unsigned int)&jpeg_buf,
                (unsigned int)((uint8_t *)&jpeg_buf + jpegSize));

    int size = jpegSize;
    int idx = 0;

    send_str(const_cast<uint8_t*>(UART_START_JPEG), sizeof(UART_START_JPEG));
    send_str(&jpegSize, sizeof(jpegSize));


    //Write Image row by row
    while (size > 0) {
        int size_to_write = (size > 256) ? 256 : size;

        send_str(&(jpeg_buf[idx]), size_to_write);
        size -= size_to_write;
        idx += size_to_write;
    }


    if (detections != nullptr) {
        printf("Sending %d detections\n", (*detections).size());
        int bbox_size_bytes = (*detections).size() * 7 * 4;
        send_str(const_cast<uint8_t*>(UART_START_BBOX), sizeof(UART_START_BBOX));
        send_str(&bbox_size_bytes, 4);

        //Write detections
        for (auto detection: *detections) {
            send_str(&detection.bbox, sizeof(detection.bbox));
            send_str(&detection.position, sizeof(detection.position));
            send_str(&detection.confidence, sizeof(detection.confidence));
        }
    }
}
#else
void jpeg_send_through_uart(const uint8_t *image_data,
    const int ml_width,
    const int ml_height)
{
    size_t jpegSize;

    /* Encode JPEG */
    jpegSize = sizeof(jpeg_buf);
    jpeg_encode(image_data, jpeg_buf, &jpegSize, ml_width, ml_height);
    /* Clean jpeg buffer to dump memory with debugger */
    RTSS_CleanDCache_by_Addr(jpeg_buf, sizeof(jpeg_buf));

    printf("dump binary memory /tmp/cam_image0_Bayer_to_RGB_320p.jpg 0x%X 0x%X \r\n",
                (unsigned int)&jpeg_buf,
                (unsigned int)((uint8_t *)&jpeg_buf + jpegSize));
#if 0
    int size = jpegSize;
    int idx = 0;

    send_str(const_cast<uint8_t*>(UART_START_JPEG), sizeof(UART_START_JPEG));
    send_str(&jpegSize, sizeof(jpegSize));


    //Write Image row by row
    while (size > 0) {
        int size_to_write = (size > 256) ? 256 : size;

        send_str(&(jpeg_buf[idx]), size_to_write);
        size -= size_to_write;
        idx += size_to_write;
    }


    if (detections != nullptr) {
        printf("Sending %d detections\n", (*detections).size());
        int bbox_size_bytes = (*detections).size() * 7 * 4;
        send_str(const_cast<uint8_t*>(UART_START_BBOX), sizeof(UART_START_BBOX));
        send_str(&bbox_size_bytes, 4);

        //Write detections
        for (auto detection: *detections) {
            send_str(&detection.bbox, sizeof(detection.bbox));
            send_str(&detection.position, sizeof(detection.position));
            send_str(&detection.confidence, sizeof(detection.confidence));
        }
    #endif
}
   
#endif
/*** (C) COPYRIGHT 2023 Nuvoton Technology Corp. ***/
