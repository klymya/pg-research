#ifndef JPEG_H
#define JPEG_H

#include <stdint.h>
#if 0
#include "Detection.hpp"

void jpeg_send_through_uart(const uint8_t *image_data,
        const int ml_width,
        const int ml_height,
        const std::vector<arm::app::pg_object_detection::Detection>* detections);

#else
void jpeg_send_through_uart(const uint8_t *image_data,
        const int ml_width,
        const int ml_height);
#endif        
#endif
