#include <stdint.h>
#include <vector>
#include "Detection.hpp"

void sendDetectionDataThroughUart(const uint8_t *image_data, 
        const int ml_width,
        const int ml_height,
        const std::vector<arm::app::pg_object_detection::Detection>& detections)
{
    (void) image_data;
    (void) ml_width;
    (void) ml_height;
    (void) detections;
}