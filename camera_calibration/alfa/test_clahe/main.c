#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Function prototype from test_clahe.c
int apply_clahe(const uint8_t *input_buffer, uint8_t *output_buffer,
                size_t width, size_t height);

int main(int argc, char *argv[]) {
    const char *input_path = "../ debug_locs/meeting_room.jpg";
    const char *output_path = "clahe_test_upd_colors.jpg";

    printf("Loading image from: %s\n", input_path);

    // Load the image
    int width, height, channels;
    uint8_t *img_data = stbi_load(input_path, &width, &height, &channels, 3);

    if (!img_data) {
        fprintf(stderr, "Error: Failed to load image '%s'\n", input_path);
        fprintf(stderr, "STB Error: %s\n", stbi_failure_reason());
        return 1;
    }

    printf("Image loaded successfully: %dx%d, %d channels\n", width, height, channels);

    // Allocate output buffer
    size_t buffer_size = width * height * 3;
    uint8_t *output_data = (uint8_t *)malloc(buffer_size);

    if (!output_data) {
        fprintf(stderr, "Error: Failed to allocate output buffer\n");
        stbi_image_free(img_data);
        return 1;
    }

    printf("Applying CLAHE...\n");

    // Apply CLAHE
    int clahe_status = apply_clahe(img_data, output_data, width, height);
    if (clahe_status != 0) {
        fprintf(stderr, "Error: CLAHE application failed\n");
        free(output_data);
        stbi_image_free(img_data);
        return 1;
    }

    printf("Saving result to: %s\n", output_path);

    // Save the result
    int result = stbi_write_jpg(output_path, width, height, 3, output_data, 95);

    if (!result) {
        fprintf(stderr, "Error: Failed to save image '%s'\n", output_path);
        free(output_data);
        stbi_image_free(img_data);
        return 1;
    }

    printf("Successfully saved CLAHE result to '%s'\n", output_path);

    // Cleanup
    free(output_data);
    stbi_image_free(img_data);

    return 0;
}
