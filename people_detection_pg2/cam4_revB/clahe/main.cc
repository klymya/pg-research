#include <stdio.h>
#include <fstream>
#include <sstream>

#include "img_320.cc"
#include "rgb2lab.h"
#include "clahe.h"

const int IMG_WIDTH = 320;
const int IMG_HEIGHT = 240;
const int IMG_CHANNELS = 3; // Assuming RGB image
const int IMG_NUM_BYTES = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;
const int IMG_NUM_BYTES_PER_CHANNEL = IMG_WIDTH * IMG_HEIGHT;


bool fillDataFromDecimalTxt(const char* filename, unsigned char* data, size_t length) {
    std::ifstream ifs(filename);
    if (!ifs) {
        return false;
    }

    size_t idx = 0;
    int value;

    while (ifs >> value) {
        if (idx >= length) {
            // More data than expected
            return false;
        }

        if (value < 0 || value > 255) {
            // Invalid byte value
            return false;
        }

        data[idx++] = static_cast<unsigned char>(value);
    }

    return (idx == length); // Ensure expected number of bytes read
}

void saveDataAsDecimalTxt(const char* filename, const unsigned char* data, size_t length) {
    std::ofstream ofs(filename);
    if (!ofs) {
        // Could not open file
        return;
    }

    for (size_t i = 0; i < length; ++i) {
        ofs << static_cast<int>(data[i]) << " ";
    }
    ofs << std::endl;
}

void saveDataAsDecimalTxt(const char* filename, const double* data, size_t length) {
    std::ofstream ofs(filename);
    if (!ofs) {
        // Could not open file
        return;
    }

    for (size_t i = 0; i < length; ++i) {
        ofs << static_cast<double>(data[i]) << " ";
    }
    ofs << std::endl;
}

double clamp(double d, double min, double max) {
    const double t = d < min ? min : d;
    return t > max ? max : t;
}

void apply_clahe(const uint8_t *input_buffer, uint8_t *output_buffer,
                 size_t width, size_t height) {
    size_t channel_buffer_size = width * height;

    uint8_t *l_channel = (uint8_t *)malloc(channel_buffer_size);
    double *a_channel = (double *)malloc(channel_buffer_size * sizeof(double));
    double *b_channel = (double *)malloc(channel_buffer_size * sizeof(double));

    if (!l_channel || !a_channel || !b_channel) {
        // Handle memory allocation failure
        free(l_channel);
        free(a_channel);
        free(b_channel);
        return;
    }

    for (size_t i = 0; i < channel_buffer_size; i++) {
        DoubleTriplet rgb_pixel = {
            input_buffer[i * 3] / 255.0,
            input_buffer[i * 3 + 1] / 255.0,
            input_buffer[i * 3 + 2] / 255.0
        };

        DoubleTriplet lab_pixel = labFromRgb(rgb_pixel);

        l_channel[i] = (uint8_t)(lab_pixel.L * 255.0 / 100.0);
        a_channel[i] = lab_pixel.A;
        b_channel[i] = lab_pixel.B;
    }

    int status = CLAHE(l_channel, width, height, 0, 255, 8, 8, 256, 3.0f);

    for (size_t i = 0; i < channel_buffer_size; i++) {
        DoubleTriplet lab_pixel = {
            l_channel[i] * 100.0 / 255.0,
            a_channel[i],
            b_channel[i]
        };

        DoubleTriplet rgb_pixel = rgbFromLab(lab_pixel);

        output_buffer[i * 3]     = (uint8_t)(clamp(rgb_pixel.r * 255.0, 0.0, 255.0));
        output_buffer[i * 3 + 1] = (uint8_t)(clamp(rgb_pixel.g * 255.0, 0.0, 255.0));
        output_buffer[i * 3 + 2] = (uint8_t)(clamp(rgb_pixel.b * 255.0, 0.0, 255.0));
    }

    free(l_channel);
    free(a_channel);
    free(b_channel);
}

void process_lenna() {
    const int LENA_IMG_WIDTH = 512;
    const int LENA_IMG_HEIGHT = 512;
    const int LENA_IMG_NUM_BYTES = LENA_IMG_WIDTH * LENA_IMG_HEIGHT;

    unsigned char *img = new unsigned char[LENA_IMG_NUM_BYTES];

    fillDataFromDecimalTxt("lenna_gray.txt", img, LENA_IMG_NUM_BYTES);

    unsigned short *img_short = new unsigned short[LENA_IMG_NUM_BYTES];

    int status = CLAHE(
        img, LENA_IMG_WIDTH, LENA_IMG_HEIGHT,
        0, 255, 8, 8, 256, 3.0f
    );
    printf("CLAHE status: %d\n", status);

    saveDataAsDecimalTxt("lenna_processed.txt", img, LENA_IMG_NUM_BYTES);

    delete[] img;
    delete[] img_short;
}

void process_lenna_rgb() {
    const int LENA_IMG_WIDTH = 512;
    const int LENA_IMG_HEIGHT = 512;
    const int LENA_IMG_NUM_BYTES = LENA_IMG_WIDTH * LENA_IMG_HEIGHT * IMG_CHANNELS;
    const int LENA_IMG_NUM_BYTES_PER_CHANNEL = LENA_IMG_WIDTH * LENA_IMG_HEIGHT;

    unsigned char *img = new unsigned char[LENA_IMG_NUM_BYTES];
    unsigned char *l_channel = new unsigned char[LENA_IMG_NUM_BYTES_PER_CHANNEL];

    fillDataFromDecimalTxt("lenna_rgb.txt", img, LENA_IMG_NUM_BYTES);

    printf("start processing\n");
    DoubleTriplet rgb_pixel, lab_pixel;
    for (int i = 0; i < LENA_IMG_NUM_BYTES_PER_CHANNEL; i++) {
        rgb_pixel = {{
            static_cast<double>(img[i * 3]) / 255.0,
            static_cast<double>(img[i * 3 + 1]) / 255.0,
            static_cast<double>(img[i * 3 + 2]) / 255.0
        }};
        lab_pixel = labFromRgb(rgb_pixel);

        l_channel[i] = static_cast<unsigned char>(lab_pixel.L * 255.0 / 100.0);
    }
    printf("done processing\n");

    saveDataAsDecimalTxt("lenna_l_channel.txt", l_channel, LENA_IMG_NUM_BYTES_PER_CHANNEL);

    int status = CLAHE(
        l_channel, LENA_IMG_WIDTH, LENA_IMG_HEIGHT,
        0, 255, 8, 8, 256, 3.0f
    );
    printf("CLAHE status: %d\n", status);

    saveDataAsDecimalTxt("lenna_l_channel_clahe.txt", l_channel, LENA_IMG_NUM_BYTES_PER_CHANNEL);

    delete[] img;
    delete[] l_channel;
}

void process_lenna_rgb_full() {
    const int LENA_IMG_WIDTH = 512;
    const int LENA_IMG_HEIGHT = 512;
    const int LENA_IMG_NUM_BYTES = LENA_IMG_WIDTH * LENA_IMG_HEIGHT * IMG_CHANNELS;
    const int LENA_IMG_NUM_BYTES_PER_CHANNEL = LENA_IMG_WIDTH * LENA_IMG_HEIGHT;

    unsigned char *img = new unsigned char[LENA_IMG_NUM_BYTES];
    unsigned char *l_channel = new unsigned char[LENA_IMG_NUM_BYTES_PER_CHANNEL];
    double *a_channel = new double[LENA_IMG_NUM_BYTES_PER_CHANNEL];
    double *b_channel = new double[LENA_IMG_NUM_BYTES_PER_CHANNEL];

    double *img_r_channel = new double[LENA_IMG_NUM_BYTES_PER_CHANNEL];
    double *img_g_channel = new double[LENA_IMG_NUM_BYTES_PER_CHANNEL];
    double *img_b_channel = new double[LENA_IMG_NUM_BYTES_PER_CHANNEL];

    fillDataFromDecimalTxt("lenna_rgb.txt", img, LENA_IMG_NUM_BYTES);

    printf("start processing\n");
    DoubleTriplet rgb_pixel, lab_pixel;
    for (int i = 0; i < LENA_IMG_NUM_BYTES_PER_CHANNEL; i++) {
        rgb_pixel = {{
            static_cast<double>(img[i * 3]) / 255.0,
            static_cast<double>(img[i * 3 + 1]) / 255.0,
            static_cast<double>(img[i * 3 + 2]) / 255.0
        }};
        lab_pixel = labFromRgb(rgb_pixel);

        l_channel[i] = static_cast<unsigned char>(lab_pixel.L * 255.0 / 100.0);
        a_channel[i] = lab_pixel.A;
        b_channel[i] = lab_pixel.B;
    }
    printf("done processing\n");

    saveDataAsDecimalTxt("lenna_l_channel.txt", l_channel, LENA_IMG_NUM_BYTES_PER_CHANNEL);

    int status = CLAHE(
        l_channel, LENA_IMG_WIDTH, LENA_IMG_HEIGHT,
        0, 255, 8, 8, 256, 3.0f
    );
    printf("CLAHE status: %d\n", status);

    saveDataAsDecimalTxt("lenna_l_channel_clahe.txt", l_channel, LENA_IMG_NUM_BYTES_PER_CHANNEL);

    for (int i = 0; i < LENA_IMG_NUM_BYTES_PER_CHANNEL; i++) {
        lab_pixel = {{static_cast<double>(l_channel[i]) / 255.0 * 100.0, a_channel[i], b_channel[i]}};
        rgb_pixel = rgbFromLab(lab_pixel);

        img[i * 3] = static_cast<unsigned char>(clamp(rgb_pixel.r * 255.0, 0.0, 255.0));
        img[i * 3 + 1] = static_cast<unsigned char>(clamp(rgb_pixel.g * 255.0, 0.0, 255.0));
        img[i * 3 + 2] = static_cast<unsigned char>(clamp(rgb_pixel.b * 255.0, 0.0, 255.0));

        img_r_channel[i] = rgb_pixel.r;
        img_g_channel[i] = rgb_pixel.g;
        img_b_channel[i] = rgb_pixel.b;
    }

    saveDataAsDecimalTxt("lenna_rgb_clahe.txt", img, LENA_IMG_NUM_BYTES);
    saveDataAsDecimalTxt("lenna_r_channel.txt", img_r_channel, LENA_IMG_NUM_BYTES_PER_CHANNEL);
    saveDataAsDecimalTxt("lenna_g_channel.txt", img_g_channel, LENA_IMG_NUM_BYTES_PER_CHANNEL);
    saveDataAsDecimalTxt("lenna_b_channel.txt", img_b_channel, LENA_IMG_NUM_BYTES_PER_CHANNEL);

    delete[] img;
    delete[] l_channel;
    delete[] a_channel;
    delete[] b_channel;
    delete[] img_r_channel;
    delete[] img_g_channel;
    delete[] img_b_channel;
}

void default_run() {
    // Assuming img_320 is defined in img_320.h as an array of bytes
    // and has been initialized with the image data.
    uint8_t *image_data = (uint8_t *)im0;  // Pointer to the image data

    // Example usage: Print the first 20 bytes of the image data
    saveDataAsDecimalTxt("image_data.txt", image_data, IMG_NUM_BYTES);

    uint8_t *output_data = new uint8_t[IMG_NUM_BYTES];

    apply_clahe(image_data, output_data, IMG_WIDTH, IMG_HEIGHT);
    saveDataAsDecimalTxt("output_data.txt", output_data, IMG_NUM_BYTES);

    delete[] output_data;
}

int main() {
    // process_lenna();
    // process_lenna_rgb();
    // process_lenna_rgb_full();
    default_run();
    return 0;
}
