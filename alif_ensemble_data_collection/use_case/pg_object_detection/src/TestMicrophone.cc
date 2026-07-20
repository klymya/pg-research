#include "TestMicrophone.h"
#include "audio_data_pdm.h"
#include "hal.h"
#include <vector>
#define FIXED_POINT 16
#include "kiss_fft.h"
#include "kiss_fftr.h"

#define AUDIO_SAMPLES 16000 // 16k samples/sec, 1sec sample
#define AUDIO_STRIDE 8000 // 0.5 seconds

#define FFT_OUT_BUF_SIZE    (1 + AUDIO_STRIDE / 2)

static int16_t audio_inf[AUDIO_SAMPLES + AUDIO_STRIDE];
const auto audioRate = AUDIO_SAMPLES;
static kiss_fft_cpx outBuf[1 + AUDIO_STRIDE / 2];

static void test_fft(void)
{
    kiss_fft_scalar *inBuf = &audio_inf[0];

    kiss_fftr_cfg fftCfg = kiss_fftr_alloc(AUDIO_STRIDE, false, NULL, NULL);
    kiss_fftr(fftCfg, inBuf, outBuf);
    kiss_fft_free(fftCfg);
    for (int i = 0;i < FFT_OUT_BUF_SIZE;++i) {
        float res = sqrt(outBuf[i].i * outBuf[i].i + outBuf[i].r * outBuf[i].r);
        if (res > 100.0) {
            printf("[%i]=%f\r\n", i, res);
        }
    }
}

static int32_t test_microphone_init(void)
{
    int err = audio_init_pdm();

    if (err) {
        printf_err("audio_init_pdm failed with error: %d\r\n", err);
        return -1;
    }
 
    return 0;
}

static int32_t test_microphone_processing(void)
{
    int err = wait_for_audio_pdm(audio_inf, AUDIO_SAMPLES);

    if (err) {
        printf_err("hal_get_audio_data failed with error: %d\r\n", err);
        return -1;
    }

    return 0;
}

static void audio_send_through_uart(void* audio_buf, int size_bytes)
{
    const uint8_t UART_START_AUDIO[] = {0xAD, 0xD0};
    int idx = 0;

    send_str(const_cast<uint8_t*>(UART_START_AUDIO), sizeof(UART_START_AUDIO));
    send_str(&size_bytes, sizeof(size_bytes));

    //Write audio
    while (size_bytes > 0) {
        int size_to_write = (size_bytes > 256) ? 256 : size_bytes;

        send_str(&(static_cast<uint8_t *>(audio_buf)[idx]), size_to_write);
        size_bytes -= size_to_write;
        idx += size_to_write;
    }
}

int32_t test_microphone(void)
{
    int ret = 0;

    if (test_microphone_init()) {
        return -1;
    }

    if (test_microphone_processing()) {
        ret = -1;
    }

    audio_deinit_pdm();

    //test_fft();
    printf("dump binary memory /tmp/audio_mic.pcm 0x%X 0x%X \r\n",
                (unsigned int)&audio_inf,
                (unsigned int)((uint8_t *)&audio_inf + AUDIO_SAMPLES * 2));

    return ret;
}

int32_t test_microphone_send(void)
{
    if (test_microphone_init()) {
        return -1;
    }

    if (test_microphone_processing()) {
        return -1;
    }

    audio_deinit_pdm();
    audio_send_through_uart(audio_inf, AUDIO_SAMPLES * 2);

    return 0;
}

