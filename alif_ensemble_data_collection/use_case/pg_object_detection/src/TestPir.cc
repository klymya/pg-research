#include "TestPir.h"
#include "pyq1548.h"
#include "log_macros.h"
#include "tx_api.h"
#include "clk.h"
#include "uart_tracelib.h"


#define DETECTION_WAIT_TIME_MS  5000

static bool pir_inited = false;
static bool pir_motion_detected = false;

void on_motion_detected(void)
{
    info("Motion detected\r\n");
    pir_motion_detected = true;
}

int32_t test_pir(void)
{
    int32_t time_left = DETECTION_WAIT_TIME_MS;
    int32_t pir_val = 0;
    uint32_t stat_cfg = 0;
    pir_motion_detected = false;

    pir_init(on_motion_detected);

    pir_cfg_set(16, PYQ1548_BLIND_TIME_1SEC,
                PYQ1548_PULSE_CNTR_1, PYQ1548_WINDOW_TIME_2SEC,
                PYQ1548_OP_MODES_WAKEUP, PYQ1548_SIGNAL_SRC_PIR_BFP,
                PYQ1548_HPF_CUTOFF_0_2HZ, PYQ1548_COUNT_WITH_BPF_SIGN);
    pir_cfg_apply();
    uint32_t cfg = pir_cfg_get_val();
    info("PIR configuration set: 0x%x\r\n", cfg);
    pir_int_enable(true);
    tx_thread_sleep(100);


    while (time_left > 0) {
        if (pir_motion_detected) {
            break;
        }
        tx_thread_sleep(50);
        time_left -= 500;
    }
    pir_int_enable(false);
    pir_deinit();
    int32_t ret = pir_motion_detected ? 0 : -1;
    return ret;
}


int32_t test_pir_send(void)
{
    const uint8_t UART_START_PIR[] = {0x0A, 0x0B};
    int size_bytes = 1;
    int32_t result = test_pir();
    uint8_t byte_result = result ? 0 : 1;

    send_str(const_cast<uint8_t*>(UART_START_PIR), sizeof(UART_START_PIR));
    send_str(&size_bytes, sizeof(size_bytes));
    send_str(&byte_result, 1);

    return result;
}
