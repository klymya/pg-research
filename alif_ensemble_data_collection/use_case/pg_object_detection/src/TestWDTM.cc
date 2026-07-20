#include "TestWDTM.h"
#include "uart_tracelib.h"

#define _30SEC  (30*1000u)


int32_t testWDTM()
{
    int32_t result = -1;
    volatile uint8_t failedCountOld = PgWDTM::WatchDog::wdtmGetCount();
    tx_thread_sleep(_30SEC / (1000 / TX_TIMER_TICKS_PER_SECOND));

    if(failedCountOld == PgWDTM::WatchDog::wdtmGetCount())
    {
        result = -1;
    }
    else
    {
        result = 0;
    }

    return result;
}

int32_t test_wd_send(void)
{
    const uint8_t UART_START_WD[] = {0x0D, 0x0E};
    int size_bytes = 1;
    int32_t result = testWDTM();
    uint8_t byte_result = result ? 0 : 1;

    send_str(const_cast<uint8_t*>(UART_START_WD), sizeof(UART_START_WD));
    send_str(&size_bytes, sizeof(size_bytes));
    send_str(&byte_result, 1);

    return result;
}
