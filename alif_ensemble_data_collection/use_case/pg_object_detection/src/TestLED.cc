#include "TestLED.h"
#include "ux_api.h"
#include "board.h"
//#include "platform_drivers.h"

void testLED(uint8_t loops)
{
    for (int it = 0; it < loops; it++)
    {
        BOARD_LEDG_Control(BOARD_LED_STATE_HIGH);
        tx_thread_sleep(1000u / (1000 / TX_TIMER_TICKS_PER_SECOND));
        BOARD_LEDG_Control(BOARD_LED_STATE_LOW);
        tx_thread_sleep(1000u / (1000 / TX_TIMER_TICKS_PER_SECOND));
        BOARD_LEDB_Control(BOARD_LED_STATE_HIGH);
        tx_thread_sleep(1000u / (1000 / TX_TIMER_TICKS_PER_SECOND));
        BOARD_LEDB_Control(BOARD_LED_STATE_LOW);
        tx_thread_sleep(1000u / (1000 / TX_TIMER_TICKS_PER_SECOND));
        BOARD_LEDR_Control(BOARD_LED_STATE_HIGH);
        tx_thread_sleep(1000u / (1000 / TX_TIMER_TICKS_PER_SECOND));
        BOARD_LEDR_Control(BOARD_LED_STATE_LOW);
        tx_thread_sleep(1000u / (1000 / TX_TIMER_TICKS_PER_SECOND));
    }
}