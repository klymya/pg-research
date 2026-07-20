/**
 * @file pg_wdtm.h
 * @brief TPL5010 timer and watchdog. Connected to P15_4 and P0_3 pins on alif chip.
 * 
 */


#ifndef __PGWDTM_H__
#define __PGWDTM_H__

#include <stdio.h>
#include <stdbool.h>
#include "Driver_GPIO.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Inits TPL5010 gpio.
 * 
 * @param callback Function ptr for callback function to be passed to IRQ.
 */
void pg_tpl5010_init(ARM_GPIO_SignalEvent_t callback);

/**
 * @brief Deinits TPL5010 gpio.
 * 
 */
void pg_tpl5010_deinit();

/**
 * @brief Checks if WAKE signal is issued and responds with DONE signal.
 * 
 */
void pg_tpl5010_wakecheck();

#ifdef __cplusplus
}
#endif

#endif //__PGWDTM_H__