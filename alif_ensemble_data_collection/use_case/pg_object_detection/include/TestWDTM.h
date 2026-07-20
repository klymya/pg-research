#pragma once
#include "PGWatchDog.h"
#include "pm.h"

#ifndef _TESTWDTM_H_
#define _TESTWDTM_H_

/**
 * @brief Run WDTM test.
 *
 */
int32_t testWDTM();


/**
 * @brief Send test result over UART.
 *
 */
int32_t test_wd_send(void);

#endif
