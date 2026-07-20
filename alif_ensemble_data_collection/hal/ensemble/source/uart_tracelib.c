/* This file was ported to work on Alif Semiconductor Ensemble family of devices. */

/* Copyright (C) 2022 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/*
 * Copyright (c) 2019-2021 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Uncomment this to disable traces to UART
//#define DISABLE_UART_TRACE

#include "Driver_USART.h"

#if !defined(DISABLE_UART_TRACE)
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdatomic.h>
#include <RTE_Device.h>
#include <RTE_Components.h>
#include CMSIS_device_header
#include "tx_api.h"
#include "log_macros.h"

#define CNTLQ     0x11
#define CNTLS     0x13
#define DEL       0x7F
#define BACKSPACE 0x08
#define CR        0x0D
#define LF        0x0A
#define ESC       0x1B

/* Select used UART.
 *  Supported UARTs:
 *      UART0
 *      UART2
 *      UART4
 */

/* UART Driver */
extern ARM_DRIVER_USART ARM_Driver_USART_(CONSOLE_UART);

/* UART Driver instance */
static ARM_DRIVER_USART *USARTdrv = &ARM_Driver_USART_(CONSOLE_UART);

const char * tr_prefix = NULL;
uint16_t prefix_len;
#define MAX_TRACE_LEN 256

static bool initialized = false;
TX_SEMAPHORE tracelib_semaphore;
static TX_EVENT_FLAGS_GROUP uart_tracelib_event_flags;

typedef enum {
    UART_CB_EVENT_SEND_COMPLETE = (1 << 0),
    UART_CB_EVENT_RECV_COMPLETE = (1 << 1),
    UART_CB_EVENT_RECV_TIMEOUT  = (1 << 2),
    UART_CB_EVENT_RECV_OVERFLOW = (1 << 3),
} UART_TRACELIB_CB_EVENTS;

void myUART_callback(uint32_t event)
{
    if (event & ARM_USART_EVENT_SEND_COMPLETE) {
        tx_event_flags_set(&uart_tracelib_event_flags,
                UART_CB_EVENT_SEND_COMPLETE, TX_OR);
    }
    if (event & ARM_USART_EVENT_RECEIVE_COMPLETE) {
        tx_event_flags_set(&uart_tracelib_event_flags,
                UART_CB_EVENT_RECV_COMPLETE, TX_OR);
    }
    if (event & ARM_USART_EVENT_RX_TIMEOUT) {
        tx_event_flags_set(&uart_tracelib_event_flags,
                UART_CB_EVENT_RECV_TIMEOUT, TX_OR);
    }
    if (event & ARM_USART_EVENT_RX_OVERFLOW) {
        tx_event_flags_set(&uart_tracelib_event_flags,
                UART_CB_EVENT_RECV_OVERFLOW, TX_OR);
    }
}

void tracelib_deinit(void)
{
    UINT status;

    if (!initialized)
        return;

    status = tx_semaphore_get(&tracelib_semaphore, TX_WAIT_FOREVER);
    if (status != TX_SUCCESS) {
        printf_err("Failed to get tracelib semaphore (0x%x)\r\n", status);
    }

    USARTdrv->PowerControl(ARM_POWER_OFF);
    USARTdrv->Uninitialize();

    status = tx_event_flags_delete(&uart_tracelib_event_flags);
    if (status != TX_SUCCESS) {
        printf_err("Failed to delete uart tracelib event flags (0x%x)\r\n", status);
    }
    status = tx_semaphore_delete(&tracelib_semaphore);
    if (status != TX_SUCCESS) {
        printf_err("Failed to delete tracelib semaphore (0x%x)\r\n", status);
    }
    initialized = false;
}

int tracelib_init(const char * prefix, uint32_t baudrate)
{
    int32_t ret = 0;
    UINT status;

    if (initialized)
        return ret;

    tr_prefix = prefix;
    if (tr_prefix) {
        prefix_len = strlen(tr_prefix);
    } else {
        prefix_len = 0;
    }

    status = tx_semaphore_create(&tracelib_semaphore,
            "uart tracelib semaphore", 1);
    if (status != TX_SUCCESS) {
        printf_err("Failed to create tracelib semaphore (0x%x)\r\n", status);
        return -1;
    }

    status = tx_event_flags_create(&uart_tracelib_event_flags,
            "uart tracelib event flags");
    if (status != TX_SUCCESS) {
        printf_err("Failed to create uart tracelib event flags (0x%x)\r\n", status);
        goto err;
    }

    /* Initialize UART driver */
    ret = USARTdrv->Initialize(myUART_callback);
    if(ret != ARM_DRIVER_OK)
    {
        goto err2;
    }

    /* Power up UART peripheral */
    ret = USARTdrv->PowerControl(ARM_POWER_FULL);
    if(ret != ARM_DRIVER_OK)
    {
        goto err2;
    }

    /* Configure UART to 115200 Bits/sec */
    ret =  USARTdrv->Control(ARM_USART_MODE_ASYNCHRONOUS |
                             ARM_USART_DATA_BITS_8       |
                             ARM_USART_PARITY_NONE       |
                             ARM_USART_STOP_BITS_1       |
                             ARM_USART_FLOW_CONTROL_NONE, baudrate);
    if(ret != ARM_DRIVER_OK)
    {
        goto err2;
    }

    /* Transmitter line */
    ret =  USARTdrv->Control(ARM_USART_CONTROL_TX, 1);
    if(ret != ARM_DRIVER_OK)
    {
        goto err2;
    }

    /* Receiver line */
    ret =  USARTdrv->Control(ARM_USART_CONTROL_RX, 1);
    if(ret != ARM_DRIVER_OK)
    {
        goto err2;
    }

    initialized = true;

    return 0;

err2:
    tx_event_flags_delete(&uart_tracelib_event_flags);
err:
    tx_semaphore_delete(&tracelib_semaphore);

    return ret;
}

int send_str(void* str, uint32_t len)
{
    int ret = ARM_DRIVER_ERROR;
    UINT status;
    ULONG actual_events = 0;

    if (!initialized) {
        return ret;
    }

    status = tx_semaphore_get(&tracelib_semaphore, TX_WAIT_FOREVER);
    if (status != TX_SUCCESS) {
        printf_err("Failed to get tracelib semaphore (send) (0x%x)\r\n", status);
        return -status;
    }

    status = tx_event_flags_set(&uart_tracelib_event_flags,
                ~UART_CB_EVENT_SEND_COMPLETE, TX_AND);
    if (status != TX_SUCCESS) {
        printf_err("Failed to clear uart (send) tracelib event flag (0x%x)\r\n", status);
        ret = -status;
        goto err;
    }

    ret = USARTdrv->Send(str, len);

    status = tx_event_flags_get(&uart_tracelib_event_flags, \
            UART_CB_EVENT_SEND_COMPLETE,                 \
            TX_AND_CLEAR,                                \
            &actual_events,                              \
            TX_WAIT_FOREVER);
    if (status != TX_SUCCESS) {
        printf_err("Failed to get tracelib event flag (send) (0x%x)\r\n", status);
        ret = -status;
    }

err:
    status = tx_semaphore_put(&tracelib_semaphore);
    if (status != TX_SUCCESS) {
        printf_err("Failed to put tracelib semaphore (send) (0x%x)\r\n", status);
        return -status;
    }

    return ret;
}

int recv_str(void* data, uint32_t len, uint32_t timeout_ms)
{
    UINT status;
    int ret = ARM_DRIVER_ERROR;
    ULONG actual_events = 0;
    uint32_t timeout_ticks = (timeout_ms == TX_WAIT_FOREVER) ? \
                    TX_WAIT_FOREVER : \
                    timeout_ms / (1000 / TX_TIMER_TICKS_PER_SECOND);

    if (!initialized)
        return ret;

    status = tx_semaphore_get(&tracelib_semaphore, timeout_ticks);
    if (status != TX_SUCCESS) {
        printf_err("Failed to get tracelib semaphore (recv) (0x%x)\r\n", status);
        return -status;
    }

    status = tx_event_flags_set(&uart_tracelib_event_flags,
                ~(UART_CB_EVENT_RECV_COMPLETE | UART_CB_EVENT_RECV_TIMEOUT), TX_AND);
    if (status != TX_SUCCESS) {
        printf_err("Failed to clear uart (recv) tracelib event flag (0x%x)\r\n", status);
        ret = -status;
        goto err;
    }

    ret = USARTdrv->Receive(data, len);

    status = tx_event_flags_get(&uart_tracelib_event_flags,           \
            UART_CB_EVENT_RECV_COMPLETE,                              \
            TX_AND_CLEAR,                                             \
            &actual_events,                                           \
            timeout_ticks);

    if (status != TX_SUCCESS) {
        if (status != TX_NO_EVENTS) {
            printf_err("Failed to get tracelib event flag (recv) (0x%x)\r\n", status);
            ret = -status;
        } else {
            ret = USARTdrv->GetRxCount();
            USARTdrv->Control(ARM_USART_ABORT_RECEIVE, 0);
        }
    } else {
        if (actual_events & UART_CB_EVENT_RECV_COMPLETE) {
            ret = len;
        } else {
            ret = USARTdrv->GetRxCount();
        }
    }

err:
    status = tx_semaphore_put(&tracelib_semaphore);
    if (status != TX_SUCCESS) {
        printf_err("Failed to put tracelib semaphore (recv) (0x%x)\r\n", status);
        return -status;
    }

    return ret;
}

void tracef(const char * format, ...)
{
    if (initialized) {
        static char buffer[MAX_TRACE_LEN];

        va_list args;
        va_start(args, format);
        if (prefix_len) {
            memcpy(buffer, tr_prefix, prefix_len);
        }
        vsnprintf(buffer + prefix_len, sizeof(buffer) - prefix_len, format, args);
        send_str(buffer, strlen(buffer));
        va_end(args);
    }
}

unsigned char UartPutc(unsigned char ch)
{
    if (ch == '\n') {
        (void)UartPutc('\r');
    }
    char c = ch;
    send_str(&c, 1);
    return ch;
}

int UartGetc(void)
{
    unsigned char c;

    if (recv_str(&c, 1, TX_WAIT_FOREVER)) {
        if (c == '\r') {
            c = '\n';
        }

        return c;
    }

    return -1;
}

unsigned int GetLine(char *lp, unsigned int len)
{
    unsigned int cnt = 0;
    int c;

    do {
        c = UartGetc();
        if (c < 0) {
            continue;
        }
        switch ((unsigned char)c) {
        case CNTLQ: /* ignore Control S/Q             */
        case CNTLS:
            break;
        case BACKSPACE:
        case DEL:
            if (cnt == 0) {
                break;
            }
            cnt--;          /* decrement count                */
            lp--;           /* and line pointer               */
            UartPutc(0x08); /* echo backspace                 */
            UartPutc(' ');
            UartPutc(0x08);
            fflush(stdout);
            break;
        case ESC:
        case 0:
            *lp = 0; /* ESC - stop editing line        */
            return 0;
        case CR: /* CR - done, stop editing line   */
            *lp = c;
            lp++;  /* increment line pointer         */
            cnt++; /* and count                      */
            c = LF;
            UartPutc(*lp = c); /* echo and store character       */
            fflush(stdout);
            lp++;  /* increment line pointer         */
            cnt++; /* and count                      */
            break;
        default:
            UartPutc(*lp = c); /* echo and store character       */
            fflush(stdout);
            lp++;  /* increment line pointer         */
            cnt++; /* and count                      */
            break;
        }
    } while (cnt < len - 2 && c != LF); /* check limit and CR             */
    *lp = 0;                            /* mark end of string             */
    return 1;
}

void tracelib_test(void)
{
    #define RECV_BUF_SIZE   500
    #define RECV_TIMEOUT_MS 10
    char buf[RECV_BUF_SIZE];
    int read_bytes;
    int total_bytes_read = 0;

    USARTdrv->Control(ARM_USART_MODE_ASYNCHRONOUS |
                             ARM_USART_DATA_BITS_8       |
                             ARM_USART_PARITY_NONE       |
                             ARM_USART_STOP_BITS_1       |
                             ARM_USART_FLOW_CONTROL_NONE, 1843200);
    do {
        read_bytes = recv_str(buf, RECV_BUF_SIZE, RECV_TIMEOUT_MS);
        if (read_bytes > 0) {
            if (!strncmp(buf, "STOP", 4)) {
                break;
            }
            total_bytes_read += read_bytes;
            buf[read_bytes] = '\0';
            send_str("ACK\r\n", 5);
        }
    } while (true);

    USARTdrv->Control(ARM_USART_MODE_ASYNCHRONOUS |
                             ARM_USART_DATA_BITS_8       |
                             ARM_USART_PARITY_NONE       |
                             ARM_USART_STOP_BITS_1       |
                             ARM_USART_FLOW_CONTROL_NONE, 115200);

    printf("Total bytes read: %d\r\n", total_bytes_read);
}
#else

int tracelib_init(const char * prefix, uint32_t baudrate)
{
    return 0;
}

int send_str(const char* str, uint32_t len)
{
    return 0;
}

void tracef(const char * format, ...)
{
}

unsigned int GetLine(char *user_input, unsigned int size)
{
    return 0;
}

#endif // DISABLE_UART_TRACE

/************************ (C) COPYRIGHT ALIF SEMICONDUCTOR *****END OF FILE****/
