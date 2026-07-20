/* Copyright (c) 2024 PointGrab. All rights reserved. */
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

#include "Driver_USART.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <RTE_Device.h>
#include <RTE_Components.h>
#include CMSIS_device_header
#include "tx_api.h"
#include "log_macros.h"

/* UART Driver */
extern ARM_DRIVER_USART ARM_Driver_USART_(LP);

/* UART Driver instance */
static ARM_DRIVER_USART *USARTdrv = &ARM_Driver_USART_(LP);

static bool initialized = false;
static TX_SEMAPHORE radio_semaphore;
static TX_EVENT_FLAGS_GROUP uart_radio_event_flags;

typedef enum {
    UART_CB_EVENT_SEND_COMPLETE = (1 << 0),
    UART_CB_EVENT_RECV_COMPLETE = (1 << 1)
} UART_RADIO_CB_EVENTS;

static void radio_callback(uint32_t event)
{
    if (event & ARM_USART_EVENT_SEND_COMPLETE) {
        tx_event_flags_set(&uart_radio_event_flags,
                UART_CB_EVENT_SEND_COMPLETE, TX_OR);
    }
    if (event & ARM_USART_EVENT_RECEIVE_COMPLETE) {
        tx_event_flags_set(&uart_radio_event_flags,
                UART_CB_EVENT_RECV_COMPLETE, TX_OR);
    }
}

void radio_deinit(void)
{
    UINT status;

    if (!initialized)
        return;

    status = tx_semaphore_get(&radio_semaphore, TX_WAIT_FOREVER);
    if (status != TX_SUCCESS) {
        printf_err("Failed to get radio semaphore (0x%x)\r\n", status);
    }
    USARTdrv->PowerControl(ARM_POWER_OFF);
    USARTdrv->Uninitialize();
    status = tx_event_flags_delete(&uart_radio_event_flags);
    if (status != TX_SUCCESS) {
        printf_err("Failed to delete uart radio event flags (0x%x)\r\n", status);
    }
    status = tx_semaphore_delete(&radio_semaphore);
    if (status != TX_SUCCESS) {
        printf_err("Failed to delete radio semaphore (0x%x)\r\n", status);
    }
    initialized = false;
}

int radio_init(const uint32_t baudrate)
{
    int32_t ret = 0;
    UINT status;

    if (initialized)
        return ret;

    status = tx_semaphore_create(&radio_semaphore,
            "uart radio semaphore", 1);
    if (status != TX_SUCCESS) {
        printf_err("Failed to create radio semaphore (0x%x)\r\n", status);
        return -1;
    }

    status = tx_event_flags_create(&uart_radio_event_flags,
            "uart radio event flags");
    if (status != TX_SUCCESS) {
        printf_err("Failed to create uart radio event flags (0x%x)\r\n", status);
        goto err;
    }

    /* Initialize UART driver */
    ret = USARTdrv->Initialize(radio_callback);
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
    tx_event_flags_delete(&uart_radio_event_flags);
err:
    tx_semaphore_delete(&radio_semaphore);
    return ret;
}

int radio_send_data(void* data, uint32_t len)
{
    int ret = ARM_DRIVER_ERROR;
    ULONG actual_events = 0;
    UINT status;

    if (!initialized)
        return ret;

    status = tx_event_flags_set(&uart_radio_event_flags,
                ~UART_CB_EVENT_SEND_COMPLETE, TX_AND);
    if (status != TX_SUCCESS) {
        printf_err("Failed to clear uart (send) radio event flag (0x%x)\r\n", status);
        return status;
    }

    status = tx_semaphore_get(&radio_semaphore, TX_WAIT_FOREVER);
    if (status != TX_SUCCESS) {
        printf_err("Failed to get radio semaphore (send) (0x%x)\r\n", status);
        return status;
    }
    ret = USARTdrv->Send(data, len);
    status = tx_semaphore_put(&radio_semaphore);
    if (status != TX_SUCCESS) {
        printf_err("Failed to put radio semaphore (send) (0x%x)\r\n", status);
        return status;
    }

    status = tx_event_flags_get(&uart_radio_event_flags, \
            UART_CB_EVENT_SEND_COMPLETE,                 \
            TX_AND_CLEAR,                                \
            &actual_events,                              \
            TX_WAIT_FOREVER);

    if (status > 0)
        return status;

    return ret;
}

/*FIXMEAV: guard the function with additional recv semaphore
or must be called only from one thread in the project*/
int radio_recv_data(void* data, uint32_t len)
{
    int ret = ARM_DRIVER_ERROR;
    ULONG actual_events = 0;
    UINT status;

    if (!initialized)
        return ret;

    status = tx_event_flags_set(&uart_radio_event_flags,
                ~UART_CB_EVENT_RECV_COMPLETE, TX_AND);
    if (status != TX_SUCCESS) {
        printf_err("Failed to clear uart (recv) radio event flag (0x%x)\r\n", status);
        return status;
    }

    status = tx_semaphore_get(&radio_semaphore, TX_WAIT_FOREVER);
    if (status != TX_SUCCESS) {
        printf_err("Failed to get radio semaphore (recv) (0x%x)\r\n", status);
        return status;
    }
    ret = USARTdrv->Receive(data, len);
    status = tx_semaphore_put(&radio_semaphore);
    if (status != TX_SUCCESS) {
        printf_err("Failed to put radio semaphore (recv) (0x%x)\r\n", status);
        return status;
    }

    status = tx_event_flags_get(&uart_radio_event_flags, \
            UART_CB_EVENT_RECV_COMPLETE,                 \
            TX_AND_CLEAR,                                \
            &actual_events,                              \
            TX_WAIT_FOREVER);

    if (status > 0)
        return status;

    return ret;
}