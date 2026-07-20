/**
 * Copyright (C) 2024 PointGrab. All rights reserved.
 * Copyright (C) 2023 Bosch Sensortec GmbH. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "Driver_SPI.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <RTE_Device.h>
#include <RTE_Components.h>
#include CMSIS_device_header
#include "bme68x.h"
#include "bme68x_common.h"
#include "log_macros.h"
#include "system_utils.h"
#include "pinconf.h"
#include "tx_api.h"

/* When set to one SPI0 is used, otherwise SPI3 as on our custom board */
#define ALIF_BOARD_TEST 0

#if ALIF_BOARD_TEST == 1
#define SPI_INTERFACE   0
#else
#define SPI_INTERFACE   3
#endif
/******************************************************************************/
/*!                Static variable definition                                 */
static uint8_t dev_addr;

extern ARM_DRIVER_SPI ARM_Driver_SPI_(SPI_INTERFACE);
static ARM_DRIVER_SPI *ptrSPI = &ARM_Driver_SPI_(SPI_INTERFACE);

static TX_EVENT_FLAGS_GROUP spi_event_flag;

#define SPI_CALLBACK_EVENT   (1 << 0)
#define SPI_MAX_TRANSFER_LEN (32)

/**
 * @fn      static int32_t pinmux_config(void)
 * @brief   SPI pinmux configuration.
 * @note    none.
 * @param   none.
 * @retval  execution status.
 */
static int32_t pinmux_config(void)
{
    int32_t ret = ARM_DRIVER_OK;

#if ALIF_BOARD_TEST != 1
    /* pinmux configurations for SPI3 pins (using A version pins) */
    ret = pinconf_set(PORT_12, PIN_4, PINMUX_ALTERNATE_FUNCTION_2, PADCTRL_READ_ENABLE);
    if (ret)
    {
        printf_err("ERROR: Failed to configure PINMUX for SPI3_MISO_PIN\n");
        return ret;
    }
    ret = pinconf_set(PORT_12, PIN_5, PINMUX_ALTERNATE_FUNCTION_2, 0);
    if (ret)
    {
        printf_err("ERROR: Failed to configure PINMUX for SPI3_MOSI_PIN\n");
        return ret;
    }
    ret = pinconf_set(PORT_12, PIN_6, PINMUX_ALTERNATE_FUNCTION_2, 0);
    if (ret)
    {
        printf_err("ERROR: Failed to configure PINMUX for SPI3_CLK_PIN\n");
        return ret;
    }
    ret = pinconf_set(PORT_12, PIN_7, PINMUX_ALTERNATE_FUNCTION_3, 0);
    if (ret)
    {
        printf_err("ERROR: Failed to configure PINMUX for SPI3_SS_PIN\n");
        return ret;
    }
#else
    /* pinmux configurations for SPI0 pins (using B version pins) */
    ret = pinconf_set(PORT_5, PIN_0, PINMUX_ALTERNATE_FUNCTION_4, PADCTRL_READ_ENABLE);
    if (ret)
    {
        printf("ERROR: Failed to configure PINMUX for SPI0_MISO_PIN\n");
        return ret;
    }
    ret = pinconf_set(PORT_5, PIN_1, PINMUX_ALTERNATE_FUNCTION_4, 0);
    if (ret)
    {
        printf("ERROR: Failed to configure PINMUX for SPI0_MOSI_PIN\n");
        return ret;
    }
    ret = pinconf_set(PORT_5, PIN_3, PINMUX_ALTERNATE_FUNCTION_3, 0);
    if (ret)
    {
        printf("ERROR: Failed to configure PINMUX for SPI0_CLK_PIN\n");
        return ret;
    }
    ret = pinconf_set(PORT_5, PIN_2, PINMUX_ALTERNATE_FUNCTION_4, 0);
    if (ret)
    {
        printf("ERROR: Failed to configure PINMUX for SPI0_SS_PIN\n");
        return ret;
    }
#endif
    return ret;
}

/**
 * @fn      static void SPI_cb_func (uint32_t event)
 * @brief   SPI callback function.
 * @note    none.
 * @param   event: SPI event.
 * @retval  none.
 */
static void spi_cb_func(uint32_t event)
{
    if (event == ARM_SPI_EVENT_TRANSFER_COMPLETE)
    {
        tx_event_flags_set(&spi_event_flag, SPI_CALLBACK_EVENT, TX_OR);
    }
}

static void spi_init(void)
{
    int32_t ret;
    uint32_t spi_control;

    ret = tx_event_flags_create(&spi_event_flag, "SPI_EVENT_FLAG");
    if (ret != TX_SUCCESS) {
        printf_err("failed to create spi%d event flag\r\n", SPI_INTERFACE);
        return;
    }

    ret = pinmux_config();
    if (ret) {
        printf_err("Error in pinmux configuration\r\n");
        return;
    }

    ret = ptrSPI->Initialize(spi_cb_func);
    if (ret != ARM_DRIVER_OK) {
        printf_err("ERROR: Failed to initialize the SPI%d\r\n", SPI_INTERFACE);
        return;
    }

    ret = ptrSPI->PowerControl(ARM_POWER_FULL);
    if (ret != ARM_DRIVER_OK) {
        printf_err("ERROR: Failed to power SPI%d\r\n", SPI_INTERFACE);
        goto error_spi_uninitialize;
    }

    spi_control = (ARM_SPI_MODE_MASTER | \
                    ARM_SPI_SS_MASTER_HW_OUTPUT | \
                    ARM_SPI_CPOL0_CPHA0 | \
                    ARM_SPI_DATA_BITS(8));

    /* Baudrate is 1MHz */
    ret = ptrSPI->Control(spi_control, 1000000);
    if (ret != ARM_DRIVER_OK) {
        printf_err("ERROR: Failed to configure SPI%d\r\n", SPI_INTERFACE);
        goto error_spi_power_off;
    }

    return;

error_spi_power_off :
    ret = ptrSPI->PowerControl(ARM_POWER_OFF);
    if (ret != ARM_DRIVER_OK) {
        printf_err("ERROR in SPI%d power off\r\n", SPI_INTERFACE);
    }

error_spi_uninitialize :
    ret = ptrSPI->Uninitialize();
    if (ret != ARM_DRIVER_OK) {
        printf_err("ERROR in SPI%d un-initialization\r\n", SPI_INTERFACE);
    }
}

static int32_t spi_transfer(const void *data_out, void *data_in, uint32_t len)
{
    int32_t ret;
    ULONG events;

    ptrSPI->Control(ARM_SPI_CONTROL_SS, ARM_SPI_SS_ACTIVE);
    tx_event_flags_set(&spi_event_flag, ~SPI_CALLBACK_EVENT, TX_AND);
    ret = ptrSPI->Transfer(data_out, data_in, len);
    tx_event_flags_get (&spi_event_flag, SPI_CALLBACK_EVENT,
        TX_AND_CLEAR, &events, TX_WAIT_FOREVER);
    ptrSPI->Control(ARM_SPI_CONTROL_SS, ARM_SPI_SS_INACTIVE);
    if (ret != ARM_DRIVER_OK) {
        printf_err("ERROR: SPI%d Failed to transfer data\r\n", SPI_INTERFACE);
    }

    return ret;
}

static inline BME68X_INTF_RET_TYPE bme68x_spi_transfer(uint8_t reg_addr, uint8_t *reg_data,
            uint32_t len, bool write_func)
{
    uint8_t tx_buf[SPI_MAX_TRANSFER_LEN] = {reg_addr};
    uint8_t rx_buf[SPI_MAX_TRANSFER_LEN];
    BME68X_INTF_RET_TYPE ret;

    if (len > (SPI_MAX_TRANSFER_LEN - 1)) {
        printf_err("Overflow of spi%d buffers, increase SPI_MAX_TRANSFER_LEN\r\n", SPI_INTERFACE);
        return BME68X_E_INVALID_LENGTH;
    }

    if (write_func) {
        memcpy(&tx_buf[1], reg_data, len);
    }

    ret = spi_transfer(tx_buf, rx_buf, len + 1);

    if (!write_func) {
        memcpy(reg_data, &rx_buf[1], len);
    }

    return ret;
}
/******************************************************************************/
/*!                User interface functions                                   */
/*!
 * SPI read function map to ALIF platform
 */
BME68X_INTF_RET_TYPE bme68x_spi_read(uint8_t reg_addr, uint8_t *reg_data,
            uint32_t len, void *intf_ptr)
{
    (void)intf_ptr;

    return bme68x_spi_transfer(reg_addr, reg_data, len, false);
}

/*!
 * SPI write function map to ALIF platform
 */
BME68X_INTF_RET_TYPE bme68x_spi_write(uint8_t reg_addr, uint8_t *reg_data,
            uint32_t len, void *intf_ptr)
{
    (void)intf_ptr;

    return bme68x_spi_transfer(reg_addr, reg_data, len, true);
}

/*!
 * Delay function map to ALIF platform
 */
void bme68x_delay_us(uint32_t period, void *intf_ptr)
{
    (void)intf_ptr;
    ULONG one_tick_us = (1000 / TX_TIMER_TICKS_PER_SECOND) * 1000;
    ULONG sleep_ticks = period / one_tick_us;
    int32_t sleep_us = period % one_tick_us;

    if (sleep_ticks) {
        tx_thread_sleep(sleep_ticks);
    }
    if (sleep_us > 0) {
        sys_busy_loop_us(sleep_us);
    }
}

void bme68x_check_rslt(const char api_name[], int8_t rslt)
{
    switch (rslt) {
        case BME68X_OK:
            /* Do nothing */
            break;
        case BME68X_E_NULL_PTR:
            printf_err("API name [%s]  Error [%d] : Null pointer\r\n", api_name, rslt);
            break;
        case BME68X_E_COM_FAIL:
            printf_err("API name [%s]  Error [%d] : Communication failure\r\n", api_name, rslt);
            break;
        case BME68X_E_INVALID_LENGTH:
            printf_err("API name [%s]  Error [%d] : Incorrect length parameter\r\n", api_name, rslt);
            break;
        case BME68X_E_DEV_NOT_FOUND:
            printf_err("API name [%s]  Error [%d] : Device not found\r\n", api_name, rslt);
            break;
        case BME68X_E_SELF_TEST:
            printf_err("API name [%s]  Error [%d] : Self test error\r\n", api_name, rslt);
            break;
        case BME68X_W_NO_NEW_DATA:
            printf_err("API name [%s]  Warning [%d] : No new data found\r\n", api_name, rslt);
            break;
        default:
            printf_err("API name [%s]  Error [%d] : Unknown error code\r\n", api_name, rslt);
            break;
    }
}

int8_t bme68x_interface_init(struct bme68x_dev *bme, uint8_t intf)
{
    int8_t rslt = BME68X_OK;

    /* Bus configuration : I2C */
    if (intf == BME68X_I2C_INTF) {
        printf_err("I2C Interface is not supported\r\n");
        return BME68X_E_DEV_NOT_FOUND;
    } else if (intf == BME68X_SPI_INTF) {
        info("SPI%d Interface\r\n", SPI_INTERFACE);
        spi_init();
        bme->read = bme68x_spi_read;
        bme->write = bme68x_spi_write;
        bme->intf = BME68X_SPI_INTF;
    }

    bme->delay_us = bme68x_delay_us;
    bme->intf_ptr = &dev_addr;
    /* The ambient temperature in deg C is used for defining the heater temperature */
    bme->amb_temp = 25;

    return rslt;
}

void bme68x_alif_deinit(void)
{
    int32_t ret;

    ret = ptrSPI->PowerControl(ARM_POWER_OFF);
    if (ret != ARM_DRIVER_OK) {
        printf_err("ERROR in SPI%d power off\r\n", SPI_INTERFACE);
    }

    ret = ptrSPI->Uninitialize();
    if (ret != ARM_DRIVER_OK) {
        printf_err("ERROR in SPI%d un-initialization\r\n", SPI_INTERFACE);
    }

    ret = tx_event_flags_delete(&spi_event_flag);
    if (ret != TX_SUCCESS) {
        printf_err("failed to delete spi%d event flag\r\n", SPI_INTERFACE);
    }
}