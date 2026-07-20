/* Copyright (C) 2022-2024 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/******************************************************************************
 * @file     board_DevKit.c
 * @brief    BOARD API implementation for Alif Development Kit (Rev. A and B)
 ******************************************************************************/

#include "board.h"

#if defined(BOARD_IS_PG_REVA_B3_VARIANT)
#include "app_map.h"
#include "global_map.h"
#include "Driver_GPIO.h"
#include "pinconf.h"
//#include "nx_eth_user.h"

#ifndef PORT_
#define _PORT_(n)  PORT_##n
#define  PORT_(n) _PORT_(n)
#endif

extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_BUTTON1_GPIO_PORT);
//extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(ETH_RST_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_CAMERA_POWER_GPIO_PORT);
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_CAMERA_RESET_GPIO_PORT);

void BOARD_Pinmux_Init()
{
	ARM_DRIVER_GPIO *BOARD_LEDRGB_GPIOdrv = &ARM_Driver_GPIO_(BOARD_LEDRGB_GPIO_PORT);
	ARM_DRIVER_GPIO *BOARD_BUTTON1_GPIOdrv = &ARM_Driver_GPIO_(BOARD_BUTTON1_GPIO_PORT);
	ARM_DRIVER_GPIO *BOARD_CAMERA_POWER_GPIOdrv = &ARM_Driver_GPIO_(BOARD_CAMERA_POWER_GPIO_PORT);
	ARM_DRIVER_GPIO *BOARD_CAMERA_RESET_GPIOdrv = &ARM_Driver_GPIO_(BOARD_CAMERA_RESET_GPIO_PORT);
	//ARM_DRIVER_GPIO *BOARD_ETH_RST_GPIOdrv = &ARM_Driver_GPIO_(ETH_RST_PORT);

	uint32_t config_uart_rx =
			PADCTRL_READ_ENABLE |
			PADCTRL_SCHMITT_TRIGGER_ENABLE |
			PADCTRL_DRIVER_DISABLED_PULL_UP;

	uint32_t config_gpio =
			PADCTRL_READ_ENABLE |
			PADCTRL_SCHMITT_TRIGGER_ENABLE |
			PADCTRL_OUTPUT_DRIVE_STRENGTH_4MA;
		
	uint32_t config_gpio_bus_keeper =
			PADCTRL_READ_ENABLE |
			PADCTRL_SCHMITT_TRIGGER_ENABLE |
			PADCTRL_DRIVER_DISABLED_BUS_REPEATER | 
			PADCTRL_OUTPUT_DRIVE_STRENGTH_4MA;

	uint32_t config_button =
			PADCTRL_READ_ENABLE |
			PADCTRL_SCHMITT_TRIGGER_ENABLE |
			PADCTRL_DRIVER_DISABLED_PULL_UP;

	uint32_t config_i2c =
			PADCTRL_READ_ENABLE |
			PADCTRL_DRIVER_DISABLED_HIGH_Z;

	uint32_t config_i3c =
			PADCTRL_READ_ENABLE |
			PADCTRL_SCHMITT_TRIGGER_ENABLE |
			PADCTRL_DRIVER_DISABLED_PULL_UP;

	/* GPIO interfaces - initial GPIO state is lowest power */
#if 0
	BOARD_BUTTON1_GPIOdrv->Initialize(BOARD_BUTTON1_PIN_NO, NULL);
	BOARD_BUTTON1_GPIOdrv->PowerControl(BOARD_BUTTON1_PIN_NO, ARM_POWER_FULL);
	BOARD_BUTTON1_GPIOdrv->SetDirection(BOARD_BUTTON1_PIN_NO, GPIO_PIN_DIRECTION_INPUT);
	pinconf_set(PORT_(BOARD_BUTTON1_GPIO_PORT), BOARD_BUTTON1_PIN_NO, PINMUX_ALTERNATE_FUNCTION_0, config_button);
#endif
	BOARD_LEDRGB_GPIOdrv->Initialize(BOARD_LEDR_PIN_NO, NULL);
	BOARD_LEDRGB_GPIOdrv->PowerControl(BOARD_LEDR_PIN_NO, ARM_POWER_FULL);
	BOARD_LEDRGB_GPIOdrv->SetValue(BOARD_LEDR_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
	BOARD_LEDRGB_GPIOdrv->SetDirection(BOARD_LEDR_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
	pinconf_set(BOARD_LEDRGB_GPIO_PORT, BOARD_LEDR_PIN_NO, PINMUX_ALTERNATE_FUNCTION_0, config_gpio);

	BOARD_LEDRGB_GPIOdrv->Initialize(BOARD_LEDG_PIN_NO, NULL);
	BOARD_LEDRGB_GPIOdrv->PowerControl(BOARD_LEDG_PIN_NO, ARM_POWER_FULL);
	BOARD_LEDRGB_GPIOdrv->SetValue(BOARD_LEDG_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
	BOARD_LEDRGB_GPIOdrv->SetDirection(BOARD_LEDG_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
	
	//*(volatile uint32_t *)(0x4900x000UL) = 1;     /* set GPIOx pin 0 high */
	//*(volatile uint32_t *)(0x4900x004UL) = 0xFF;  /* set all GPIOx pins as output */

	pinconf_set(BOARD_LEDRGB_GPIO_PORT, BOARD_LEDG_PIN_NO, PINMUX_ALTERNATE_FUNCTION_0, config_gpio_bus_keeper);

	BOARD_LEDRGB_GPIOdrv->Initialize(BOARD_LEDB_PIN_NO, NULL);
	BOARD_LEDRGB_GPIOdrv->PowerControl(BOARD_LEDB_PIN_NO, ARM_POWER_FULL);
	BOARD_LEDRGB_GPIOdrv->SetValue(BOARD_LEDB_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
	BOARD_LEDRGB_GPIOdrv->SetDirection(BOARD_LEDB_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
	pinconf_set(BOARD_LEDRGB_GPIO_PORT, BOARD_LEDB_PIN_NO, PINMUX_ALTERNATE_FUNCTION_0, config_gpio);

#if 0
	BOARD_ETH_RST_GPIOdrv->Initialize(ETH_RST_PIN, NULL);
	BOARD_ETH_RST_GPIOdrv->PowerControl(ETH_RST_PIN, ARM_POWER_FULL);
	BOARD_ETH_RST_GPIOdrv->SetValue(ETH_RST_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
	BOARD_ETH_RST_GPIOdrv->SetDirection(ETH_RST_PIN, GPIO_PIN_DIRECTION_OUTPUT);
	pinconf_set(ETH_RST_PORT, ETH_RST_PIN, PINMUX_ALTERNATE_FUNCTION_0, config_gpio);
	HAL_Delay(1000000);
	BOARD_ETH_RST_GPIOdrv->SetValue(ETH_RST_PIN, GPIO_PIN_OUTPUT_STATE_HIGH);

	BOARD_CAMERA_POWER_GPIOdrv->Initialize(BOARD_CAMERA_POWER_PIN_NO, NULL);
	BOARD_CAMERA_POWER_GPIOdrv->PowerControl(BOARD_CAMERA_POWER_PIN_NO, ARM_POWER_FULL);
	BOARD_CAMERA_POWER_GPIOdrv->SetValue(BOARD_CAMERA_POWER_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
	BOARD_CAMERA_POWER_GPIOdrv->SetDirection(BOARD_CAMERA_POWER_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
	pinconf_set(PORT_(BOARD_CAMERA_POWER_GPIO_PORT), BOARD_CAMERA_POWER_PIN_NO, PINMUX_ALTERNATE_FUNCTION_0, config_gpio);

	BOARD_CAMERA_RESET_GPIOdrv->Initialize(BOARD_CAMERA_RESET_PIN_NO, NULL);
	BOARD_CAMERA_RESET_GPIOdrv->PowerControl(BOARD_CAMERA_RESET_PIN_NO, ARM_POWER_FULL);
	BOARD_CAMERA_RESET_GPIOdrv->SetValue(BOARD_CAMERA_RESET_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
	BOARD_CAMERA_RESET_GPIOdrv->SetDirection(BOARD_CAMERA_RESET_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
	pinconf_set(PORT_(BOARD_CAMERA_RESET_GPIO_PORT), BOARD_CAMERA_RESET_PIN_NO, PINMUX_ALTERNATE_FUNCTION_0, config_gpio);
#endif
	/* CAMERA clock output */
	pinconf_set(PORT_0,  PIN_3, PINMUX_ALTERNATE_FUNCTION_6, 0);   				// P0_3: CAM_XVCLK (mux mode 6)



	/* UART2 interface */
	//pinconf_set(PORT_1, PIN_0, PINMUX_ALTERNATE_FUNCTION_1, config_uart_rx);	// P1_0: RX  (mux mode 1)
	//pinconf_set(PORT_1, PIN_1, PINMUX_ALTERNATE_FUNCTION_1, 0);					// P1_1: TX  (mux mode 1)

    /* UART3 interface */
	//pinconf_set(PORT_1,  PIN_2, PINMUX_ALTERNATE_FUNCTION_1, config_uart_rx);   // P1_2: RX  (mux mode 1)
	//pinconf_set(PORT_1,  PIN_3, PINMUX_ALTERNATE_FUNCTION_1, 0);                // P1_3: TX  (mux mode 1)

	/* UART4 interface */
	//pinconf_set(PORT_12, PIN_1, PINMUX_ALTERNATE_FUNCTION_2, config_uart_rx);	// P12_1: RX  (mux mode 2)
	//pinconf_set(PORT_12, PIN_2, PINMUX_ALTERNATE_FUNCTION_2, 0);				// P12_2: TX  (mux mode 2)

    /* UART6 interface */
    //pinconf_set(PORT_14, PIN_0, PINMUX_ALTERNATE_FUNCTION_2, config_uart_rx);   // P14_0: RX  (mux mode 2)
    //pinconf_set(PORT_14, PIN_1, PINMUX_ALTERNATE_FUNCTION_2, 0);                // P14_1: TX  (mux mode 2)

    /* I2C0 interface */
    //pinconf_set(PORT_3, PIN_4, PINMUX_ALTERNATE_FUNCTION_5, config_i2c);		// P3_4: SCL (mux mode 5)
    //pinconf_set(PORT_3, PIN_5, PINMUX_ALTERNATE_FUNCTION_5, config_i2c);		// P3_5: SDA (mux mode 5)


    /* I2C1 interface (camera, touch) */
    pinconf_set(PORT_7, PIN_2, PINMUX_ALTERNATE_FUNCTION_5, config_i2c);		// P7_2: SDA (mux mode 5)
    pinconf_set(PORT_7, PIN_3, PINMUX_ALTERNATE_FUNCTION_5, config_i2c);		// P7_3: SCL (mux mode 5)

	/* I3C interface (gyro) */
	//pinconf_set(PORT_7, PIN_6, PINMUX_ALTERNATE_FUNCTION_6, config_i3c);		// P7_6: SDA (mux mode 6)
	//pinconf_set(PORT_7, PIN_7, PINMUX_ALTERNATE_FUNCTION_6, config_i3c);		// P7_7: SCL (mux mode 6)

}

void BOARD_Clock_Init()
{
	/* Configure any SoC clock muxes and dividers
	 * if not already covered in driver code
	 */

	/* CFGPER0 Registers */
	/* CFGPER0_BASE
	 * ipclk_force[31] set to 1 to force peripheral clocks on
	 * pclk_force[30] set to 1 to force interface clocks (PCLK) on
	 */
  //HW_REG32(CFGPER0_BASE, 0) = (1U << 31) | (1U << 30);

    /* Camera pixel clock (XVCLK) control
     * camera_pixclk_ctrl.divider[24:16] up to 0x1FF
     * camera_pixclk_ctrl.select[4] 0: 400 or 1: 480
     * camera_pixclk_ctrl.enable[0]
     */
  //HW_REG32(CAMERA_PIXCLK_CTRL, 0) = (32U << 16) | (0U << 4) | 1U;

    /* CGU Registers */
    /* OSC Control Register
     * sys_xtal_sel[0] 0: 76P8 HFRC, 1: 38P4 HFXO
     * SYS_REFCLK, SYSPLL, CPUPLL
     *
     * periph_xtal_sel[4] 0: 76P8 HFRC, 1: 38P4 HFXO
     * SYS_UART, MRAM_EFUSE, I2S, CAN_FD, MIPI, ES0, ES1
     *
     * clkmon_ena[16] 0: disable 1: enable HFXO clock monitor
     * xtal_dead[20] read-only status of HFXO clock monitor
     */
  //HW_REG32(CLOCKGEN_BASE, 0x00) &= ~(1U | 1U << 4);
  //HW_REG32(CLOCKGEN_BASE, 0x00) |=  (1U | 1U << 4);

    /* CLK Enable Register
     * hfxo[18] output of sys_xtal_clk (76P8 HFRC or 38P4 HFXO)
     * clk160M [20]
     * clk100M [21]
     * clk20M  [22]
     * clk38p4M[23]
     */
  //HW_REG32(CLOCKGEN_BASE, 0x14) |=  1U << 18;
  //HW_REG32(CLOCKGEN_BASE, 0x14) |= 15U << 20;

  //HW_REG32(VBATREGS_SE, 0) = 0;               // 0: 32.768kHz LFRC, 1: 32.768kHz LFXO
}

void BOARD_Power_Init()
{
	/* Configure any board level power supplies */
}

void BOARD_BUTTON1_GetState(BOARD_BUTTON_STATE *state)
{
	ARM_DRIVER_GPIO *BOARD_BUTTON1_GPIOdrv = &ARM_Driver_GPIO_(BOARD_BUTTON1_GPIO_PORT);
	uint32_t value;
	BOARD_BUTTON1_GPIOdrv->GetValue(BOARD_BUTTON1_PIN_NO, &value);
	*state = (BOARD_BUTTON_STATE)value;
}

void BOARD_LEDR_Control(BOARD_LED_STATE state)
{
	ARM_DRIVER_GPIO *BOARD_LEDRGB_GPIOdrv = &ARM_Driver_GPIO_(BOARD_LEDRGB_GPIO_PORT);
	BOARD_LEDRGB_GPIOdrv->SetValue(BOARD_LEDR_PIN_NO, (GPIO_PIN_OUTPUT_STATE) state);
}

void BOARD_LEDG_Control(BOARD_LED_STATE state)
{
	ARM_DRIVER_GPIO *BOARD_LEDG_GPIOdrv = &ARM_Driver_GPIO_(BOARD_LEDRGB_GPIO_PORT);
	BOARD_LEDG_GPIOdrv->SetValue(BOARD_LEDG_PIN_NO, (GPIO_PIN_OUTPUT_STATE) state);
}

void BOARD_LEDB_Control(BOARD_LED_STATE state)
{
	ARM_DRIVER_GPIO *BOARD_LEDG_GPIOdrv = &ARM_Driver_GPIO_(BOARD_LEDRGB_GPIO_PORT);
	BOARD_LEDG_GPIOdrv->SetValue(BOARD_LEDB_PIN_NO, (GPIO_PIN_OUTPUT_STATE) state);
}

#endif
