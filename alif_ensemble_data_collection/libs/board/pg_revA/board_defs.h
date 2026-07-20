/* Copyright (C) 2022 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/******************************************************************************
 * @file     board.h
 * @brief    BOARD API
 ******************************************************************************/
//#define DC_BOARD
#define BAT_BOARD

#define BOARD_CAMERA_I2C_INSTANCE               1

#define BOARD_I2S_INSTANCE                      3

#define BOARD_PDM_INSTANCE                      2

#define BOARD_UART1_INSTANCE                    0 

#define BOARD_UART2_INSTANCE                    4

#define BOARD_UART3_INSTANCE                    3

#define BOARD_UART4_INSTANCE                    6

#define CLICKBOARD_I2C_INSTANCE                 0
#define CLICKBOARD_UART_INSTANCE                0
#define CLICKBOARD_SPI_INSTANCE                 3

#define BOARD_LED_COUNT                         3
#define BOARD_LEDRGB_COUNT                      3
#define BOARD_BUTTON_COUNT                      2

// <o> User BUTTON1 (JOY_SW4 - down) GPIO port number and pin number
#define BOARD_BUTTON1_GPIO_PORT                 3
#define BOARD_BUTTON1_PIN_NO                    6

// <o> User BUTTON2 (JOY_SW5 - click) GPIO port number and pin number
#define BOARD_BUTTON2_GPIO_PORT                 LP
#define BOARD_BUTTON2_PIN_NO                    4

#ifdef BAT_BOARD	
#define BOARD_LEDRGB_GPIO_PORT                  1               
#define BOARD_LEDR_PIN_NO                       3

// <o> User LEDG GPIO port number and pin number
#define BOARD_LEDG_PIN_NO                       4

// <o> User LEDB GPIO port number and pin number
#define BOARD_LEDB_PIN_NO                       5

#endif
#ifdef DC_BOARD	
// <o> User LEDR GPIO port number and pin number
#define BOARD_LEDRGB_GPIO_PORT                  1               
#define BOARD_LEDR_PIN_NO                       0

// <o> User LEDG GPIO port number and pin number
#define BOARD_LEDG_PIN_NO                       1

// <o> User LEDB GPIO port number and pin number
#define BOARD_LEDB_PIN_NO                       2
#endif

// <o> LCD panel reset GPIO port number and pin number
#define BOARD_LCD_RESET_GPIO_PORT               LP
#define BOARD_LCD_RESET_PIN_NO                  5

// <o> LCD panel backlight GPIO port number and pin number
#define BOARD_LCD_BACKLIGHT_GPIO_PORT           6
#define BOARD_LCD_BACKLIGHT_PIN_NO              1

// <o> Touch screen reset GPIO port number and pin number
#define BOARD_TOUCH_RESET_GPIO_PORT             4
#define BOARD_TOUCH_RESET_PIN_NO                0

// <o> Touch screen interrupt GPIO port number and pin number
#define BOARD_TOUCH_INT_GPIO_PORT               9
#define BOARD_TOUCH_INT_PIN_NO                  4

// <o> Camera module power GPIO port number and pin number
#define BOARD_CAMERA_POWER_GPIO_PORT            14
#define BOARD_CAMERA_POWER_PIN_NO               7

// <o> Camera module reset GPIO port number and pin number
#define BOARD_CAMERA_RESET_GPIO_PORT            15
#define BOARD_CAMERA_RESET_PIN_NO               0

// <o> Camera module flash GPIO port number and pin number
#define BOARD_CAMERA_FLASH_GPIO_PORT            9
#define BOARD_CAMERA_FLASH_PIN_NO               2

// <o> Camera module trigger GPIO port number and pin number
#define BOARD_CAMERA_TRIGGER_GPIO_PORT          9
#define BOARD_CAMERA_TRIGGER_PIN_NO             3
