/**
 * @file wdtm.c
 * @brief Watch dog timer file.
 * 
 */

#include "pg_wdtm.h"
#include <RTE_Device.h>
#include <RTE_Components.h>
#include CMSIS_device_header
#include "pinconf.h"
#include "delay.h"
#include "services_lib_interface.h"
#include "log_macros.h"

#define WAKE_PORT   PORT_15
#define WAKE_PIN    PIN_4

#define DONE_PORT   PORT_0
#define DONE_PIN    PIN_3
#define WDTM_DELAY  (10u)

extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(LP);
ARM_DRIVER_GPIO *pGpioWake = &ARM_Driver_GPIO_(LP);

extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(DONE_PORT);
static ARM_DRIVER_GPIO *pGpioDone = &ARM_Driver_GPIO_(DONE_PORT);

/**************************** STATIC FUNCTIONS ****************************/

/**
 * @brief Toggles DONE pin with 
 * 
 */
static void pg_tpl5010_done();

/**************************** PUBLIC FUNCTIONS ****************************/

void pg_tpl5010_init(ARM_GPIO_SignalEvent_t callback)
{
    int32_t retW = pinconf_set(WAKE_PORT, WAKE_PIN, PINMUX_ALTERNATE_FUNCTION_0, PADCTRL_READ_ENABLE | PADCTRL_DRIVER_DISABLED_PULL_UP);
    int32_t retD = pinconf_set(DONE_PORT, DONE_PIN, PINMUX_ALTERNATE_FUNCTION_0, 0);

    if ((ARM_DRIVER_OK != retW) || (ARM_DRIVER_OK != retD))
    {
        printf_err("Failed to pinconf WDTM gpio");
    }

    retW = pGpioWake->Initialize(WAKE_PIN, callback);
    retD = pGpioDone->Initialize(DONE_PIN, NULL);

    if ((ARM_DRIVER_OK != retW) || (ARM_DRIVER_OK != retD))
    {
        printf_err("Failed to init WDTM gpio");
    }

    retW = pGpioWake->PowerControl(WAKE_PIN, ARM_POWER_FULL);
    retD = pGpioDone->PowerControl(DONE_PIN, ARM_POWER_FULL);

    if ((ARM_DRIVER_OK != retW) || (ARM_DRIVER_OK != retD))
    {
        printf_err("Failed to set power control for WDTM gpio");
    }

    retW = pGpioWake->SetDirection(WAKE_PIN, GPIO_PIN_DIRECTION_INPUT);
    retD = pGpioDone->SetDirection(DONE_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    if ((ARM_DRIVER_OK != retW) || (ARM_DRIVER_OK != retD))
    {
        printf_err("Failed to set direction for WDTM gpio");
    }

    pGpioDone->SetValue(DONE_PIN, GPIO_PIN_OUTPUT_STATE_LOW);

    uint32_t control = ARM_GPIO_IRQ_SENSITIVE_EDGE
                     | ARM_GPIO_IRQ_EDGE_SENSITIVE_SINGLE
                     | ARM_GPIO_IRQ_POLARITY_HIGH;

    retW = pGpioWake->Control(WAKE_PIN, ARM_GPIO_ENABLE_INTERRUPT, &control);

    if (ARM_DRIVER_OK != retW)
    {
        printf_err("Failed to set control for WDTM wake gpio");
    }

    pg_tpl5010_done();
}

void pg_tpl5010_wakecheck()
{
    if (pGpioWake->GetValue)
    {
        pg_tpl5010_done();
    }
} 

void pg_tpl5010_deinit()
{
    pGpioDone->Uninitialize(WAKE_PIN);
    pGpioWake->Uninitialize(DONE_PIN);
}

static void pg_tpl5010_done()
{
    pGpioDone->SetValue(DONE_PIN, GPIO_PIN_OUTPUT_STATE_HIGH);
    SERVICES_wait_ms(WDTM_DELAY);
    pGpioDone->SetValue(DONE_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
}

