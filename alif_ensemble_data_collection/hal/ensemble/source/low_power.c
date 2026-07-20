#include <stdio.h>
#include <RTE_Components.h>
#include CMSIS_device_header
#include "Driver_LPTIMER.h"

/* LPTIMER Driver instance 0 */
extern ARM_DRIVER_LPTIMER DRIVER_LPTIMER0;
static ARM_DRIVER_LPTIMER *lptimerDrv = &DRIVER_LPTIMER0;

#define LPTIMER_CHANNEL    0

/**
  \fn           void lptimer_callback(uint8_t event)
  \brief        lptimer callback
  \return       none
*/
static void lptimer_callback(uint8_t event)
{
    int32_t ret = 0;

    if(event == ARM_LPTIMER_EVENT_UNDERFLOW)
    {
        /* User code for call back */
        //printf("\r\n LPTIMER CB\r\n");
        ret = lptimerDrv->Stop(LPTIMER_CHANNEL);
        if(ret != ARM_DRIVER_OK)
        {
            //printf("ERROR: Failed to Stop channel %d\n", LPTIMER_CHANNEL);
        }
    }

    return;
}

/**
  @fn           void lptimer_uninitialize(void)
  @brief        LPTIMER un-initializtion:
  @return       none
*/
static int lptimer_uninitialize(void)
{
    int32_t ret;

    /* Un-initialize lptimer driver */
    ret = lptimerDrv->Uninitialize(LPTIMER_CHANNEL);
    if (ret != ARM_DRIVER_OK) {
        printf("\r\n Error: LPTIMER Uninitialize failed.\r\n");
    }
    return ret;
}

/**
  @fn           int lptimer_power_off(void)
  @brief        LPTIMER power-off:
  @return       none
*/
static int lptimer_power_off(void)
{
    int32_t ret;

    /* Power off LPTIMER peripheral */
    ret = lptimerDrv->PowerControl(LPTIMER_CHANNEL, ARM_POWER_OFF);
    if(ret != ARM_DRIVER_OK) {
        printf("\r\n Error: LPTIMER Power OFF failed.\r\n");
        return ret;
    }

    ret = lptimer_uninitialize();

    return ret;
}

/**
  @fn           int set_lptimer(uint32_t  timeout)
  @brief        set lptimer timeout value (in second i.e. timeout = 10 means 10 sec)
  @return       none
*/
int32_t set_lptimer(uint32_t timeout_sec)
{
    int32_t   ret;
    uint32_t  count;

    /*
     *Configuring the lptimer channel for the timeout in seconds
     *Clock Source depends on RTE_LPTIMER_CHANNEL_CLK_SRC in RTE_Device.h
     *RTE_LPTIMER_CHANNEL_CLK_SRC = 0 : 32.768KHz freq (Default)
    */

    count = timeout_sec * (32768);

    /**< Loading the counter value >*/
    ret = lptimerDrv->Control(LPTIMER_CHANNEL, ARM_LPTIMER_SET_COUNT1, &count);
    if (ret != ARM_DRIVER_OK) {
        printf("ERROR: channel '%d'failed to load count\r\n", LPTIMER_CHANNEL);
        lptimer_power_off();
        return ret;
    }

    printf("\r\n Setting lptimer for %lu seconds \r\n", timeout_sec);

    ret = lptimerDrv->Start(LPTIMER_CHANNEL);
    if(ret != ARM_DRIVER_OK) {
        printf("ERROR: failed to start channel '%d'\r\n", LPTIMER_CHANNEL);
        lptimer_uninitialize();
    }

    return ret;
}

/**
  @fn           int32_t lptimer_init(void)
  @brief        LPTIMER Initialization
  @return       Status
*/
int32_t lptimer_init(void)
{
    int32_t ret;

    /* Initialize LPTIMER driver */
    ret = lptimerDrv->Initialize(LPTIMER_CHANNEL, lptimer_callback);
    if (ret != ARM_DRIVER_OK) {
        printf("\r\n Error: LPTIMER init failed\r\n");
        return ret;
    }

    /* Enable the power for LPTIMER */
    ret = lptimerDrv->PowerControl(LPTIMER_CHANNEL, ARM_POWER_FULL);
    if (ret != ARM_DRIVER_OK) {
        printf("\r\n Error: LPTIMER Power up failed\n");
        lptimer_uninitialize();
    }

    return ret;
}