/* Copyright (C) 2022 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/* System Includes */
#include "RTE_Components.h"
#include "RTE_Device.h"
#include "global_map.h"

/* Project Includes */
#include "camera.h"
#include "Driver_GPIO.h"
#include "tx_api.h"

#include "board.h"

#ifndef RTE_CPI
#include "Driver_Camera_Controller.h"
extern ARM_DRIVER_CAMERA_CONTROLLER Driver_CAMERA0;
static const ARM_DRIVER_CAMERA_CONTROLLER * const camera = &Driver_CAMERA0;

#define CPI_CAMERA_SENSOR_CONFIGURE             CAMERA_SENSOR_CONFIGURE
#define CPI_EVENTS_CONFIGURE                    CAMERA_EVENTS_CONFIGURE
#define CPI_CAMERA_SENSOR_GAIN                  CAMERA_SENSOR_GAIN
#define ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED    ARM_CAMERA_CONTROLLER_EVENT_CAMERA_CAPTURE_STOPPED
#define RESOLUTION_PARAMETER                    CAMERA_RESOLUTION_560x560

#ifdef BOARD_CAMERA_POWER_GPIO_PORT
#define MANUAL_CAMERA_POWER
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_CAMERA_POWER_GPIO_PORT);
static ARM_DRIVER_GPIO * const GPIO_Driver_PWR = &ARM_Driver_GPIO_(BOARD_CAMERA_POWER_GPIO_PORT);
#endif
#else
#include "Driver_CPI.h"
extern ARM_DRIVER_CPI Driver_CPI;
static const ARM_DRIVER_CPI * const camera = &Driver_CPI;
#endif

#include "Driver_Common.h"
#include "image_processing.h"

#include <stdatomic.h>
#include <stdio.h>

#include "board.h"

/* Camera callback events */
typedef enum {
    CAM_CB_EVENT_CAPTURE_STOPPED      = (1 << 0),
    CAM_CB_EVENT_ERROR                = (1 << 1)
}CAMERA_CB_EVENTS;

static uint8_t* buf              = 0;
static bool initialized = false;
TX_EVENT_FLAGS_GROUP camera_event_flags;


static void CameraEventHandler(uint32_t event)
{
    (void)event;

    if (event & ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED) {
        /* Transfer Success: Capture Stop detected, Wake-up Thread. */
        //BOARD_LEDG_Control(BOARD_LED_STATE_LOW);
        tx_event_flags_set(&camera_event_flags, CAM_CB_EVENT_CAPTURE_STOPPED, TX_OR);
    }

    if (event & ARM_CPI_EVENT_ERR_CAMERA_INPUT_FIFO_OVERRUN) {
        /* Transfer Error: Received FIFO over-run, Wake-up Thread. */
        tx_event_flags_set(&camera_event_flags, CAM_CB_EVENT_ERROR, TX_OR);
    }

    if (event & ARM_CPI_EVENT_ERR_CAMERA_OUTPUT_FIFO_OVERRUN) {
        /* Transfer Error: Received FIFO over-run, Wake-up Thread. */
        tx_event_flags_set(&camera_event_flags, CAM_CB_EVENT_ERROR, TX_OR);
    }

    if (event & ARM_CPI_EVENT_ERR_HARDWARE) {
        /* Transfer Error: Received Hardware error, Wake-up Thread. */
        tx_event_flags_set(&camera_event_flags, CAM_CB_EVENT_ERROR, TX_OR);
    }
}

void camera_deinit(void)
{
    if (!initialized)
        return;
    camera->PowerControl(ARM_POWER_OFF);
    camera->Uninitialize();
    tx_event_flags_delete(&camera_event_flags);
}

int32_t camera_init(uint8_t* buffer)
{
    initialized = false;
    if (initialized)
        return 0;

    if (tx_event_flags_create(&camera_event_flags, "camera event flags") != TX_SUCCESS) {
        printf("Failed to create camera event flags\n");
        return -1;
    }
#ifdef MANUAL_CAMERA_POWER
    GPIO_Driver_PWR->SetValue(BOARD_CAMERA_POWER_PIN_NO, GPIO_PIN_OUTPUT_STATE_HIGH);
#endif

    //////////////////////////////////////////////////////////////////////////////
    // Camera initialization
    //////////////////////////////////////////////////////////////////////////////
#ifdef RESOLUTION_PARAMETER
    int32_t res = camera->Initialize(RESOLUTION_PARAMETER, CameraEventHandler);
#else
    //BOARD_LEDG_Control(BOARD_LED_STATE_HIGH);
    int32_t res = camera->Initialize(CameraEventHandler);
    //BOARD_LEDG_Control(BOARD_LED_STATE_LOW);
#endif

    if (res != ARM_DRIVER_OK) {
        return res;
    }

    res = camera->PowerControl(ARM_POWER_FULL);
    if (res != ARM_DRIVER_OK) {
        return res;
    }

#ifdef RESOLUTION_PARAMETER
    res = camera->Control(CPI_CAMERA_SENSOR_CONFIGURE, RESOLUTION_PARAMETER);
#else
    res = camera->Control(CPI_CAMERA_SENSOR_CONFIGURE, 0);
#endif
    if (res != ARM_DRIVER_OK) {
        return res;
    }

#ifdef CPI_CONFIGURE
    res = camera->Control(CPI_CONFIGURE, 0);
    if (res != ARM_DRIVER_OK) {
        return res;
    }
#endif

    res = camera->Control(CPI_EVENTS_CONFIGURE, 
            ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED | \
            ARM_CPI_EVENT_ERR_CAMERA_INPUT_FIFO_OVERRUN | \
            ARM_CPI_EVENT_ERR_CAMERA_OUTPUT_FIFO_OVERRUN | \
            ARM_CPI_EVENT_ERR_HARDWARE);
    if (res != ARM_DRIVER_OK) {
        return res;
    }

    buf = buffer;
    initialized = true;

    return res;
}

#if 1
void camera_sleep(uint32_t SleepTimeMiliSeconds)
{
    //camera->Control(CAMERA_SLEEP);

    tx_thread_sleep(SleepTimeMiliSeconds / (1000 / TX_TIMER_TICKS_PER_SECOND));

    //camera->Control(CAMERA_WAKE);    
}
#endif
void camera_start(uint32_t mode)
{
#if 1
    if (mode == CAMERA_MODE_SNAPSHOT) {
        camera->CaptureFrame(buf);
    } else {
        camera->CaptureVideo(buf);
    }
#endif        
}

int32_t camera_gain(uint32_t gain)
{
    return camera->Control(CPI_CAMERA_SENSOR_GAIN, gain);
}

int32_t camera_wait(uint32_t timeout_ms)
{
    (void) timeout_ms;
    ULONG actual_events = 0;

    tx_event_flags_get(&camera_event_flags, \
            CAM_CB_EVENT_CAPTURE_STOPPED      |   \
            CAM_CB_EVENT_ERROR,                   \
            TX_OR_CLEAR,                          \
            &actual_events,                       \
            timeout_ms / (1000 / TX_TIMER_TICKS_PER_SECOND));

    if ((!(actual_events & CAM_CB_EVENT_CAPTURE_STOPPED) && \
            (actual_events & CAM_CB_EVENT_ERROR)) || (!actual_events)) {
        return ARM_DRIVER_ERROR;
    }

    return ARM_DRIVER_OK;
}
