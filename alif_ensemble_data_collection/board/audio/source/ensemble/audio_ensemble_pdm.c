/* System Includes */
#include <stdio.h>
#include <stdlib.h>
#include "tx_api.h"

/* Project Includes */
/* include for PDM Driver */
#include "Driver_PDM.h"
#include "pinconf.h"
#include "RTE_Components.h"
#include "log_macros.h"

/* PDM Channel 2 configurations */
#define CH2_PHASE             0x0000001F
#define CH2_GAIN              0x000000ff   /*D*/
#define CH2_PEAK_DETECT_TH    0x00060002
#define CH2_PEAK_DETECT_ITV   0x0004002D

/* PDM Channel 7 configurations */
#define CH7_PHASE             0x00000003
#define CH7_GAIN              0x000000ff  /*13*/
#define CH7_PEAK_DETECT_TH    0x00060002
#define CH7_PEAK_DETECT_ITV   0x00020027

/* PDM driver instance */
extern ARM_DRIVER_PDM Driver_PDM;
static ARM_DRIVER_PDM *PDMdrv = &Driver_PDM;

TX_EVENT_FLAGS_GROUP    event_flags_pdm;

typedef enum{
    PDM_CALLBACK_ERROR_EVENT           = (1 << 0),
    PDM_CALLBACK_CAPTURE_COMPLETE_EVENT= (1 << 1),
    PDM_CALLBACK_AUDIO_DETECTION_EVENT = (1 << 2)
}PDM_CB_EVENTS;

PDM_CH_CONFIG pdm_coef_reg;

uint32_t fir_coeff[] = {
						0x003, 0x014, 0x018, 0x7fa, 0x7ea, 0x00f,
						0x019, 0x7e3, 0x7e6, 0x033, 0x015, 0x7ad,
						0x7fd, 0x085, 0x7d1, 0x717, 0x0e6, 0x3db
					   };

void PDM_fifo_callback(uint32_t event)
{
    if (event & ARM_PDM_EVENT_ERROR) {
        tx_event_flags_set(&event_flags_pdm, PDM_CALLBACK_ERROR_EVENT, TX_OR);
    }

    if (event & ARM_PDM_EVENT_CAPTURE_COMPLETE) {
        tx_event_flags_set(&event_flags_pdm, PDM_CALLBACK_CAPTURE_COMPLETE_EVENT, TX_OR);
    }

    if (event & ARM_PDM_EVENT_AUDIO_DETECTION) {
        tx_event_flags_set(&event_flags_pdm, PDM_CALLBACK_AUDIO_DETECTION_EVENT, TX_OR);
    }
}

int audio_init_pdm(void)
{
    ARM_DRIVER_VERSION version;
    int32_t retval;

    version = PDMdrv->GetVersion();
    printf("PDM version api:%X driver:%X...\r\n", version.api, version.drv);

    retval = tx_event_flags_create(&event_flags_pdm,
            "audio pdm event flags");
    if (retval != TX_SUCCESS) {
        printf_err("Failed to create audio pdm event flags (%ld)\r\n", retval);
        return -1;
    }

    /* Initialize PDM driver */
    retval = PDMdrv->Initialize(PDM_fifo_callback);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM init failed\r\n");
        return -1;
    }

    /* Enable the power for PDM */
    retval = PDMdrv->PowerControl(ARM_POWER_FULL);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Power up failed\r\n");
        goto error_uninitialize;
    }

    /* To select the PDM channel 2 & 7 */
    retval = PDMdrv->Control(ARM_PDM_SELECT_CHANNEL,
            ARM_PDM_MASK_CHANNEL_2 |
            ARM_PDM_MASK_CHANNEL_7, 0);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM channel select control failed\r\n");
        goto error_poweroff;
    }

    /* Select Standard voice PDM mode */
    retval = PDMdrv->Control(ARM_PDM_MODE, ARM_PDM_MODE_STANDARD_VOICE_512_CLK_FRQ, 0);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Standard voice control mode failed\r\n");
        goto error_poweroff;
    }

    /* Select the DC blocking IIR filter */
    retval = PDMdrv->Control(ARM_PDM_BYPASS_IIR_FILTER, ENABLE, 0);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM DC blocking IIR control failed\r\n");
        goto error_poweroff;
    }

    /* Set Channel 2 Phase value */
    retval = PDMdrv->Control(ARM_PDM_CHANNEL_PHASE, 2, CH2_PHASE);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    /* Set Channel 2 Gain value */
    retval = PDMdrv->Control(ARM_PDM_CHANNEL_GAIN, 2, CH2_GAIN);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    /* Set Channel 2 Peak detect threshold value */
    retval = PDMdrv->Control(ARM_PDM_CHANNEL_PEAK_DETECT_TH, 2, CH2_PEAK_DETECT_TH);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    /* Set Channel 2 Peak detect ITV value */
    retval = PDMdrv->Control(ARM_PDM_CHANNEL_PEAK_DETECT_ITV, 2, CH2_PEAK_DETECT_ITV);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    pdm_coef_reg.ch_num = 2;
    memcpy(pdm_coef_reg.ch_fir_coef, fir_coeff, sizeof(pdm_coef_reg.ch_fir_coef)); /* Channel 2 fir coefficient */
    pdm_coef_reg.ch_iir_coef = 0x00000004;   /* Channel IIR Filter Coefficient */

    retval = PDMdrv->Config(&pdm_coef_reg);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    /* Set Channel 7 Phase value */
    retval = PDMdrv->Control(ARM_PDM_CHANNEL_PHASE, 7, CH7_PHASE);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    /* Set Channel 7 Gain value */
    retval = PDMdrv->Control(ARM_PDM_CHANNEL_GAIN, 7, CH7_GAIN);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    /* Set Channel 7 Peak detect threshold value */
    retval = PDMdrv->Control(ARM_PDM_CHANNEL_PEAK_DETECT_TH, 7, CH7_PEAK_DETECT_TH);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    /* Set Channel 7 Peak detect ITV value */
    retval = PDMdrv->Control(ARM_PDM_CHANNEL_PEAK_DETECT_ITV, 7, CH7_PEAK_DETECT_ITV);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    pdm_coef_reg.ch_num = 7;
    memcpy(pdm_coef_reg.ch_fir_coef, fir_coeff, sizeof(pdm_coef_reg.ch_fir_coef)); /* Channel 7 fir coefficient */
    pdm_coef_reg.ch_iir_coef = 0x00000004;   /* Channel IIR Filter Coefficient */

    retval = PDMdrv->Config(&pdm_coef_reg);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Channel_Config failed\r\n");
        goto error_uninitialize;
    }

    return 0;

error_poweroff:
    retval = PDMdrv->PowerControl(ARM_POWER_OFF);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM power off failed\r\n");
    }

error_uninitialize:
    retval = PDMdrv->Uninitialize();
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Uninitialize failed\r\n");
    }

    return -1;
}

int wait_for_audio_pdm(int16_t* sample_buf, uint32_t sample_num)
{
    int32_t retval;
    ULONG actual_events = 0;

    /* Receive the audio samples */
    retval = PDMdrv->Receive((int16_t*)sample_buf, sample_num);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Receive failed %ld\r\n", retval);
        return -1;
    }

    /* wait for the call back event */
    retval = tx_event_flags_get(&event_flags_pdm, \
            PDM_CALLBACK_ERROR_EVENT | PDM_CALLBACK_CAPTURE_COMPLETE_EVENT, \
            TX_OR_CLEAR,\
            &actual_events,\
            TX_WAIT_FOREVER);

    if (retval != TX_SUCCESS) {
        printf_err("Error: PDM tx_event_flags_get\r\n");
        return -1;
    }

    /* PDM channel audio detection event */
    if (actual_events & PDM_CALLBACK_AUDIO_DETECTION_EVENT) {
        printf("PDM audio detect event: data in the audio channel\r\n");
    }

    /* PDM fifo overflow error event */
    if (actual_events & PDM_CALLBACK_ERROR_EVENT) {
        printf_err("PDM error event: Fifo overflow\r\n");
    }

    /* PDM fifo alomost full warning event */
    if (actual_events & PDM_CALLBACK_CAPTURE_COMPLETE_EVENT) {
        printf("PDM event: capture complete\r\n");
        return 0;
    }

    return -1;
}

void audio_deinit_pdm(void)
{
    int32_t retval;

    retval = PDMdrv->PowerControl(ARM_POWER_OFF);
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM power off failed\r\n");
    }

    retval = PDMdrv->Uninitialize();
    if (retval != ARM_DRIVER_OK) {
        printf_err("Error: PDM Uninitialize failed\r\n");
    }

    tx_event_flags_delete(&event_flags_pdm);
}