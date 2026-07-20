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
 * @file     demo_usbx_device_cdc_acm_app.c
 * @author   Anil
 * @email    anil.appana@alifsemi.com
 * @version  V1.0.0
 * @date     31-May-2022
 * @brief    This is CDC ACM USB read and write.
 * @Note     None.
 *******************************************************************************/
#include "ux_api.h"
#include "ux_system.h"
#include "ux_utility.h"
#include "ux_dcd_dwc3.h"
#include "ux_device_stack.h"
#include "ux_device_class_cdc_acm.h"
#include "system_utils.h"
// #include "backup4k_sram.h"
#include "Driver_Common.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "se_services_port.h"
#include "Driver_GPIO.h"
#include "Driver_LPTIMER.h"
#include "aipm.h"
#include "pm.h"
#if defined(RTE_Compiler_IO_STDOUT)
#include "retarget_stdout.h"
#endif

// #include "board.h"
// #include "testLED.h"
// #include "image_data.h"
// #include "image_processing.h"
// #include "jpeg.h"

#define DC_BOARD 1
#define POWER_MODE  2
//#define WITH_GOTO_STOP 1

/* Define constants.  */
#define ONE_KB                            1024
#define UX_DEMO_NS_SIZE                   (32*ONE_KB)
#define UX_DEMO_STACK_SIZE                (32*ONE_KB)

static ULONG  error_counter;
/* CDC-ACM reception data buffer. */
static UCHAR recv_buffer[512] = {0x00};

/* A pointer to store CDC-ACM device instance. */
static UX_SLAVE_CLASS_CDC_ACM* volatile g_cdc = UX_NULL;

/* Define local function prototypes.  */

VOID    demo_thread_entry(ULONG arg);
VOID    ux_cdc_device0_instance_activate(VOID *activated);
VOID    ux_cdc_device0_instance_acm_parameter(VOID *activated);
VOID    ux_cdc_device0_instance_deactivate(VOID *deactivated);
VOID    error_handler(void);

void stop_lptimer_in_start(void);
static int32_t lptimer_init(void);
static int set_lptimer(uint32_t  timeout);
static int lptimer_error_power_off(void);
static int lptimer_error_uninitialize(void);
static void lptimer_callback0(uint8_t event);
static uint32_t services_configure_stop_mode(void);
void pm_display_wakeup_reason(void);
static void execute_stop_mode_usecase(uint32_t mode_number);
int set_runp(void);

/* LPTIMER Driver instance 0 */
#define LPTIMER_CHANNEL    0

extern ARM_DRIVER_LPTIMER DRIVER_LPTIMER0;
static ARM_DRIVER_LPTIMER *lptimerDrv = &DRIVER_LPTIMER0;

TX_THREAD  demo_thread;
#ifdef DC_BOARD    
uint8_t dma_buf[UX_DEMO_NS_SIZE]__attribute__((section("usb_dma_buf")));
UX_SLAVE_CLASS_CDC_ACM_PARAMETER cdc_acm0_parameter;

UX_SLAVE_CLASS_CDC_ACM_LINE_CODING_PARAMETER CDC_ACM_LineCoding =
{
    9600, /* baud rate */
    0x00,   /* stop bits-1 */
    0x00,   /* parity - none */
    0x08    /* nb. of bits 8 */
};

UX_SLAVE_CLASS_CDC_ACM_LINE_STATE_PARAMETER CDC_ACM_LineState =
{
    1,
    1
};

/* Change the endpoint packet size later */
#define DEVICE_FRAMEWORK_LENGTH_FULL_SPEED 93
UCHAR device_framework_full_speed[] = {

    /*
    Device descriptor 18 bytes
    0x02 bDeviceClass: CDC class code
    0x00 bDeviceSubclass: CDC class sub code 0x00 bDeviceProtocol: CDC Device protocol
    idVendor & idProduct - https://www.linux-usb.org/usb.ids
    */

    0x12, 0x01, 0x00, 0x02,
    0x02, 0x00, 0x01, 0x40,
    0x25, 0x05, 0xa7, 0xa4,
    0x00, 0x01, 0x01, 0x02,
    0x03, 0x01,

    /* Configuration  descriptor 9 bytes  */
    0x09, 0x02, 0x4b, 0x00, 0x02, 0x01, 0x00, 0xC0, 0x01,

    /* Interface association descriptor  */
    0x08, 0x0b, 0x00, 0x02, 0x02, 0x02, 0x00, 0x00,

    /* Communication Class Interface Descriptor */
    0x09, 0x04, 0x00, 0x00, 0x01, 0x02, 0x02, 0x01, 0x00,

    /* Header Functional Descriptor 5 bytes */
    0x05, 0x24, 0x00, 0x10, 0x01,

    /* ACM Functional Descriptor */
    0x04, 0x24, 0x02, 0x0f,
    /* Union Functional Descriptor 5 bytes */
    0x05, 0x24, 0x06, 0x00, 0x01,

    /* Call Management Functional Descriptor */
    0x05, 0x24, 0x01, 0x03, 0x01, /* Data interface */

    /* High Speed Notify  descriptor 7 bytes */
    0x07, 0x05, 0x83, 0x03, 0x0a, 0x00, 0x09,

    /* Data Interface Descriptor  */
    0x09, 0x04, 0x01, 0x00, 0x02, 0x0A, 0x00, 0x00, 0x00,

    /* ACM HS IN Endpoint descriptor 7 bytes*/
    0x07, 0x05, 0x82, 0x02, 0x00, 0x02, 0x00,

    /* ACM HS OUT Endpoint descriptor 7 bytes */
    0x07, 0x05, 0x02, 0x02, 0x00, 0x02, 0x00

};

#define DEVICE_FRAMEWORK_LENGTH_HIGH_SPEED 103

UCHAR device_framework_high_speed[] = {

    /*
    Device descriptor 18 bytes
    0x02 bDeviceClass: CDC class code
    0x00 bDeviceSubclass: CDC class sub code 0x00 bDeviceProtocol: CDC Device protocol
    idVendor & idProduct - https://www.linux-usb.org/usb.ids
    */

    0x12, 0x01, 0x00, 0x02,
    0x02, 0x00, 0x01, 0x40,
    0x25, 0x05, 0xa7, 0xa4,
    0x00, 0x01, 0x01, 0x02,
    0x03, 0x01,

   // XXX added device qualifier descriptor
   0x0a, 0x06, 0x00, 0x02,0x02, 0x00, 0x00, 0x40, 0x01, 0x00,

    /* Configuration  descriptor 9 bytes  */
    0x09, 0x02, 0x4b, 0x00, 0x02, 0x01, 0x00, 0xC0, 0x01,

    /* Interface association descriptor */
    0x08, 0x0b, 0x00, 0x02, 0x02, 0x02, 0x00, 0x00,

    /* Communication Class Interface Descriptor */
    0x09, 0x04, 0x00, 0x00, 0x01, 0x02, 0x02, 0x01, 0x00,

    /* Header Functional Descriptor 5 bytes */
    0x05, 0x24, 0x00, 0x10, 0x01,

    /* ACM Functional Descriptor */
    0x04, 0x24, 0x02, 0x0f,

    /* Union Functional Descriptor 5 bytes */
    0x05, 0x24, 0x06, 0x00, 0x01,

    /* Call Management Functional Descriptor */
    0x05, 0x24, 0x01, 0x03, 0x01, /* Data interface */

    /* High Speed Notify  descriptor 7 bytes  */
    0x07, 0x05, 0x81, 0x03, 0x0a, 0x00, 0x09,

    /* Data Interface Descriptor Requirement  */
    0x09, 0x04, 0x01, 0x00, 0x02, 0x0A, 0x00, 0x00, 0x00,

    /* ACM HS IN Endpoint descriptor 7 bytes*/
    0x07, 0x05, 0x82, 0x02, 0x00, 0x02, 0x00,

    /* ACM HS OUT Endpoint descriptor 7 bytes */
    0x07, 0x05, 0x01, 0x02, 0x00, 0x02, 0x00
};

    /* String Device Framework :
     Byte 0 and 1 : Word containing the language ID : 0x0904 for US
     Byte 2       : Byte containing the index of the descriptor
     Byte 3       : Byte containing the length of the descriptor string
    */

#define STRING_FRAMEWORK_LENGTH 39
UCHAR string_framework[] = {

  (UCHAR) (0x0409), /* 0 Supported Language Code */
  (UCHAR) (0x0409 >> 8), /* 1 Supported Language Code */
  0x01, /* 2 Index */
  17, /* 3 bLength */
  'A', 'l', 'i', 'f', 'S', 'e', 'm', 'i', 'c', 'o', 'n', 'd', 'u', 'c', 't', 'o', 'r',

  (UCHAR) (0x0409), /* 0 Supported Language Code */
  (UCHAR) (0x0409 >> 8), /* 1 Supported Language Code */
  0x02, /* 2 Index */
  6, /* 3 bLength */
  'D', 'e', 'v', 'k', 'i', 't',

  (UCHAR) (0x0409), /* 0 Supported Language Code */
  (UCHAR) (0x0409 >> 8), /* 1 Supported Language Code */
  0x03, /* 2 Index */
  4, /* 3 bLength */
  '1', '2', '0', '0'
};

    /* Multiple languages are supported on the device, to add
       a language besides english, the unicode language code must
       be appended to the language_id_framework array and the length
     adjusted accordingly. */
#define LANGUAGE_ID_FRAMEWORK_LENGTH 2
UCHAR language_id_framework[] = {

    /* English. */
        0x09, 0x04
    };
#endif
void services_print_runp(void)
{
    uint32_t  service_error_code;
    uint32_t  error_code;
    run_profile_t runp = {0};


    error_code = SERVICES_get_run_cfg(se_services_s_handle, &runp, &service_error_code);
    while ((error_code != 0) || (service_error_code != 0));

    printf("RUN Power Profile:\r\n");
    printf("power_domains = 0x%lx\r\n", runp.power_domains);
    printf("dcdc_voltage = %lu\r\n", runp.dcdc_voltage);
    printf("dcdc_mode = %d\r\n", runp.dcdc_mode);
    printf("aon_clk_src = %d\r\n", runp.aon_clk_src);
    printf("run_clk_src = %d\r\n", runp.run_clk_src);
    printf("cpu_clk_freq = %d\r\n", runp.cpu_clk_freq);
    printf("scaled_clk_freq = %d\r\n", runp.scaled_clk_freq);
    printf("memory_blocks = 0x%lx\r\n", runp.memory_blocks);
    printf("ip_clock_gating = 0x%lx\r\n", runp.ip_clock_gating);
    printf("phy_pwr_gating = 0x%lx\r\n", runp.phy_pwr_gating);
    printf("vdd_ioflex_3V3 = %d\r\n", runp.vdd_ioflex_3V3);
}
    
// __attribute__((section("backup4ksram")));
// using this option fails when tring to program the software through SE 
//__attribute__((section("backup4ksram"))) volatile uint8_t FirstRunAfterPOR; //0x4902C000;
#define FirstRunAfterPOR    *(volatile uint32_t *)(0x4902C000)

int main()
{
    int x = 100;
    sys_busy_loop_init();
    while (x-- > 0) {
        sys_busy_loop_us(1000);
    }

    volatile uint32_t debug_loop = 0;
    while(debug_loop);

    uint32_t error_code = 0;
    uint32_t service_error_code = 0;

    //*(volatile uint32_t *)(0x49001000UL) = 0x38;  /* set GPIOx pin 0 high */
    //*(volatile uint32_t *)(0x49001004UL) = 0x38;  /* set all GPIOx pins as output */

#ifdef WITH_GOTO_STOP
    //backup4k_sram_init();
    if (NVIC_GetPendingIRQ(LPTIMER0_IRQ_IRQn)) 
    {
        FirstRunAfterPOR = 0;
        stop_lptimer_in_start();
    }
    else
    {
        FirstRunAfterPOR = 1;
    }
#endif    

    //if (FirstRunAfterPOR)
    // BOARD_Pinmux_Init();

    /* Initialize the SE services */
    se_services_port_init();
    set_runp();

#ifdef DC_BOARD    
    error_code = SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_USB, true, &service_error_code);
    while ((error_code != 0) || (service_error_code != 0));
#endif
    error_code = SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_CLK_100M, true, &service_error_code);
    while ((error_code != 0) || (service_error_code != 0));

    error_code = SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_HFOSC, true, &service_error_code);
    while ((error_code != 0) || (service_error_code != 0));

#if defined(RTE_Compiler_IO_STDOUT_User)
    int32_t ret = stdout_init();
    while (ret != 0);
#endif

    tx_kernel_enter();
    return 0;
}

VOID tx_application_define(VOID *first_unused_memory)
{
    UINT status;
#ifdef DC_BOARD
    /* Initialize USBX Memory */
    status = ux_system_initialize(dma_buf, UX_DEMO_NS_SIZE, UX_NULL, 0x00);
    if (status != UX_SUCCESS) error_handler();

    /* The code below is required for installing the device portion of USBX.
       In this demo, DFU is possible and we have a call back for state change. */
    status = ux_device_stack_initialize(device_framework_high_speed, DEVICE_FRAMEWORK_LENGTH_HIGH_SPEED,
                                        device_framework_full_speed, DEVICE_FRAMEWORK_LENGTH_FULL_SPEED,
                                        string_framework, STRING_FRAMEWORK_LENGTH,
                                        language_id_framework, LANGUAGE_ID_FRAMEWORK_LENGTH,UX_NULL);
    if (status != UX_SUCCESS) error_handler();

    /* Setting cdc acm activation and deavtivation functionality */
    cdc_acm0_parameter.ux_slave_class_cdc_acm_instance_activate = ux_cdc_device0_instance_activate;

    cdc_acm0_parameter.ux_slave_class_cdc_acm_instance_deactivate = ux_cdc_device0_instance_deactivate;

    cdc_acm0_parameter.ux_slave_class_cdc_acm_parameter_change = ux_cdc_device0_instance_acm_parameter;

    /* Register the class driver as CDC-ACM  */
    status = ux_device_stack_class_register(_ux_system_slave_class_cdc_acm_name, ux_device_class_cdc_acm_entry, 1, 0, (VOID *) &cdc_acm0_parameter);
    if (status != UX_SUCCESS) error_handler();

    status =  _ux_dcd_dwc3_initialize();
    if (status != UX_SUCCESS) error_handler();
#endif

    /* Create the main demo thread.  */
    status = tx_thread_create(&demo_thread, "USBx demo", demo_thread_entry, 0, first_unused_memory, UX_DEMO_STACK_SIZE, 20, 20, 100, TX_AUTO_START);
    if (status != UX_SUCCESS) error_handler();
}

void GoToStopPowerMode()
{
    services_configure_stop_mode();

    if ((lptimer_init() != ARM_DRIVER_OK) ||
            (set_lptimer(5) != ARM_DRIVER_OK)) {
        printf("Failed to initialize low power timer\n");
    }

    execute_stop_mode_usecase(POWER_MODE);
}

VOID demo_thread_entry(ULONG arg)
{
    ULONG actual_length = 0;
    UINT status;

    while (1) {
        if(g_cdc != UX_NULL) {
            status = ux_device_class_cdc_acm_read(g_cdc, recv_buffer, sizeof(recv_buffer), (ULONG *) &actual_length);
            if((status != UX_SUCCESS)&&(status != UX_TRANSFER_NO_ANSWER)) {
                error_handler();
            }

            status = ux_device_class_cdc_acm_write(g_cdc, recv_buffer, actual_length, (ULONG *) &actual_length);
            if((status != UX_SUCCESS) && (status != UX_TRANSFER_NO_ANSWER)) {
                error_handler();
            }

            tx_thread_sleep(1);  /* give time for USB to transfer */

            /* if the character '0' is received on USB ACM then the M55 HP will shut down */
            if ('0' == recv_buffer[0]) {
				/* un-initialize the USB stack */
				ux_device_stack_class_unregister(_ux_system_slave_class_cdc_acm_name, ux_device_class_cdc_acm_entry);
				ux_device_stack_uninitialize();
				ux_system_uninitialize();

				GoToStopPowerMode();
            }
        }
    }
}

/// @brief 
/// @param  
/// @return 
VOID  ux_cdc_device0_instance_activate(VOID *activated)
{
    /* Save the CDC instance.  */
    g_cdc = (UX_SLAVE_CLASS_CDC_ACM *) activated;

}

/// @brief 
/// @param  
/// @return 
VOID  ux_cdc_device0_instance_deactivate(VOID *deactivated)
{
    g_cdc = UX_NULL;
}

/// @brief 
/// @param  
/// @return 
VOID  ux_cdc_device0_instance_acm_parameter(VOID *activated)
{
    /* USER CODE BEGIN USBD_CDC_ACM_ParameterChange */
    UX_PARAMETER_NOT_USED(activated);
    ULONG request;
    UX_SLAVE_TRANSFER *transfer_request;
    UX_SLAVE_DEVICE *device;

    /* Get the pointer to the device */
    device = &_ux_system_slave -> ux_system_slave_device;

    /* Get the pointer to the transfer request associated with the control endpoint */
    transfer_request = &device -> ux_slave_device_control_endpoint.ux_slave_endpoint_transfer_request;

    request = *(transfer_request -> ux_slave_transfer_request_setup + UX_SETUP_REQUEST);

    switch (request)
    {
        case UX_SLAVE_CLASS_CDC_ACM_SET_LINE_CODING :
            /* Get the Line Coding parameters */
            if (ux_device_class_cdc_acm_ioctl(g_cdc, UX_SLAVE_CLASS_CDC_ACM_IOCTL_SET_LINE_CODING,
                                            &CDC_ACM_LineCoding) != UX_SUCCESS) {
                printf("Set line coding error\n");
            }
            break;

        case UX_SLAVE_CLASS_CDC_ACM_GET_LINE_CODING :
            /* Set the Line Coding parameters */
            if (ux_device_class_cdc_acm_ioctl(g_cdc, UX_SLAVE_CLASS_CDC_ACM_IOCTL_GET_LINE_CODING,
                                            &CDC_ACM_LineCoding) != UX_SUCCESS) {
                printf("Get line coding failed\n");
            }
            break;

        case UX_SLAVE_CLASS_CDC_ACM_SET_CONTROL_LINE_STATE :
            if (ux_device_class_cdc_acm_ioctl(g_cdc, UX_SLAVE_CLASS_CDC_ACM_IOCTL_SET_LINE_STATE,
                                            &CDC_ACM_LineState) != UX_SUCCESS) {
                printf("Set line state failed\n");
            }
            break;

        default :
            break;
    }
    /* USER CODE END USBD_CDC_ACM_ParameterChange */
    return;
}

/// @brief 
/// @param  
/// @return 
VOID  error_handler(void)
{
    /* Increment error counter.  */
    error_counter++;
    printf("ERR : Initialization of USB failed with error counter :%ld\n",error_counter);
    while(1)
    {
        /* Error - just spin here!  Look at call tree in debugger
           to see where the error occurred.  */
    }
}

/// @brief 
/// @param  
/// @return 
int set_runp()
{
    uint32_t  service_error_code;
    uint32_t  error_code;
    run_profile_t runp = {0};

    runp.dcdc_mode = DCDC_MODE_PWM;
    runp.dcdc_voltage = 825;
    runp.vdd_ioflex_3V3 = IOFLEX_LEVEL_1V8;
    runp.aon_clk_src = CLK_SRC_LFXO;
    runp.run_clk_src = CLK_SRC_PLL;
    runp.power_domains = PD_SYST_MASK | PD_DBSS_MASK;
    runp.phy_pwr_gating =  USB_PHY_MASK;// | MIPI_PLL_DPHY_MASK | MIPI_TX_DPHY_MASK | MIPI_RX_DPHY_MASK | LDO_PHY_MASK;
    runp.memory_blocks = BACKUP4K_MASK | MRAM_MASK | SERAM_MASK | SRAM0_MASK | SRAM1_MASK;
#if defined(M55_HE)
    runp.cpu_clk_freq = CLOCK_FREQUENCY_160MHZ;
#else
    runp.cpu_clk_freq = CLOCK_FREQUENCY_400MHZ;
#endif

    error_code = SERVICES_set_run_cfg(se_services_s_handle, &runp, &service_error_code);
    while ((error_code != 0) || (service_error_code != 0));

    return 0;
}

/// @brief 
/// @param  
/// @return 
void pm_display_wakeup_reason(void)
{
    if (NVIC_GetPendingIRQ(LPTIMER0_IRQ_IRQn)) {
        printf("\r\nWakeup Reason (LPTIMER) Interrupt!\r\n");
    }
}

/// @brief 
/// @param  
/// @return 
static void execute_stop_mode_usecase(uint32_t mode_number)
{
    (void)mode_number;

    /* STOP_1 through STOP_3 will have BOD enabled */
    if (mode_number < 4) {
        ANA_REG->VBAT_ANA_REG3 |= 1U << 8;
    } else {
        ANA_REG->VBAT_ANA_REG3 &= ~(1U << 8);
    }

    /* STOP_5 will have RTC and LFXO disabled */
    if (mode_number < 5) {
        VBAT->RTC_CLK_EN = 1;
    } else {
        VBAT->RTC_CLK_EN = 0;
        ANA_REG->MISC_CTRL &= ~1U;
        ANA_REG->VBAT_ANA_REG1 &= ~(1U << 12);
    }

    for (int i = 0; i < (32 * 15); ++i) {
        if (NVIC_GetPendingIRQ(i) &&
                ((i < HWSEM_IRQ0_IRQn) || (i > HWSEM_IRQ15_IRQn)) &&
                ((i < PPU0_IRQ_IRQn) || (i > PPU2_IRQ_IRQn)) &&
                ((i <= CAM_IRQ_IRQn) || (i >= QEC0_CMPA_IRQ_IRQn))) {
            printf("FIXME:Pending interrupt: %d\r\n", i);
            NVIC_ClearPendingIRQ(i);
        }
        if (NVIC_GetActive(i)) {
            printf("FIXME:Active interrupt: %d\r\n", i);
            NVIC_DisableIRQ(i);
        }
    }

    // tx_interrupt_control(TX_INT_DISABLE);

    // radio_deinit();
    // tracelib_deinit();
    // extern void camera_deinit(void);
    // camera_deinit();
    // arm_ethosu_npu_deinit();
    // backup4k_sram_recalc_crc();

    __disable_irq();
    while(1) pm_core_enter_deep_sleep_request_subsys_off();
    __enable_irq();

    printf("THIS SHOULD NEVER HAPPEN! Device goes to STOP MODE\n");

    /* Should never get here unless an interrupt happened before we went to stop */
    pm_display_wakeup_reason();
    while(1);
}

/// @brief 
/// @param  
/// @return 
static uint32_t services_configure_stop_mode(void)
{
    uint32_t        service_error_code;
    uint32_t        error_code;
    off_profile_t   offp = {0};

    offp.aon_clk_src   = CLK_SRC_LFXO;
    offp.ewic_cfg      = EWIC_VBAT_TIMER/* | EWIC_VBAT_GPIO*/;
    offp.wakeup_events = WE_LPTIMER0/* | WE_LPGPIO2*/;
    offp.vdd_ioflex_3V3 = IOFLEX_LEVEL_1V8;
    offp.vtor_address  = SCB->VTOR;
    offp.memory_blocks = BACKUP4K_MASK | MRAM_MASK | SERAM_MASK | SRAM0_MASK | SRAM1_MASK;

    error_code = SERVICES_set_off_cfg(se_services_s_handle,
                                &offp,
                                &service_error_code);
    //printf("SE: set_off_cfg = %s\n",
    //        (error_code | service_error_code) == SERVICES_REQ_SUCCESS ? "ok" : "failed");

    return (error_code | service_error_code);
}

/// @brief 
/// @param  
/// @return 
static void lptimer_callback0(uint8_t event)
{
    if (event == ARM_LPTIMER_EVENT_UNDERFLOW) {
        printf("LPTimer: ARM_LPTIMER_EVENT_UNDERFLOW\n");
    }
}

/// @brief 
/// @param  
/// @return 
static int lptimer_error_uninitialize(void)
{
    int32_t ret = -1;

    /* Un-initialize lptimer driver */
    ret = lptimerDrv->Uninitialize(LPTIMER_CHANNEL);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: LPTIMER Uninitialize failed.\r\n");
    }
    return ret;
}

/// @brief 
/// @param  
/// @return 
static int lptimer_error_power_off(void)
{
    int32_t ret = -1;

    /* Power off LPTIMER peripheral */
    ret = lptimerDrv->PowerControl(LPTIMER_CHANNEL, ARM_POWER_OFF);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: LPTIMER Power OFF failed.\r\n");
        return ret;
    }

    ret = lptimer_error_uninitialize();

    return ret;
}

/// @brief 
/// @param  
/// @return 
static int set_lptimer(uint32_t  timeout)
{
    int32_t   ret;
    uint32_t  count = 0;

    /*
     *Configuring the lptimer channel for the timeout in seconds
     *Clock Source depends on RTE_LPTIMER_CHANNEL_CLK_SRC in RTE_Device.h
     *RTE_LPTIMER_CHANNEL_CLK_SRC = 0 : 32.768KHz freq (Default)
    */
    count = timeout * (32768);

    /**< Loading the counter value >*/
    ret = lptimerDrv->Control(0, ARM_LPTIMER_SET_COUNT1, &count);
    if(ret != ARM_DRIVER_OK)
    {
        printf("ERROR: channel '%d'failed to load count\r\n", LPTIMER_CHANNEL);
        lptimer_error_power_off();
        return ret;
    }

    //printf("\r\n Setting lptimer for %ld seconds \r\n", timeout);

    ret = lptimerDrv->Start(LPTIMER_CHANNEL);
    if(ret != ARM_DRIVER_OK)
    {
        printf("ERROR: failed to start channel '%d'\r\n", LPTIMER_CHANNEL);
        lptimer_error_uninitialize();
        return ret;
    }

    return ret;
}

/// @brief 
/// @param  
/// @return 
static int32_t lptimer_init(void)
{
    int32_t              ret = -1;

    /* Initialize LPTIMER driver */
    ret = lptimerDrv->Initialize(LPTIMER_CHANNEL, lptimer_callback0);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: LPTIMER init failed\r\n");
        return ret;
    }

    /* Enable the power for LPTIMER */
    ret = lptimerDrv->PowerControl(LPTIMER_CHANNEL, ARM_POWER_FULL);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: LPTIMER Power up failed\n");
        lptimer_error_uninitialize();
        return ret;
    }

    return ret;
}

/// @brief 
/// @param  
/// @return 
void stop_lptimer_in_start()
{
    lptimer_init();
    int32_t ret = lptimerDrv->Stop(LPTIMER_CHANNEL);
    if (ret != ARM_DRIVER_OK) {
        // uart not yet initialized
        while(1);
    }
    lptimer_error_power_off();
}
