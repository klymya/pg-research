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
#include "backup4k_sram.h"
#include "se_services_port.h"
#include "RTE_Components.h"
#if defined(RTE_Compiler_IO_STDOUT)
#include "Driver_Common.h"
#include "Driver_LPTIMER.h"
#include "Driver_GPIO.h"
#include "retarget_stdout.h"
#endif  /* RTE_Compiler_IO_STDOUT */
#include "pm.h"
#include "aipm.h"

#include "board.h"
#include "testLED.h"
#include "image_data.h"
#include "image_processing.h"
#include "jpeg.h"

//#define WITH_GOTO_STOP

/* Define constants.  */
#define ONE_KB                            1024
#define UX_DEMO_NS_SIZE                   (32*ONE_KB)
#define UX_DEMO_STACK_SIZE_32             (32*ONE_KB)

static ULONG  error_counter;
/* CDC-ACM reception data buffer. */
static UCHAR recv_buffer[512] = {0x00};

/* A pointer to store CDC-ACM device instance. */
static UX_SLAVE_CLASS_CDC_ACM* g_cdc = UX_NULL;

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


/* Define global data structures.   */
#define POWER_MODE  2

/* LPTIMER Driver instance 0 */
#define LPTIMER_CHANNEL    0

extern ARM_DRIVER_LPTIMER DRIVER_LPTIMER0;
static ARM_DRIVER_LPTIMER *lptimerDrv = &DRIVER_LPTIMER0;

TX_THREAD  demo_thread;
#ifdef DC_BOARD    
uint8_t dma_buf[UX_DEMO_NS_SIZE]__attribute__((section("usb_dma_buf")));
UX_SLAVE_CLASS_CDC_ACM_PARAMETER cdc_acm0_parameter;

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
    0x08, 0x0b, 0x00,
    0x02, 0x02, 0x02, 0x00, 0x00,

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

#define DEVICE_FRAMEWORK_LENGTH_HIGH_SPEED 93

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

    /* Configuration  descriptor 9 bytes  */
    0x09, 0x02, 0x4b, 0x00, 0x02, 0x01, 0x00, 0xC0, 0x01,

    /* Interface association descriptor */
    0x08, 0x0b, 0x00,
    0x02, 0x02, 0x02, 0x00, 0x00,

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
    0x07, 0x05, 0x83, 0x03, 0x0a, 0x00, 0x09,

    /* Data Interface Descriptor Requirement  */
    0x09, 0x04, 0x01, 0x00, 0x02, 0x0A, 0x00, 0x00, 0x00,

    /* ACM HS IN Endpoint descriptor 7 bytes*/
    0x07, 0x05, 0x82, 0x02, 0x00, 0x02, 0x00,

    /* ACM HS OUT Endpoint descriptor 7 bytes */
        0x07, 0x05, 0x02, 0x02, 0x00, 0x02, 0x00

};

    /* String Device Framework :
     Byte 0 and 1 : Word containing the language ID : 0x0904 for US
     Byte 2       : Byte containing the index of the descriptor
     Byte 3       : Byte containing the length of the descriptor string
    */

#define STRING_FRAMEWORK_LENGTH 42
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


    error_code = SERVICES_get_run_cfg(se_services_s_handle,
                                        &runp,
                                        &service_error_code);

    if (error_code != SERVICES_REQ_SUCCESS)
        return;

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
extern volatile uint8_t FirstRunAfterPOR; 

int  main(void)
{
    UINT    error_code = 0;
    UINT    service_error_code = 0;
    run_profile_t   runp = {0};

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
    BOARD_Pinmux_Init();

    /* Initialize the SE services */
    se_services_port_init();

#ifdef DC_BOARD    
    /* Example code to enable the CLKEN_USB clock */
    error_code = SERVICES_clocks_enable_clock(se_services_s_handle,
                                       /*clock_enable_t*/ CLKEN_USB,
                                               /*bool enable   */ true,
                                                          &service_error_code);
    if(error_code)
    {
        printf("SE: clk enable = %d\n", error_code);
        return 0;
    }
#endif
    error_code = SERVICES_clocks_enable_clock(se_services_s_handle,
        /*clock_enable_t*/ CLKEN_CLK_100M,
                /*bool enable   */ true,
                           &service_error_code);
    if(error_code)
    {
        printf("SE: clk enable = %d\n", error_code);
        return 0;
    }

    error_code = SERVICES_clocks_enable_clock(se_services_s_handle,
        /*clock_enable_t*/ CLKEN_HFOSC,
                /*bool enable   */ true,
                           &service_error_code);
    if(error_code)
    {
        printf("SE: clk enable = %d\n", error_code);
        return 0;
    }

    set_runp();

    #if defined(RTE_Compiler_IO_STDOUT_User)
    int32_t ret;
    ret = stdout_init();
    if(ret != ARM_DRIVER_OK)
    {
        while(1)
        {
        }
    }
    #endif

#ifndef WITH_GOTO_STOP    
    services_print_runp();

    /* Enter the ThreadX kernel.  */
    printf("Started USBx driver app\n");
#endif

    tx_kernel_enter();
    return(0);
}


void  tx_application_define(void *first_unused_memory)
{

   UINT    status;
#ifdef DC_BOARD
    /* Initialize USBX Memory */
   status = ux_system_initialize(dma_buf, UX_DEMO_NS_SIZE, UX_NULL, 0x00);

    if(status != UX_SUCCESS)
    {
       error_handler();
    }

    /* The code below is required for installing the device portion of USBX.
       In this demo, DFU is possible and we have a call back for state change. */
    status =  ux_device_stack_initialize(device_framework_high_speed, DEVICE_FRAMEWORK_LENGTH_HIGH_SPEED,
                                       device_framework_full_speed, DEVICE_FRAMEWORK_LENGTH_FULL_SPEED,
                                       string_framework, STRING_FRAMEWORK_LENGTH,
                                       language_id_framework, LANGUAGE_ID_FRAMEWORK_LENGTH,UX_NULL);

    if(status != UX_SUCCESS)
    {
       error_handler();
    }
    /* Setting cdc acm activation and deavtivation functionality */
    cdc_acm0_parameter.ux_slave_class_cdc_acm_instance_activate = ux_cdc_device0_instance_activate;

    cdc_acm0_parameter.ux_slave_class_cdc_acm_instance_deactivate = ux_cdc_device0_instance_deactivate;

    cdc_acm0_parameter.ux_slave_class_cdc_acm_parameter_change = ux_cdc_device0_instance_acm_parameter;

    /* Register the class driver as CDC-ACM  */
    status =  ux_device_stack_class_register(_ux_system_slave_class_cdc_acm_name, ux_device_class_cdc_acm_entry,
                                                1, 0, (VOID *)&cdc_acm0_parameter);
    if(status != UX_SUCCESS)
    {
        error_handler();
    }

    status =  _ux_dcd_dwc3_initialize();

    if(status != UX_SUCCESS)
    {
        error_handler();
    }
#endif
    /* Create the main demo thread.  */
    status = tx_thread_create(&demo_thread, "USBx demo", demo_thread_entry, 0,  first_unused_memory,UX_DEMO_STACK_SIZE_32,
            20, 20, 100, TX_AUTO_START);

    if(status != UX_SUCCESS)
    {
       error_handler();
    }
}

#ifdef WITH_GOTO_STOP

void GoToStopPowerMode()
{
    services_configure_stop_mode();

    if ((lptimer_init() != ARM_DRIVER_OK) ||
            (set_lptimer(5) != ARM_DRIVER_OK)) {
        printf("Failed to initialize low power timer\n");
        return 0;
    } 
    else
    {
        //printf("Low power timer initialized!\n");
    }

    execute_stop_mode_usecase(POWER_MODE);
}

VOID    demo_thread_entry(ULONG arg)
{
    ULONG actual_length = 0;
    UINT status;

    //BOARD_LEDG_Control(BOARD_LED_STATE_HIGH);

    int err = image_init();
    
    //BOARD_LEDG_Control(BOARD_LED_STATE_LOW);
    if (0 != err) 
    {
        while(1)  
        {
            tx_thread_sleep(1000u / (1000 / TX_TIMER_TICKS_PER_SECOND));
        }
    }

    const uint8_t* currImage = get_image_data(CIMAGE_X, CIMAGE_Y);        

#if DC_BOARD    
    do{
        status = ux_device_class_cdc_acm_read(g_cdc, recv_buffer,sizeof(recv_buffer), (ULONG *) &actual_length);
        if((status != UX_SUCCESS)&&(status != UX_TRANSFER_NO_ANSWER))
        {
            error_handler();
        }
    }while(actual_length == 0);

    status = ux_device_class_cdc_acm_write(g_cdc, currImage, 921776/*CIMAGE_X * CIMAGE_Y * RGB_BYTES*/, (ULONG *) &actual_length);
    if((status != UX_SUCCESS) && (status != UX_TRANSFER_NO_ANSWER))
    {
        error_handler();
    }

    tx_thread_sleep(2000u / (1000 / TX_TIMER_TICKS_PER_SECOND));

    /* un-initialize the USB stack */
    ux_device_stack_class_unregister(_ux_system_slave_class_cdc_acm_name, ux_device_class_cdc_acm_entry);
    ux_device_stack_uninitialize();
    ux_system_uninitialize();
#endif
    GoToStopPowerMode();
}

#else
/* HM0360 Camera xsleep GPIO port */
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(RTE_HM0360_CAMERA_SENSOR_XSLEEP_GPIO_PORT);
static ARM_DRIVER_GPIO *MyGPIO_Driver_CAM_XSLP = &ARM_Driver_GPIO_(RTE_HM0360_CAMERA_SENSOR_XSLEEP_GPIO_PORT);

void MY_Camera_Sleep()
{
    int32_t ret = 0;

    ret = MyGPIO_Driver_CAM_XSLP->SetValue(RTE_HM0360_CAMERA_SENSOR_XSLEEP_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
    if(ret != ARM_DRIVER_OK)
        return ret;
}

void MY_Camera_Wake()
{
    int32_t ret = 0;

    ret = MyGPIO_Driver_CAM_XSLP->SetValue(RTE_HM0360_CAMERA_SENSOR_XSLEEP_PIN_NO, GPIO_PIN_OUTPUT_STATE_HIGH);
    if(ret != ARM_DRIVER_OK)
        return ret;
}

const uint8_t* currImage;
VOID    demo_thread_entry(ULONG arg)
{
    ULONG actual_length = 0;
    UINT status;
#if 1
#if 1
    int err = image_init();
    if (0 != err) 
    {
        while(1)  
        {
            tx_thread_sleep(1000u / (1000 / TX_TIMER_TICKS_PER_SECOND));
        }
    }

    while(1)
    {
        //BOARD_LEDG_Control(BOARD_LED_STATE_HIGH);
        currImage = get_image_data(CIMAGE_X, CIMAGE_Y);   
        //BOARD_LEDG_Control(BOARD_LED_STATE_LOW);     

#if 1
        //BOARD_LEDG_Control(BOARD_LED_STATE_LOW);
#ifdef DC_BOARD
        do{
            status = ux_device_class_cdc_acm_read(g_cdc, recv_buffer,sizeof(recv_buffer), (ULONG *) &actual_length);
            if((status != UX_SUCCESS)&&(status != UX_TRANSFER_NO_ANSWER))
            {
                error_handler();
            }
        }while(actual_length == 0);

        status = ux_device_class_cdc_acm_write(g_cdc, currImage, 921776/*CIMAGE_X * CIMAGE_Y * RGB_BYTES*/, (ULONG *) &actual_length);
        if((status != UX_SUCCESS) && (status != UX_TRANSFER_NO_ANSWER))
        {
            error_handler();
        }
#if 1
        //MY_Camera_Sleep();
        //tx_thread_sleep(1000u / (1000 / TX_TIMER_TICKS_PER_SECOND));
        //MY_Camera_Wake();
#else        
        image_sleep(1000);
#endif        

#endif        
#endif            
    }

#else
    int err = image_init();
    if (0 == err) {
        printf("Image sensor initialized\n");
        const uint8_t* currImage = get_image_data(CIMAGE_X, CIMAGE_Y);

        do{
            status = ux_device_class_cdc_acm_read(g_cdc, recv_buffer,sizeof(recv_buffer), (ULONG *) &actual_length);
            if((status != UX_SUCCESS)&&(status != UX_TRANSFER_NO_ANSWER))
            {
                error_handler();
            }
        }while(actual_length == 0);

        status = ux_device_class_cdc_acm_write(g_cdc, currImage, 921776/*CIMAGE_X * CIMAGE_Y * RGB_BYTES*/, (ULONG *) &actual_length);
        if((status != UX_SUCCESS) && (status != UX_TRANSFER_NO_ANSWER))
        {
            error_handler();
        }
#if 0     
        if (currImage) {
            jpeg_send_through_uart(currImage, CIMAGE_X, CIMAGE_Y);
        }
#endif
    }
#endif    
#if 1        
    testLED(2);
#endif       
#else
    while (1)
    {
       if(g_cdc != UX_NULL)
       {

          status = ux_device_class_cdc_acm_read(g_cdc, recv_buffer,sizeof(recv_buffer), (ULONG *) &actual_length);
          if((status != UX_SUCCESS)&&(status != UX_TRANSFER_NO_ANSWER))
          {
              error_handler();
          }
          status = ux_device_class_cdc_acm_write(g_cdc, recv_buffer,actual_length, (ULONG *) &actual_length);
          if((status != UX_SUCCESS) && (status != UX_TRANSFER_NO_ANSWER))
          {
              error_handler();
          }
          status = ux_device_class_cdc_acm_write(g_cdc, recv_buffer,actual_length, (ULONG *) &actual_length);
          if((status != UX_SUCCESS) && (status != UX_TRANSFER_NO_ANSWER))
          {
              error_handler();
          }

       }
    }
#endif    
}
#endif
#ifdef DC_BOARD    
/// @brief 
/// @param  
/// @return 
VOID  ux_cdc_device0_instance_activate(VOID *activated)
{
     /* Save the CDC instance.  */
    g_cdc = (UX_SLAVE_CLASS_CDC_ACM *)activated;

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
    /* For now only notifications.  */
}
#endif
/// @brief 
/// @param  
/// @return 
VOID  error_handler(void)
{

    UINT    error_code = 0;
    UINT    service_error_code = 0;
    run_profile_t   runp = {0};
    /* Increment error counter.  */
    error_counter++;
    printf("ERR : Initialization of USB failed with error counter :%ld\n",error_counter);
    /* Example code to enable the CLKEN_USB clock */
    error_code = SERVICES_clocks_enable_clock(se_services_s_handle,
                                       /*clock_enable_t*/ CLKEN_USB,
                                               /*bool disable   */ false,
                                                         &service_error_code);
    if(error_code)
    {
        printf("SE: clk disable = %d\n", error_code);
    }

    /* Get the current run configuration from SE */
    error_code = SERVICES_get_run_cfg(se_services_s_handle, &runp,
                                              &service_error_code);
    if(error_code)
    {
         printf("SE: get_run_cfg error = %d\n", error_code);
    }
    runp.phy_pwr_gating &= ~USB_PHY_MASK;

    /* Set the current run configuration to SE */
    error_code = SERVICES_set_run_cfg(se_services_s_handle, &runp,
                                              &service_error_code);
    if(error_code)
    {
         printf("SE: set_run_cfg error = %d\n", error_code);
    }
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

    /* Get the current run configuration from SE */
    error_code = SERVICES_get_run_cfg(se_services_s_handle,
                                      &runp,
                                      &service_error_code);
    //printf("SE: get_run_cfg = %s\n",
    //        (error_code | service_error_code) == SERVICES_REQ_SUCCESS ? "ok" : "failed");

    runp.dcdc_mode = DCDC_MODE_PWM;
    runp.dcdc_voltage = 825;
    runp.vdd_ioflex_3V3       = IOFLEX_LEVEL_1V8;
    runp.ip_clock_gating = 0;
    runp.aon_clk_src = CLK_SRC_LFXO;
    runp.scaled_clk_freq = SCALED_FREQ_XO_HIGH_DIV_38_4_MHZ;
    runp.power_domains = PD_VBAT_AON_MASK | PD_SSE700_AON_MASK | PD_SYST_MASK | PD_SESS_MASK;
    runp.memory_blocks = MRAM_MASK | SRAM1_MASK | SRAM0_MASK | BACKUP4K_MASK | SERAM_MASK;

#if defined(_DEBUG)
    runp.power_domains |= PD_DBSS_MASK;
#endif
 
    #ifdef DC_BOARD
        runp.phy_pwr_gating |=  USB_PHY_MASK | MIPI_PLL_DPHY_MASK | MIPI_TX_DPHY_MASK | MIPI_RX_DPHY_MASK | LDO_PHY_MASK;
        runp.memory_blocks = SRAM2_MASK | SRAM3_MASK | SRAM1_MASK | SRAM0_MASK | MRAM_MASK;
    #else  //BATTERY_BOARD 
        runp.phy_pwr_gating |=  MIPI_PLL_DPHY_MASK | MIPI_TX_DPHY_MASK | MIPI_RX_DPHY_MASK | LDO_PHY_MASK;
        runp.memory_blocks = SRAM1_MASK | SRAM0_MASK | MRAM_MASK;
    #endif    

    runp.run_clk_src = CLK_SRC_PLL;
#if defined(M55_HE)
    runp.cpu_clk_freq = CLOCK_FREQUENCY_160MHZ;
#else
    runp.cpu_clk_freq = CLOCK_FREQUENCY_400MHZ;
#endif

    /* Set the new run configuration */
    error_code = SERVICES_set_run_cfg(se_services_s_handle,
                                      &runp,
                                      &service_error_code);
    //printf("SE: set_run_cfg = %s\n",
    //        (error_code | service_error_code) == SERVICES_REQ_SUCCESS ? "ok" : "failed");

    return (error_code | service_error_code);
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

#if defined(M55_HE)
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
#endif

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
    NVIC_DisableIRQ(SysTick_IRQn);
    NVIC_DisableIRQ(PendSV_IRQn);

    //Disable IRQs
    __disable_irq();
    SysTick->CTRL &= ~(SysTick_CTRL_ENABLE_Msk);

    pm_core_enter_deep_sleep_request_subsys_off();

    __enable_irq();
    printf("THIS SHOULD NEVER HAPPEN! Device goes to STOP MODE\n");

    //FIXMEAV: recover USART and threadx here so that we could print the wakeup reason below
    /* Should never get here unless an interrupt happened before we went to stop */
    pm_display_wakeup_reason();
    //FIXMEAV: reset board here, or recover to working state (prefer former)
    while (1) {};
}

/// @brief 
/// @param  
/// @return 
static uint32_t services_configure_stop_mode(void)
{
    uint32_t        service_error_code;
    uint32_t        error_code;
    off_profile_t   offp;

    /* Get the current off configuration from SE */
    error_code = SERVICES_get_off_cfg(se_services_s_handle,
                            &offp,
                            &service_error_code);
    //printf("SE: get_off_cfg = %s\n",
    //        (error_code | service_error_code) == SERVICES_REQ_SUCCESS ? "ok" : "failed");
    if ((error_code | service_error_code) != SERVICES_REQ_SUCCESS) {
        return (error_code | service_error_code);
    }

    offp.power_domains = 0;
    offp.aon_clk_src   = CLK_SRC_LFXO;
    offp.stby_clk_src  = CLK_SRC_HFXO;
    offp.stby_clk_freq = SCALED_FREQ_XO_LOW_DIV_38_4_MHZ;
    offp.ewic_cfg      = EWIC_VBAT_TIMER/* | EWIC_VBAT_GPIO*/;
    offp.wakeup_events = WE_LPTIMER0/* | WE_LPGPIO2*/;
    offp.vdd_ioflex_3V3 = IOFLEX_LEVEL_1V8;
    offp.dcdc_mode = DCDC_MODE_PWM;
    offp.dcdc_voltage = 825;
    offp.ip_clock_gating = 0;
    offp.phy_pwr_gating = 0;
    offp.vtor_address  = SCB->VTOR;
    offp.vtor_address_ns  = SCB->VTOR;
    /*FIXME: From ALIF support:  We have some issues with the current SE
     * release so please add MRAM_MASK to memory_blocks for off profile.
     */
    offp.memory_blocks = MRAM_MASK | SERAM_MASK | BACKUP4K_MASK | SRAM0_MASK | SRAM1_MASK | SRAM4_1_MASK | SRAM4_2_MASK | SRAM5_1_MASK | SRAM5_2_MASK;

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
