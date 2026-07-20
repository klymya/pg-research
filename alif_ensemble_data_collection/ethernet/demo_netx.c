/*
 * demo_netx.c
 *
 * This program demonstrates the azure-rtos netxduo stack and Alif's netxduo
 * Ethernet driver.
 *
 * The demo creates an UDP echo server and a TCP echo server listening on the
 * standard echo server port (7). Also, ICMP is enabled for the created NetXDuo
 * IP instance. So the system will also respond to incoming ping requests.
 *
 * Author   : Silesh C V <silesh@alifsemi.com>
 *
 * Copyright (C) 2021 ALIF SEMICONDUCTOR
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of ALIF SEMICONDUCTOR nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/**
  \file demo_netx.c
  \brief Implements a demo program using the NetxDuo network stack.
 */

/* Comment out the below line and set SERVER_IP_ADDR to use a static address */
#define DEMO_USES_DHCP

#include <inttypes.h>

#include   "tx_api.h"
#include   "nx_api.h"

#include <stdint.h>
#include "RTE_Components.h"
#include "Driver_Common.h"
#include "Driver_LPTIMER.h"

#if defined(RTE_Compiler_IO_STDOUT)
#include "retarget_stdout.h"
#endif  /* RTE_Compiler_IO_STDOUT */

#include "board.h"

#include "se_services_port.h"
#include "services_lib_protocol.h"
#include "services_lib_api.h"
#include "pm.h"
#include "aipm.h"

#include "uart_tracelib.h"

#ifdef DEMO_USES_DHCP
#include   "nxd_dhcp_client.h"
#endif
/* Driver include */
#include "nx_eth_mac.h"

#define     DEMO_STACK_SIZE     2048
#define     PACKET_SIZE         1536

/** \brief The pool size. Allocate a pool to support 24 packets */
#define     POOL_SIZE           ((sizeof(NX_PACKET) + PACKET_SIZE) * 24)

#ifdef DEMO_USES_DHCP
#define SERVER_IP_ADDR          IP_ADDRESS(0, 0, 0, 0)
#else
/** \brief static IP address used by the demo application */
#define SERVER_IP_ADDR          IP_ADDRESS(192, 168, 1, 11)
#endif

/** \brief The port number that the TCP echo server listens on */
#define TCP_SERVER_PORT         7
/** \brief The port number that the UDP echo server listens on */
#define UDP_SERVER_PORT         7

/* Define the ThreadX and NetX object control blocks...  */
TX_THREAD               dhcp_client_thread;
TX_THREAD               udp_server_thread;
TX_THREAD               tcp_server_thread;

NX_PACKET_POOL          pool_0;
NX_IP                   ip_0;

#ifdef DEMO_USES_DHCP
NX_DHCP                 dhcp_client;
#endif

/* \brief The pool buffer area. Allocated in the "eth_buf" section */
UCHAR                   pool_buffer[POOL_SIZE]__attribute__((section("eth_buf")));

NX_UDP_SOCKET           udp_socket;
NX_TCP_SOCKET           tcp_socket;

ULONG error_counter;

/* Define thread prototypes.  */
void dhcp_client_thread_entry(ULONG thread_input);
void udp_server_thread_entry(ULONG thread_input);
void tcp_server_thread_entry(ULONG thread_input);

/* Define TCP callbacks */
void tcp_server_connect_received(NX_TCP_SOCKET *server_socket, UINT port);
void tcp_server_disconnect_received(NX_TCP_SOCKET *server_socket);

#define POWER_MODE  2

#define LPTIMER_CHANNEL    0
/* LPTIMER Driver instance 0 */
extern ARM_DRIVER_LPTIMER DRIVER_LPTIMER0;
static ARM_DRIVER_LPTIMER *lptimerDrv = &DRIVER_LPTIMER0;

extern uint32_t se_services_s_handle;

static SERVICES_version_data_t se_version_data;
/* unique part of the EUI64 (5 bytes). The full EUI64 is Alif OID (0x785994) (3 bytes) + unique id (5 bytes). */
/* These can be used as a unique id of the device. */
static uint8_t se_alif_EUI64[5];
static char alif_eui64_str[sizeof(se_alif_EUI64) * 2 + 1];

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

void services_print_offp(void)
{
    uint32_t service_error_code;
    uint32_t error_code;
    off_profile_t offp;

    error_code = SERVICES_get_off_cfg(se_services_s_handle,
                            &offp,
                            &service_error_code);

    if (error_code != SERVICES_REQ_SUCCESS)
        return;

    printf("OFF Power Profile:\r\n");
    printf("power_domains = 0x%lx\r\n", offp.power_domains);
    printf("dcdc_voltage = %lu\r\n", offp.dcdc_voltage);
    printf("dcdc_mode = %d\r\n", offp.dcdc_mode);
    printf("aon_clk_src = %d\r\n", offp.aon_clk_src);
    printf("stby_clk_src = %d\r\n", offp.stby_clk_src);
    printf("stby_clk_freq = %d\r\n", offp.stby_clk_freq);
    printf("memory_blocks = 0x%lx\r\n", offp.memory_blocks);
    printf("ip_clock_gating = 0x%lx\r\n", offp.ip_clock_gating);
    printf("phy_pwr_gating = 0x%lx\r\n", offp.phy_pwr_gating);
    printf("vdd_ioflex_3V3 = %d\r\n", offp.vdd_ioflex_3V3);
    printf("wakeup_events = 0x%lx\r\n", offp.wakeup_events);
    printf("ewic_cfg = 0x%lx\r\n", offp.ewic_cfg);
    printf("vtor_address = 0x%lx\r\n", offp.vtor_address);
    printf("vtor_address_ns = 0x%lx\r\n", offp.vtor_address_ns);
}

int set_runp()
{
    uint32_t  service_error_code;
    uint32_t  error_code;
    run_profile_t runp = {0};

    /* Get the current run configuration from SE */
    error_code = SERVICES_get_run_cfg(se_services_s_handle,
                                      &runp,
                                      &service_error_code);
    printf("SE: get_run_cfg = %s\n",
            (error_code | service_error_code) == SERVICES_REQ_SUCCESS ? "ok" : "failed");

    runp.dcdc_mode = DCDC_MODE_PWM;
    runp.dcdc_voltage = 825;
    runp.vdd_ioflex_3V3       = IOFLEX_LEVEL_1V8;
    runp.ip_clock_gating = 0;
    runp.aon_clk_src = CLK_SRC_LFXO;
    runp.scaled_clk_freq = SCALED_FREQ_XO_HIGH_DIV_38_4_MHZ;
    runp.power_domains = PD_VBAT_AON_MASK | PD_SSE700_AON_MASK | PD_SYST_MASK | PD_SESS_MASK;
    runp.memory_blocks = MRAM_MASK | SRAM1_MASK | SRAM0_MASK | BACKUP4K_MASK | SERAM_MASK;

    runp.phy_pwr_gating = MIPI_PLL_DPHY_MASK | MIPI_TX_DPHY_MASK | MIPI_RX_DPHY_MASK | LDO_PHY_MASK;
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
    printf("SE: set_run_cfg = %s\n",
            (error_code | service_error_code) == SERVICES_REQ_SUCCESS ? "ok" : "failed");

    return (error_code | service_error_code);
}
/**
 * @function    void mipi_clocks_init(void)
  \brief        Initializes MIPI clocks
  \param[in]    none
  \return       int
  */
int services_mipi_clocks_init(void)
{
    uint32_t  service_error_code;
    uint32_t  error_code;

    /* Enable MIPI Clocks */
    error_code = SERVICES_clocks_enable_clock(
        se_services_s_handle, CLKEN_CLK_100M, true, &service_error_code);
    //printf("SE: MIPI 100MHz clock enable = %s\n", error_code == SERVICES_REQ_SUCCESS ? "ok" : "failed");
    if ((error_code | service_error_code) != SERVICES_REQ_SUCCESS) {
        return (error_code | service_error_code);
    }

    error_code = SERVICES_clocks_enable_clock(
        se_services_s_handle, CLKEN_HFOSC, true, &service_error_code);
    //printf("SE: MIPI 38.4Mhz(HFOSC) clock enable = %s\n", error_code == SERVICES_REQ_SUCCESS ? "ok" : "failed");

    return (error_code | service_error_code);
}

void pm_display_wakeup_reason(void)
{
    if (NVIC_GetPendingIRQ(LPTIMER0_IRQ_IRQn)) {
        printf("\r\nWakeup Reason (LPTIMER) Interrupt!\r\n");
    }
}

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

char /*const*/* services_alif_eui64_str_get(void)
{
    return alif_eui64_str;
}

SERVICES_version_data_t const* service_version_data_get(void)
{
    return &se_version_data;
}

void services_init(void)
{
    /* Initialize the SE services */
    se_services_port_init();
    uint32_t service_error_code;
    uint32_t error_code;

    service_error_code = SERVICES_system_get_device_data(se_services_s_handle,
                            &se_version_data, &error_code);
    service_error_code |= SERVICES_system_get_eui_extension(se_services_s_handle,
                            false, se_alif_EUI64, &error_code);
    for (unsigned int i = 0; i < sizeof(se_alif_EUI64); ++i) {
        sprintf(alif_eui64_str + i * 2, "%02X", se_alif_EUI64[i]);
    }
    if ((service_error_code != SERVICES_REQ_SUCCESS) ||
            services_mipi_clocks_init()) {
        //FIXMEAV: reset board here?
        while(1);
    }
}

static uint32_t services_configure_stop_mode(void)
{
    uint32_t        service_error_code;
    uint32_t        error_code;
    off_profile_t   offp;

    /* Get the current off configuration from SE */
    error_code = SERVICES_get_off_cfg(se_services_s_handle,
                            &offp,
                            &service_error_code);
    printf("SE: get_off_cfg = %s\n",
            (error_code | service_error_code) == SERVICES_REQ_SUCCESS ? "ok" : "failed");
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
    offp.memory_blocks = MRAM_MASK | SERAM_MASK | BACKUP4K_MASK;

    error_code = SERVICES_set_off_cfg(se_services_s_handle,
                                &offp,
                                &service_error_code);
    printf("SE: set_off_cfg = %s\n",
            (error_code | service_error_code) == SERVICES_REQ_SUCCESS ? "ok" : "failed");

    return (error_code | service_error_code);
}

void uart_callback(uint32_t event)
{
    (void)event;
}

static void lptimer_callback0(uint8_t event)
{
    if (event == ARM_LPTIMER_EVENT_UNDERFLOW) {
        printf("LPTimer: ARM_LPTIMER_EVENT_UNDERFLOW\n");
    }
}

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

    printf("\r\n Setting lptimer for %ld seconds \r\n", timeout);

    ret = lptimerDrv->Start(LPTIMER_CHANNEL);
    if(ret != ARM_DRIVER_OK)
    {
        printf("ERROR: failed to start channel '%d'\r\n", LPTIMER_CHANNEL);
        lptimer_error_uninitialize();
        return ret;
    }

    return ret;
}

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

void HAL_Delay(uint32_t Delay)
{
    volatile uint32_t DelayCounter = Delay;
    while(DelayCounter--);
}


int main()
{
    stop_lptimer_in_start();

    BOARD_Pinmux_Init();
#if 0
    BOARD_LEDR_Control(BOARD_LED_STATE_HIGH);
    HAL_Delay(10000000);
    BOARD_LEDR_Control(BOARD_LED_STATE_LOW);
    HAL_Delay(10000000);
    BOARD_LEDG_Control(BOARD_LED_STATE_HIGH);
    HAL_Delay(10000000);
    BOARD_LEDG_Control(BOARD_LED_STATE_LOW);
    HAL_Delay(10000000);
    BOARD_LEDB_Control(BOARD_LED_STATE_HIGH);
    HAL_Delay(10000000);
    BOARD_LEDB_Control(BOARD_LED_STATE_LOW);    
#endif
    services_init();

    (void)set_runp();

    tracelib_init(NULL, uart_callback);

    services_configure_stop_mode();

    HAL_Delay(10000000);

    printf("Hello\n\r");
#if 0
    if ((lptimer_init() != ARM_DRIVER_OK) ||
            (set_lptimer(5) != ARM_DRIVER_OK)) {
        printf("Failed to initialize low power timer\n");
        return 0;
    } else {
        printf("Low power timer initialized!\n");
    }

    execute_stop_mode_usecase(POWER_MODE);
#endif
    /* Enter the ThreadX kernel.  */
    tx_kernel_enter();
}


/* Define what the initial system looks like.  */
void    tx_application_define(void *first_unused_memory)
{
CHAR *pointer;
UINT  status;

    /* Setup the working pointer.  */
    pointer =  (CHAR *)first_unused_memory;

    /* Create the dhcp client thread */
    tx_thread_create(&dhcp_client_thread, "DHCP client thread", dhcp_client_thread_entry, 0,
                     pointer, DEMO_STACK_SIZE,
                     4, 4, TX_NO_TIME_SLICE, TX_AUTO_START);

    pointer =  pointer + DEMO_STACK_SIZE;

    /* Create the main thread.  */
    tx_thread_create(&udp_server_thread, "UDP echo server thread", udp_server_thread_entry, 0,
                     pointer, DEMO_STACK_SIZE,
                     4, 4, TX_NO_TIME_SLICE, TX_DONT_START);

    pointer =  pointer + DEMO_STACK_SIZE;

    tx_thread_create(&tcp_server_thread, "TCP echo server thread", tcp_server_thread_entry, 0,
                     pointer, DEMO_STACK_SIZE,
                     3, 3, TX_NO_TIME_SLICE, TX_DONT_START);

    pointer =  pointer + DEMO_STACK_SIZE;

    /* Initialize the NetX system.  */
    nx_system_initialize();

    /* Create a packet pool.  */
    status =  nx_packet_pool_create(&pool_0, "NetX Main Packet Pool", PACKET_SIZE, pool_buffer, POOL_SIZE);

    if (status != NX_SUCCESS)
    {
        error_counter++;
    }

    /* Create an IP instance with static IP */
    status = nx_ip_create(&ip_0, "NetX IP Instance 0", SERVER_IP_ADDR, 0xFFFFFF00UL, &pool_0, nx_eth_driver,
                          pointer, 2048, 1);
    pointer =  pointer + 2048;

    if (status != NX_SUCCESS)
    {
        error_counter++;
    }

    /* Enable IP fragmentation on the IP instance. */
    status = nx_ip_fragment_enable(&ip_0);

    if (status != NX_SUCCESS)
    {
        error_counter++;
    }

    /* Enable ARP and supply ARP cache memory for the IP Instance */
    status =  nx_arp_enable(&ip_0, (void *)pointer, 1024);
    pointer = pointer + 1024;

    if (status != NX_SUCCESS)
    {
        error_counter++;
    }

    /* Enable ICMP traffic */
    status = nxd_icmp_enable(&ip_0);

    if (status != NX_SUCCESS)
    {
        error_counter++;
    }

    /* Enable UDP traffic */
    status =  nx_udp_enable(&ip_0);

    if (status != NX_SUCCESS)
    {
        error_counter++;
    }

    /* Enable TCP processing */
    status =  nx_tcp_enable(&ip_0);

    if (status != NX_SUCCESS)
    {
        error_counter++;
    }
}

/**
  \fn        void dhcp_client_thread_entry(ULONG thread_input)
  \brief     Thread function for the DHCP client thread.
  \param[in] thread_input. Unused.
*/
void dhcp_client_thread_entry(ULONG thread_input)
{
UINT status;
ULONG actual_status;
ULONG ip_address;
ULONG network_mask;

    BOARD_BUTTON_STATE state;

    NX_PARAMETER_NOT_USED(thread_input);

    /* Let the IP thread execute */
    tx_thread_sleep(NX_IP_PERIODIC_RATE);

    BOARD_LEDR_Control(BOARD_LED_STATE_HIGH);
    tx_thread_sleep(100);

    BOARD_LEDR_Control(BOARD_LED_STATE_LOW);
    tx_thread_sleep(100);

    BOARD_LEDG_Control(BOARD_LED_STATE_HIGH);
    tx_thread_sleep(100);

    BOARD_LEDG_Control(BOARD_LED_STATE_LOW);
    tx_thread_sleep(100);

    BOARD_LEDB_Control(BOARD_LED_STATE_HIGH);
    tx_thread_sleep(100);

    BOARD_LEDB_Control(BOARD_LED_STATE_LOW);    
    
#if 0
    __disable_interrupts();
    if ((lptimer_init() != ARM_DRIVER_OK) ||
            (set_lptimer(5) != ARM_DRIVER_OK)) {
        printf("Failed to initialize low power timer\n");
        return 0;
    } else {
        printf("Low power timer initialized!\n");
    }

    execute_stop_mode_usecase(POWER_MODE);
#else

    do
    {
        /* Get the link status. */
        status = nx_ip_status_check(&ip_0, NX_IP_LINK_ENABLED,
                    &actual_status, NX_IP_PERIODIC_RATE);
        printf("Waiting for Link up...\n");
    } while (status != NX_SUCCESS);

#ifdef DEMO_USES_DHCP
    status = nx_dhcp_create(&dhcp_client, &ip_0, "dhcp-client");

    if (status != NX_SUCCESS)
    {
        error_counter++;
        return;
    }

    /*
     * Instruct the DHCP client to use our default packet pool.
     * Note that for this to work, NX_DHCP_CLIENT_USER_CREATE_PACKET_POOL should
     * be defined in nxd_dhcp_client.h.
     */
    status = nx_dhcp_packet_pool_set(&dhcp_client, &pool_0);

    if (status != NX_SUCCESS)
    {
        error_counter++;
        return;
    }

    /* Start the DHCP Client.  */
    nx_dhcp_start(&dhcp_client);

    /* Wait for DHCP to assign the IP address. */
    do
    {
        /* Check for address resolution.  */
        status = nx_ip_status_check(&ip_0, NX_IP_ADDRESS_RESOLVED,
                    &actual_status, 10 * NX_IP_PERIODIC_RATE);
        printf("Waiting for IP address resolution...\n");

    } while (status != NX_SUCCESS);
#endif

    /*
     * We have successfully received an IP address (if DHCP is enabled),
     * retrieve and print this info.
     */
    status = nx_ip_address_get(&ip_0, &ip_address, &network_mask);

    if (status != NX_SUCCESS)
    {
        error_counter++;
        return;
    }

    printf("IP Address: %lu.%lu.%lu.%lu Network Mask: %lu.%lu.%lu.%lu\n",
                (ip_address >> 24) & 0xff, (ip_address >> 16) & 0xff,
                (ip_address >> 8) & 0xff, ip_address & 0xff,
                (network_mask >> 24) & 0xff, (network_mask >> 16) & 0xff,
                (network_mask >> 8) & 0xff, network_mask & 0xff);

    /* Wake up TCP/UDP server threads */
    tx_thread_resume(&udp_server_thread);
    tx_thread_resume(&tcp_server_thread);
#endif    
}

/**
  \fn        void udp_server_thread_entry(ULONG thread_input)
  \brief     Thread function for the UDP echo server thread.
  \param[in] thread_input. Unused.
*/
void udp_server_thread_entry(ULONG thread_input)
{
UINT status;
NX_PACKET *rx_packet;
ULONG ip_address;
UINT protocol;
UINT port;
UINT interface_index;
NXD_ADDRESS ipv4_address;

    NX_PARAMETER_NOT_USED(thread_input);

    /* Let the IP thread execute */
    tx_thread_sleep(NX_IP_PERIODIC_RATE);

    /* Create a UDP socket.  */
    status = nx_udp_socket_create(&ip_0, &udp_socket, "UDP Socket", NX_IP_NORMAL, NX_FRAGMENT_OKAY, 0x80, 5);

    if (status != NX_SUCCESS)
    {
        error_counter++;
        return;
    }

    /* Bind the UDP socket to the echo server port.  */
    status =  nx_udp_socket_bind(&udp_socket, UDP_SERVER_PORT, TX_WAIT_FOREVER);

    if (status != NX_SUCCESS)
    {
        error_counter++;
        return;
    }

    while (1)
    {
        /* Receive a UDP packet.  */
        status =  nx_udp_socket_receive(&udp_socket, &rx_packet, TX_WAIT_FOREVER);

        if (status != NX_SUCCESS)
        {
            error_counter++;
            break;
        }

        /* Extract client information from UDP packet interface.*/
        status = nx_udp_packet_info_extract(rx_packet, &ip_address,
                            &protocol, &port, &interface_index);

        if (status != NX_SUCCESS)
        {
            error_counter++;
            break;
        }

        /* Set up the client address */
        ipv4_address.nxd_ip_version = NX_IP_VERSION_V4;
        ipv4_address.nxd_ip_address.v4 = ip_address;

        /* Send back the received packet */
        status =  nxd_udp_socket_send(&udp_socket, rx_packet, &ipv4_address, port);

        if (status != NX_SUCCESS)
        {
            error_counter++;
            break;
        }
    }
}

/**
  \fn        void tcp_server_thread_entry(ULONG thread_input)
  \brief     Thread function for the TCP echo server thread.
  \param[in] thread_input. Unused.
*/
void tcp_server_thread_entry(ULONG thread_input)
{
UINT       status;
NX_PACKET *packet_ptr;
ULONG      actual_status;

    NX_PARAMETER_NOT_USED(thread_input);

    /* Wait 1 second for the IP thread to finish its initilization. */
    tx_thread_sleep(NX_IP_PERIODIC_RATE);

    /* Ensure the IP instance has been initialized.  */
    status =  nx_ip_status_check(&ip_0, NX_IP_INITIALIZE_DONE, &actual_status, NX_IP_PERIODIC_RATE);

    if (status != NX_SUCCESS)
    {
        error_counter++;
        return;
    }

    /* Create a TCP socket */
    status =  nx_tcp_socket_create(&ip_0, &tcp_socket, "Server Socket",
                                   NX_IP_NORMAL, NX_FRAGMENT_OKAY, NX_IP_TIME_TO_LIVE, 100,
                                   NX_NULL, tcp_server_disconnect_received);

    if (status != NX_SUCCESS)
    {
        error_counter++;
    }

    /* Setup this thread to listen.  */
    status =  nx_tcp_server_socket_listen(&ip_0, TCP_SERVER_PORT, &tcp_socket, 5, tcp_server_connect_received);

    if (status != NX_SUCCESS)
    {
        error_counter++;
    }

    /* Loop to create and establish connections.  */
    while (1)
    {
        /* Accept a client socket connection.  */
        status =  nx_tcp_server_socket_accept(&tcp_socket, NX_WAIT_FOREVER);

        if (status != NX_SUCCESS)
        {
            error_counter++;
            break;
        }

        while (1)
        {

            /* Receive a TCP message from the socket.  */
            status =  nx_tcp_socket_receive(&tcp_socket, &packet_ptr, NX_WAIT_FOREVER);

            if (status != NX_SUCCESS)
            {
                /* If status == NX_NOT_CONNECTED, that just means that the client has disconnected */
                if (status != NX_NOT_CONNECTED)
                {
                    error_counter++;
                }
                break;
            }

            /* Send the packet back */
            status =  nx_tcp_socket_send(&tcp_socket, packet_ptr, NX_IP_PERIODIC_RATE);

            if (status != NX_SUCCESS)
            {
                error_counter++;
                break;
            }
        }

        /* Disconnect the server socket.  */
        status =  nx_tcp_socket_disconnect(&tcp_socket, 10 * NX_IP_PERIODIC_RATE);

        if (status != NX_SUCCESS)
        {
            error_counter++;
            break;
        }

        /* Unaccept the server socket.  */
        status =  nx_tcp_server_socket_unaccept(&tcp_socket);

        if (status != NX_SUCCESS)
        {
            error_counter++;
            break;
        }

        /* Setup server socket for listening again.  */
        status =  nx_tcp_server_socket_relisten(&ip_0, TCP_SERVER_PORT, &tcp_socket);

        if (status != NX_SUCCESS)
        {
            error_counter++;
            break;
        }
    }
}

/**
  \fn        void tcp_server_connect_received(NX_TCP_SOCKET *socket_ptr, UINT port)
  \brief     TCP connect received callback function.
  \param[in] socket_ptr. The socket on which the connection request has been received.
  \param[in] port. The port on which the connection request has been received.
*/
void tcp_server_connect_received(NX_TCP_SOCKET *socket_ptr, UINT port)
{
    /* Check for proper socket and port.  */
    if ((socket_ptr != &tcp_socket) || (port != TCP_SERVER_PORT))
    {
        error_counter++;
    }
}

/**
  \fn        void tcp_server_disconnect_received(NX_TCP_SOCKET *socket)
  \brief     TCP disconnect received callback function.
  \param[in] socket_ptr. The socket on which the disconnect event has been received.
*/
void tcp_server_disconnect_received(NX_TCP_SOCKET *socket)
{
    /* Check for proper disconnected socket.  */
    if (socket != &tcp_socket)
    {
        error_counter++;
    }
}
