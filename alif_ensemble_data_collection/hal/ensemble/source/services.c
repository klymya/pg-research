/* Copyright (C) 2024 PointGrab - All Rights Reserved. */
/* Copyright (C) 2023 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/* System Includes */
#include <stdio.h>

/*RTE configuration includes */
#include <RTE_Device.h>
#include <RTE_Components.h>
#include CMSIS_device_header
#include "Driver_Common.h"

/* SE Services */
#include "services_main.h"
#include "services_lib_protocol.h"
#include "services_lib_api.h"
#include "services_lib_interface.h"
#include "pm.h"
#include "aipm.h"
#include "low_power.h"
#include "tx_api.h"
#include "uart_tracelib.h"
#include "uart_radio.h"
#include "log_macros.h"
#include "backup4k_sram.h"

#define POWER_MODE  2

static SERVICES_version_data_t se_version_data;
/* unique part of the EUI64 (5 bytes). The full EUI64 is Alif OID (0x785994) (3 bytes) + unique id (5 bytes). */
/* These can be used as a unique id of the device. */
static uint8_t se_alif_EUI64[5];
static char alif_eui64_str[sizeof(se_alif_EUI64) * 2 + 1];

inline int services_check_sync(void)
{
    if (se_services_get_handle() == SE_SERVICES_BAD_HANDLE) {
        printf("SE: not synchronized!\r\n");

        return -1;
    }

    return 0;
}

void services_print_runp(void)
{
    uint32_t  service_error_code;
    uint32_t  error_code;
    run_profile_t runp = {0};

    if (services_check_sync()) {
        return;
    }

    error_code = SERVICES_get_run_cfg(se_services_get_handle(),
                                      &runp,
                                      &service_error_code);

    if (error_code != SERVICES_REQ_SUCCESS)
        return;

    warn("RUN Power Profile:\r\n");
    warn("power_domains = 0x%lx\r\n", runp.power_domains);
    warn("dcdc_voltage = %lu\r\n", runp.dcdc_voltage);
    warn("dcdc_mode = %d\r\n", runp.dcdc_mode);
    warn("aon_clk_src = %d\r\n", runp.aon_clk_src);
    warn("run_clk_src = %d\r\n", runp.run_clk_src);
    warn("cpu_clk_freq = %d\r\n", runp.cpu_clk_freq);
    warn("scaled_clk_freq = %d\r\n", runp.scaled_clk_freq);
    warn("memory_blocks = 0x%lx\r\n", runp.memory_blocks);
    warn("ip_clock_gating = 0x%lx\r\n", runp.ip_clock_gating);
    warn("phy_pwr_gating = 0x%lx\r\n", runp.phy_pwr_gating);
    warn("vdd_ioflex_3V3 = %d\r\n", runp.vdd_ioflex_3V3);
}

void services_print_offp(void)
{
    uint32_t service_error_code;
    uint32_t error_code;
    off_profile_t offp;

    if (services_check_sync()) {
        return;
    }

    error_code = SERVICES_get_off_cfg(se_services_get_handle(),
                            &offp,
                            &service_error_code);

    if (error_code != SERVICES_REQ_SUCCESS)
        return;

    warn("OFF Power Profile:\r\n");
    warn("power_domains = 0x%lx\r\n", offp.power_domains);
    warn("dcdc_voltage = %lu\r\n", offp.dcdc_voltage);
    warn("dcdc_mode = %d\r\n", offp.dcdc_mode);
    warn("aon_clk_src = %d\r\n", offp.aon_clk_src);
    warn("stby_clk_src = %d\r\n", offp.stby_clk_src);
    warn("stby_clk_freq = %d\r\n", offp.stby_clk_freq);
    warn("memory_blocks = 0x%lx\r\n", offp.memory_blocks);
    warn("ip_clock_gating = 0x%lx\r\n", offp.ip_clock_gating);
    warn("phy_pwr_gating = 0x%lx\r\n", offp.phy_pwr_gating);
    warn("vdd_ioflex_3V3 = %d\r\n", offp.vdd_ioflex_3V3);
    warn("wakeup_events = 0x%lx\r\n", offp.wakeup_events);
    warn("ewic_cfg = 0x%lx\r\n", offp.ewic_cfg);
    warn("vtor_address = 0x%lx\r\n", offp.vtor_address);
    warn("vtor_address_ns = 0x%lx\r\n", offp.vtor_address_ns);
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
    run_profile_t runp = {0};

    if (services_check_sync()) {
        return -1;
    }

    /* Enable MIPI Clocks */
    error_code = SERVICES_clocks_enable_clock(
        se_services_get_handle(), CLKEN_CLK_100M, true, &service_error_code);
    printf("SE: MIPI 100MHz clock enable = %s\n",
            error_code == SERVICES_REQ_SUCCESS ? "ok" : "failed");
    if (error_code != SERVICES_REQ_SUCCESS) {
        return -1;
    }

    error_code = SERVICES_clocks_enable_clock(
        se_services_get_handle(), CLKEN_HFOSC, true, &service_error_code);
    printf("SE: MIPI 38.4Mhz(HFOSC) clock enable = %s\n",
            error_code == SERVICES_REQ_SUCCESS ? "ok" : "failed");
    if (error_code != SERVICES_REQ_SUCCESS) {
        goto error_disable_100mhz_clk;
    }

    /* Get the current run configuration from SE */
    error_code = SERVICES_get_run_cfg(se_services_get_handle(),
                                      &runp,
                                      &service_error_code);
    printf("SE: get_run_cfg = %s\n",
            error_code == SERVICES_REQ_SUCCESS ? "ok" : "failed");
    if (error_code) {
        goto error_disable_hfosc_clk;
    }

    runp.memory_blocks = MRAM_MASK | SRAM1_MASK | SRAM0_MASK;

    runp.phy_pwr_gating = MIPI_PLL_DPHY_MASK | MIPI_TX_DPHY_MASK | MIPI_RX_DPHY_MASK | LDO_PHY_MASK;
    runp.run_clk_src = CLK_SRC_PLL;
#if defined(M55_HE)
    runp.cpu_clk_freq = CLOCK_FREQUENCY_160MHZ;
#else
    runp.cpu_clk_freq = CLOCK_FREQUENCY_400MHZ;
#endif

    /* Set the new run configuration */
    error_code = SERVICES_set_run_cfg(se_services_get_handle(),
                                      &runp,
                                      &service_error_code);
    printf("SE: set_run_cfg = %s\n",
            error_code == SERVICES_REQ_SUCCESS ? "ok" : "failed");
    if (error_code) {
        goto error_disable_hfosc_clk;
    }

    return 0;

error_disable_hfosc_clk:
    error_code = SERVICES_clocks_enable_clock(
        se_services_get_handle(), CLKEN_HFOSC, false, &service_error_code);
    if(error_code != SERVICES_REQ_SUCCESS)
        printf("SE: MIPI 38.4Mhz(HFOSC)  clock disable = %lu\n", error_code);

error_disable_100mhz_clk:
    error_code = SERVICES_clocks_enable_clock(
        se_services_get_handle(), CLKEN_CLK_100M, false, &service_error_code);
    if(error_code != SERVICES_REQ_SUCCESS)
        printf("SE: MIPI 100MHz clock disable = %lu\n", error_code);

    return -1;
}

void pm_display_wakeup_reason(void)
{
    if (NVIC_GetPendingIRQ(LPTIMER0_IRQ_IRQn)) {
        printf("\r\nWakeup Reason (LPTIMER) Interrupt!\r\n");
    }
}

static void execute_stop_mode_usecase(uint32_t mode_number)
{
    UNUSED(mode_number);

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

    tx_interrupt_control(TX_INT_DISABLE);

    radio_deinit();
    tracelib_deinit();
    extern void camera_deinit(void);
    camera_deinit();
    arm_ethosu_npu_deinit();
    backup4k_sram_recalc_crc();
    NVIC_DisableIRQ(SysTick_IRQn);
    NVIC_DisableIRQ(PendSV_IRQn);

    //Disable IRQs
    __disable_irq();
    SysTick->CTRL &= ~(SysTick_CTRL_ENABLE_Msk);

    pm_core_enter_deep_sleep_request_subsys_off();

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
    if (se_services_port_init() != SE_SERVICES_BAD_HANDLE) {
        uint32_t service_error_code;
        uint32_t error_code;

        service_error_code = SERVICES_system_get_device_data(se_services_get_handle(),
                                &se_version_data, &error_code);
        service_error_code |= SERVICES_system_get_eui_extension(se_services_get_handle(),
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
}

static uint32_t services_configure_stop_mode(void)
{
    uint32_t        service_error_code;
    uint32_t        error_code;
    off_profile_t   offp;

    if (services_check_sync()) {
        return SERVICES_REQ_NOT_ACKNOWLEDGE;
    }

    /* Get the current off configuration from SE */
    error_code = SERVICES_get_off_cfg(se_services_get_handle(),
                            &offp,
                            &service_error_code);
    printf("SE: get_off_cfg = %s\n",
            error_code == SERVICES_REQ_SUCCESS ? "ok" : "failed");
    if (error_code != SERVICES_REQ_SUCCESS) {
        return error_code;
    }

    offp.power_domains = PD_VBAT_AON_MASK;
    offp.aon_clk_src   = CLK_SRC_LFXO;
    offp.stby_clk_src  = CLK_SRC_HFXO;
    offp.ewic_cfg      = EWIC_VBAT_TIMER/* | EWIC_VBAT_GPIO*/;
    offp.wakeup_events = WE_LPTIMER0/* | WE_LPGPIO2*/;
    offp.vdd_ioflex_3V3 = IOFLEX_LEVEL_1V8;
    //offp.vtor_address  = SCB->VTOR;
    /*FIXME: From ALIF support:  We have some issues with the current SE
     * release so please add MRAM_MASK to memory_blocks for off profile.
     */
    offp.memory_blocks = MRAM_MASK | SERAM_MASK | BACKUP4K_MASK;

    error_code = SERVICES_set_off_cfg(se_services_get_handle(),
                                &offp,
                                &service_error_code);
    printf("SE: set_off_cfg = %s\n",
            error_code == SERVICES_REQ_SUCCESS ? "ok" : "failed");

    return error_code;
}

void services_execute_aipm_stop_mode(const uint32_t wakeup_in_sec)
{
    services_configure_stop_mode();

    if ((lptimer_init() != ARM_DRIVER_OK) ||
            (set_lptimer(wakeup_in_sec) != ARM_DRIVER_OK)) {
        printf("Failed to initialize low power timer\n");
        return;
    } else {
        printf("Low power timer initialized!\n");
    }

    execute_stop_mode_usecase(POWER_MODE);
}