/* This file was ported to work on Alif Semiconductor Ensemble family of devices. */

/* Copyright (C) 2022 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "platform_drivers.h"

#include "log_macros.h"     /* Logging functions */
#include "peripheral_memmap.h"
#include "fault_handler.h"
#include <string.h>         /* For strncpy */
#include <time.h>
#include <inttypes.h>
#include <stdatomic.h>

#include "RTE_Device.h"
#include "RTE_Components.h"
#include "Driver_HWSEM.h"
#include "board.h"
#include "uart_tracelib.h"
#include "services_lib_api.h"
#include "pm.h"
#include "services.h"
#include "backup4k_sram.h"
#include "uart_radio.h"
#include "SEGGER_RTT.h"

#include CMSIS_device_header

#define TRACELIB_UART_BAUDRATE 921600
#define RADIO_UART_BAUDRATE 115200

#if defined(ARM_NPU)
#include "ethosu_driver.h"
#include "ethosu_npu_init.h"

#if defined(ETHOS_U_BASE_ADDR)
    #if (ETHOS_U_NPU_BASE != ETHOS_U_BASE_ADDR)
        #error "NPU component configured with incorrect NPU base address."
    #endif /* (ETHOS_U_NPU_BASE != ETHOS_U_BASE_ADDR) */
#else
    #error "ETHOS_U_BASE_ADDR should have been defined by the NPU component."
#endif /* defined(ETHOS_U_BASE_ADDR) */
#endif /* ARM_NPU */

uint32_t tprof1, tprof2, tprof3, tprof4, tprof5;

/** Platform name */
static const char* s_platform_name = DESIGN_NAME;

#ifdef COPY_VECTORS

static VECTOR_TABLE_Type MyVectorTable[496] __attribute__((aligned (2048))) __attribute__((section (".bss.noinit.ram_vectors")));

static void copy_vtor_table_to_ram()
{
    if (SCB->VTOR == (uint32_t) MyVectorTable) {
        return;
    }
    memcpy(MyVectorTable, (const void *) SCB->VTOR, sizeof MyVectorTable);
    __DMB();
    // Set the new vector table into use.
    SCB->VTOR = (uint32_t) MyVectorTable;
    __DSB();
}
#endif

static void print_reset_reason(const PM_RESET_STATUS last_reset_reason)
{
    switch(last_reset_reason)
    {
    case PM_RESET_STATUS_POR_OR_SOC_OR_HOST_RESET:
        printf("\r\nReset Reason [%d] (caused by POR or Secure Enclave initiated reset)\r\n", last_reset_reason);
        break;
    case PM_RESET_STATUS_NSRST_RESET:
        printf("\r\nReset Reason [%d] (caused by NSRST pin being asserted)\r\n", last_reset_reason);
        break;
    case PM_RESET_STATUS_EXTERNAL_SYS_RESET:
        printf("\r\nReset Reason [%d] (caused by a reset request to the power domain)\r\n", last_reset_reason);
        break;
    default:
        printf("\r\nReset Reason [%d](UNKNOWN)\r\n", last_reset_reason);
        break;
    }
}

static void print_board_revision(void)
{
#ifdef BOARD_IS_PG_REVA_VARIANT
    printf("PointGrab Board Rev.%c\r\n", 'A' + (int)BOARD_get_revision());
#endif
}

int platform_init(void)
{
    PM_RESET_STATUS last_reset_reason;

    /* Get the last reason for the reboot */
    last_reset_reason = pm_get_subsystem_reset_status();

    /* Turn off PRIVDEFENA - only way to have address 0 unmapped */
    ARM_MPU_Enable(MPU_CTRL_HFNMIENA_Msk);

#ifdef COPY_VECTORS
    copy_vtor_table_to_ram();
#endif

    /* Forces retarget code to be included in build */
    extern void _clock_init(void);
    _clock_init();

    SEGGER_RTT_Init();

    extern ARM_DRIVER_HWSEM ARM_Driver_HWSEM_(0);
    ARM_DRIVER_HWSEM *HWSEMdrv = &ARM_Driver_HWSEM_(0);

    HWSEMdrv->Initialize(NULL);
    /* Only 1 core will do the pinmux */
#if LP_PERIPHERAL_BASE == 0x70000000 // Old core, so old CMSIS pack
    if (HWSEMdrv->Lock() == ARM_DRIVER_OK) {
#else
    if (HWSEMdrv->TryLock() == ARM_DRIVER_OK) {
#endif
        /* We're first to acquire the lock - we do it */
        BOARD_Power_Init();
        BOARD_Clock_Init();
        BOARD_Pinmux_Init();

        /* Lock a second time to raise the count to 2 - the signal that we've finished */
        HWSEMdrv->Lock();
    } else {
        /* Someone else got there first - they did it or are doing it. Wait until we see count 2 indicating they've finished */
        while (HWSEMdrv->GetCount() < 2);
    }

    HWSEMdrv->Uninitialize();

    services_init();

    tracelib_init(NULL, TRACELIB_UART_BAUDRATE);
    radio_init(RADIO_UART_BAUDRATE);
    fault_dump_enable(true);

    pm_display_wakeup_reason();
    print_reset_reason(last_reset_reason);
    print_board_revision();
    backup4k_sram_init();
    info("Processor internal clock: %" PRIu32 "Hz\n", GetSystemCoreClock());
    info("%s: complete\n", __FUNCTION__);

#if defined(ARM_NPU)

    int state;

    /* If Arm Ethos-U NPU is to be used, we initialise it here */
    if (0 != (state = arm_ethosu_npu_init())) {
        return state;
    }

#endif /* ARM_NPU */

    /* Print target design info */
    info("Target system design: %s\n", s_platform_name);

    return 0;
}

void platform_release(void)
{
    __disable_irq();
}

const char* platform_name(void)
{
    return s_platform_name;
}

uint64_t ethosu_address_remap(uint64_t address, int index)
{
    UNUSED(index);
    /* Double cast to avoid build warning about pointer/integer size mismatch */
    return LocalToGlobal((void *) (uint32_t) address);
}
typedef struct address_range
{
    uintptr_t base;
    uintptr_t limit;
} address_range_t;

const address_range_t no_need_to_invalidate_areas[] = {
    /* TCM is never cached */
    {
        .base = 0x00000000,
        .limit = 0x01FFFFFF,
    },
    {
        .base = 0x20000000,
        .limit = 0x21FFFFFF,
    },
    /* MRAM should never change while running */
    {
        .base = MRAM_BASE,
        .limit = MRAM_BASE + MRAM_SIZE - 1,
    }
};

static bool check_need_to_invalidate(const void *p, size_t bytes)
{
    uintptr_t base = (uintptr_t) p;
    if (bytes == 0) {
        return false;
    }
    uintptr_t limit = base + bytes - 1;
    for (unsigned int i = 0; i < sizeof no_need_to_invalidate_areas / sizeof no_need_to_invalidate_areas[0]; i++) {
        if (base >= no_need_to_invalidate_areas[i].base && limit <= no_need_to_invalidate_areas[i].limit) {
            return false;
        }
    }
    return true;
}

bool ethosu_area_needs_invalidate_dcache(const void *p, size_t bytes)
{
    /* API says null pointer can be passed */
    if (!p) {
        return true;
    }
    /* We know we have a cache and assume the cache is on */
    return check_need_to_invalidate(p, bytes);
}

bool ethosu_area_needs_flush_dcache(const void *p, size_t bytes)
{
    /* We're not using writeback for any Ethos areas */
    UNUSED(p);
    UNUSED(bytes);
    return false;
}

void MPU_Load_Regions(void)
{
    //FIXME: review this function
    /* This is a complete map - the startup code enables PRIVDEFENA that falls back
     * to the system default, but we will turn it off later.
     */
    static const ARM_MPU_Region_t mpu_table[] __STARTUP_RO_DATA_ATTRIBUTE = {
    { // ITCM (alias) at 01000000, SRAM0 at 02000000, SRAM1 at 08000000
    .RBAR = ARM_MPU_RBAR(0x01000000, ARM_MPU_SH_NON, 0, 1, 0),  // RW, NP, XA
    .RLAR = ARM_MPU_RLAR(0x0FFFFFFF, 2)
    },
#if LP_PERIPHERAL_BASE != 0x70000000
    { // SSE-700 Host Peripheral Region (1A000000)
    .RBAR = ARM_MPU_RBAR(0x1A000000, ARM_MPU_SH_NON, 0, 0, 1),  // RW, P, XN
    .RLAR = ARM_MPU_RLAR(0x1FFFFFFF, 0)
    },
#endif
    { // DTCM (20000000)
    .RBAR = ARM_MPU_RBAR(DTCM_BASE, ARM_MPU_SH_NON, 0, 1, 1),  // RW, NP, XN
    .RLAR = ARM_MPU_RLAR(DTCM_BASE + DTCM_SIZE - 1, 2)
    },
    { // Combined RAM (30000000) - firewall translation with some system packages
    .RBAR = ARM_MPU_RBAR(0x30000000, ARM_MPU_SH_NON, 0, 1, 0),  // RW, NP, XN
    .RLAR = ARM_MPU_RLAR(0x3067FFFF, 2)
    },
    { // General peripherals
    .RBAR = ARM_MPU_RBAR(0x40000000, ARM_MPU_SH_NON, 0, 0, 1),  // RW, P, XN
    .RLAR = ARM_MPU_RLAR(0x4FFFFFFF, 0)
    },
    { // Other core's DTCM
#if defined(M55_HE)
    .RBAR = ARM_MPU_RBAR(SRAM2_BASE, ARM_MPU_SH_OUTER, 0, 1, 0),  // RW, NP, XA
    .RLAR = ARM_MPU_RLAR(SRAM3_BASE + SRAM3_SIZE - 1, 1)  // HP TCM (SRAM2 + SRAM3)
#elif defined(M55_HP)
    .RBAR = ARM_MPU_RBAR(SRAM4_BASE, ARM_MPU_SH_OUTER, 0, 1, 0),  // RW, NP, XA
    .RLAR = ARM_MPU_RLAR(SRAM5_BASE + SRAM5_SIZE - 1, 1)  // HE TCM (SRAM4 + SRAM5)
#else
  #error device not specified!
#endif
    },
#if LP_PERIPHERAL_BASE == 0x70000000
    { // LP- Peripheral & PINMUX Regions (70000000)
    .RBAR = ARM_MPU_RBAR(0x70000000, ARM_MPU_SH_NON, 0, 0, 1),  // RW, P, XN
    .RLAR = ARM_MPU_RLAR(0x71FFFFFF, 0)
    },
#endif
    { // MRAM (80000000)
    .RBAR = ARM_MPU_RBAR(MRAM_BASE, ARM_MPU_SH_NON, 1, 1, 0),  // RO, NP, XA
    .RLAR = ARM_MPU_RLAR(MRAM_BASE + MRAM_SIZE - 1, 1)
    },
    { // System PPB
    .RBAR = ARM_MPU_RBAR(0xE0000000, ARM_MPU_SH_NON, 1, 0, 1),  // RW, P, XN
    .RLAR = ARM_MPU_RLAR(0xE00FFFFF, 7)
    },
    };

    /* Define the possible Attribute regions */
    ARM_MPU_SetMemAttr(0, ARM_MPU_ATTR(   /* Attr0, Device Memory */
                          ARM_MPU_ATTR_DEVICE,
                          ARM_MPU_ATTR_DEVICE_nGnRE));
    ARM_MPU_SetMemAttr(1, ARM_MPU_ATTR(   /* Attr1, Normal Memory, Write-Back, Read-Write-Allocate */
                          ARM_MPU_ATTR_MEMORY_(1,1,1,1),
                          ARM_MPU_ATTR_MEMORY_(1,1,1,1)));
    ARM_MPU_SetMemAttr(2, ARM_MPU_ATTR(   /* Attr2, Normal Memory, Transient, Write Through, Read Allocate */
                          ARM_MPU_ATTR_MEMORY_(0,0,1,0),
                          ARM_MPU_ATTR_MEMORY_(0,0,1,0)));
    ARM_MPU_SetMemAttr(7, ARM_MPU_ATTR(   /* Attr7, Device Memory nGnRnE*/
                          ARM_MPU_ATTR_DEVICE,
                          ARM_MPU_ATTR_DEVICE_nGnRnE));

    /* Load the regions from the table */
    ARM_MPU_Load(0, &mpu_table[0], sizeof(mpu_table)/sizeof(ARM_MPU_Region_t));
}
