#pragma once

#include "tx_api.h"
#include "pg_wdtm.h"
#include <iostream>

namespace PgWDTM
{

/**
 * @brief WatchDog class. Sync between WDTM handshake signals is done 
 * with semaphores from IRQ and thread context
 * 
 */
class WatchDog
{
public:
    WatchDog();
    ~WatchDog();

    /**
     * @brief Feeds the WDTM.
     * 
     */
    void feed();

    /**
     * @brief Releases watchdog semaphore, called from IRQ.
     * 
     */
    static void smphrRelease();

    /**
     * @brief Acquires watchdog semaphore.
     * 
     */
    void smphrAcquire();

    /**
     * @brief Gets current count of WDTM counter.
     * 
     * @return uint32_t Current counter.
     */
    static uint32_t wdtmGetCount();

private:

    /**
     * @brief WDTM thread. Answers to WAKE signal.
     * 
     * @param param Class instance is passed to thread.
     */
    static void wdtm_thread(ULONG param);

    /**
     * @brief Callback event for GPIO IRQ.
     * 
     * @param event Event occur mask.
     */
    static void wdtm_callback(uint32_t event);

    TX_THREAD m_wdtm_thread;

    /**
     * @brief Semaphore for sync between IRQ and thread. 
     * Used in IRQ context so needs to be static, defined in cpp file.
     * 
     */
    static TX_SEMAPHORE m_wdtm_smphr; 

    static const uint32_t m_stack_size = 512;
    static const uint8_t m_thread_prio = 5; // Check this

    static uint32_t m_wdtm_count;

    UCHAR *m_stack;
};
}