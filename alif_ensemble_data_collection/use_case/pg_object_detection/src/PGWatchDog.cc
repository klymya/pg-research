#include "PGWatchDog.h"

namespace PgWDTM
{
    //Definition of semaphore
    TX_SEMAPHORE WatchDog::m_wdtm_smphr; 
    uint32_t WatchDog::m_wdtm_count = 0;
    
    WatchDog::WatchDog()
    {
        pg_tpl5010_init(wdtm_callback);
        m_stack = new UCHAR[m_stack_size];

        tx_semaphore_create(&m_wdtm_smphr, "wdtm_smph", 0);
        UINT txRet = tx_thread_create(&m_wdtm_thread, "wdtm_thread", wdtm_thread, 
                        (ULONG)this, m_stack, m_stack_size, m_thread_prio, 
                        1, TX_NO_TIME_SLICE, TX_AUTO_START);

        if (TX_SUCCESS != txRet)
        {
            std::cout << "Failed to create thread for WDTM\n" << std::endl;
        }
    }

    WatchDog::~WatchDog()
    {
        tx_thread_terminate(&m_wdtm_thread);
        tx_thread_delete(&m_wdtm_thread);
        tx_semaphore_delete(&m_wdtm_smphr);
        delete [] m_stack;
    }

    void WatchDog::smphrRelease()
    {
        tx_semaphore_put(&m_wdtm_smphr);
    }
    
    void WatchDog::smphrAcquire()
    {
        tx_semaphore_get(&m_wdtm_smphr, TX_WAIT_FOREVER);
    }

    void WatchDog::feed()
    {
        pg_tpl5010_wakecheck();
    }

    void WatchDog::wdtm_thread(ULONG param)
    {
        WatchDog *instance = reinterpret_cast<WatchDog*>(param);

        while(1)
        {
            instance->smphrAcquire();
            instance->feed(); 

// This is only used for testing purposes.
#ifdef WDTM_TEST
            if (m_wdtm_count >= UINT32_MAX)
            {
                m_wdtm_count = 0;
            }
            else
            {
                m_wdtm_count++;
            }
#endif
        }
    }

    void WatchDog::wdtm_callback(uint32_t event)
    {
        smphrRelease();
    }

    uint32_t WatchDog::wdtmGetCount()
    {
        return m_wdtm_count;
    }

}
