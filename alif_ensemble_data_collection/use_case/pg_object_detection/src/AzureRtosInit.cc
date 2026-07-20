#include <stdlib.h>
#include <stdio.h>
#include "AzureRtosInit.h"

#ifndef TARGET_PLATFORM_NATIVE
#include "tx_api.h"
#include "log_macros.h"
#include "RadioComms.h"
#include "PGProto.h"

/* Define what the initial system looks like.  */
#define MAIN_THREAD_STACK_SIZE          (1024 * 20)
#define RADIO_COMMS_THREAD_STACK_SIZE   (1024 * 1)

extern char __APP_HEAP_SIZE;

static TX_THREAD main_thread_h;
static TX_THREAD comms_thread_h;
static TX_BYTE_POOL byte_pool_0;

extern void main_thread(ULONG thread_input);

int main()
{
    /* Enter the ThreadX kernel.  */
    tx_kernel_enter();

    return 0;
}

void my_stack_error_handler(TX_THREAD *thread_ptr)
{
    printf_err("Stack overflow thread name = %s!\r\n", thread_ptr->tx_thread_name);
}

void radio_comms_thread_resume(void)
{
    tx_thread_resume(&comms_thread_h);
}

void tx_application_define(void *first_unused_memory)
{
    UINT ret;
    CHAR *pointer = nullptr;

    ret = tx_thread_stack_error_notify(my_stack_error_handler);
    if (ret != TX_SUCCESS) {
        goto err;
    }

    /* Create a byte memory pool from which to allocate the thread stacks.  */
    ret = tx_byte_pool_create(&byte_pool_0, const_cast<char*>("byte pool 0"), first_unused_memory, (uint32_t)&__APP_HEAP_SIZE);
    if (ret != TX_SUCCESS) {
        goto err;
    }

    /* Allocate the stack for main thread.  */
    ret = tx_byte_allocate(&byte_pool_0, (VOID **) &pointer, MAIN_THREAD_STACK_SIZE, TX_NO_WAIT);
    if (ret != TX_SUCCESS) {
        goto err;
    }

    /* Create the main thread.  */
    ret = tx_thread_create (&main_thread_h, const_cast<char*>("Main Thread"), main_thread, 0,
            pointer, MAIN_THREAD_STACK_SIZE, 1, 1, TX_NO_TIME_SLICE, TX_AUTO_START);

    if (ret != TX_SUCCESS) {
        goto err;
    }

    /* Allocate the stack for the comms thread.  */
    ret = tx_byte_allocate(&byte_pool_0, (VOID **) &pointer,
        RADIO_COMMS_THREAD_STACK_SIZE, TX_NO_WAIT);
    if (ret != TX_SUCCESS) {
        goto err;
    }

    ret = tx_thread_create(&comms_thread_h, const_cast<char*>("Comms Thread"), radio_comms_thread, 1,
            pointer, RADIO_COMMS_THREAD_STACK_SIZE,
            2, 2, TX_NO_TIME_SLICE, TX_DONT_START);
    if (ret != TX_SUCCESS) {
        goto err;
    }

    radio_comms_init();
    pb_init();

    return;
err:
    for(;;) {
        /* Should never get here */

        //FIXMEAV: if we got here something fundamentally went wrong
        //What can we do here? Set an LED on? 
    }
}
#endif //TARGET_PLATFORM_NATIVE