#include "tx_api.h"
#include "nx_api.h"
#include "log_macros.h"
#include "nx_eth_mac.h"
#include "hal.h"

#define PACKET_SIZE     1536
#define POOL_SIZE       ((sizeof(NX_PACKET) + PACKET_SIZE) * 16)
#define ARP_CACHE_SIZE  1024

static UCHAR pool_buffer[POOL_SIZE]__attribute__((section("eth_buf")));
static NX_PACKET_POOL pool_0;
static NX_IP ip_0;
static bool not_first_run;
static uint32_t dst_default_ip;

uint32_t ReadUserInputAsIPAddr(uint32_t default_addr)
{
    char chInput[32];
    uint32_t octet_1;
    uint32_t octet_2;
    uint32_t octet_3;
    uint32_t octet_4;

    memset(chInput, 0, sizeof(chInput));
    hal_get_user_input(chInput, sizeof(chInput));
    if (sscanf(chInput, "%u.%u.%u.%u", &octet_1, &octet_2, &octet_3, &octet_4) != 4) {
        return default_addr;
    }

    return IP_ADDRESS(static_cast<uint8_t>(octet_1),
                    static_cast<uint8_t>(octet_2),
                    static_cast<uint8_t>(octet_3),
                    static_cast<uint8_t>(octet_4));
}

int32_t test_ethernet(void)
{
    CHAR *ip_mem_pointer = NULL;
    CHAR *arp_cache_pointer = NULL;
    UINT  status;
    ULONG actual_status;
    /* Declare variable address to hold the destination address. */
    NXD_ADDRESS ip_address;
    NX_PACKET* response_ptr;

    if (!not_first_run) {
        uint32_t default_ip = IP_ADDRESS(192, 168, 0, 11);
        uint32_t default_mask = 0xffffff00UL;

        dst_default_ip = IP_ADDRESS(192, 168, 0, 2);
        not_first_run = true;

        printf("Set ALIF's IP addr (%u.%u.%u.%u): ", default_ip >> 24,
                                    (default_ip >> 16) & 0xff,
                                    (default_ip >> 8) & 0xff,
                                    default_ip & 0xff);
        fflush(stdout);
        default_ip = ReadUserInputAsIPAddr(default_ip);
        printf("Set ALIF's Net Mask (%u.%u.%u.%u): ", default_mask >> 24,
                                    (default_mask >> 16) & 0xff,
                                    (default_mask >> 8) & 0xff,
                                    default_mask & 0xff);
        fflush(stdout);
        default_mask = ReadUserInputAsIPAddr(default_mask);

        printf("Initializing Net system...\r\n");
        /* Initialize the NetX system.  */
        nx_system_initialize();

        /* Create a packet pool.  */
        status =  nx_packet_pool_create(&pool_0, "NetX Main Packet Pool", PACKET_SIZE, pool_buffer, POOL_SIZE);

        /* Check for pool creation error. */
        if (status) {
            printf_err("Failed to create packet poll (0x%x)\r\n", status);
            goto err2;
        }

        ip_mem_pointer = static_cast<CHAR*>(malloc(2048));
        /* Create an IP instance.  */
        status = nx_ip_create(&ip_0, "NetX IP Instance 0",
                            default_ip,
                            default_mask, &pool_0,
                            nx_eth_driver,
                            ip_mem_pointer, 2048, 1);
        if (status) {
            printf_err("Failed to create IP instance (0x%x)\r\n", status);
            goto err1;
        }

        /* Enable ARP and supply ARP cache memory for IP Instance 0.  */
        arp_cache_pointer = static_cast<CHAR*>(malloc(ARP_CACHE_SIZE));
        status =  nx_arp_enable(&ip_0, (void *)arp_cache_pointer, ARP_CACHE_SIZE);
        if (status) {
            printf_err("Failed to enable ARP (0x%x)\r\n", status);
            goto err;
        }

        /* Enable ICMP */
        status = nxd_icmp_enable(&ip_0);
        if (status) {
            printf_err("Failed to enable ICMP (0x%x)\r\n", status);
            goto err;
        }

        /* Let the IP thread execute */
        tx_thread_sleep(NX_IP_PERIODIC_RATE);

        status = nx_ip_status_check(&ip_0, NX_IP_LINK_ENABLED, &actual_status, 1000);

        if (status) {
            printf_err("Failed to check link status (0x%x)\r\n", status);
            return -1;
        } else {
            printf("Ethernet link is up.\r\n");
        }
    }

    printf("Set host to ping IP addr (%u.%u.%u.%u): ", dst_default_ip >> 24,
                                    (dst_default_ip >> 16) & 0xff,
                                    (dst_default_ip >> 8) & 0xff,
                                    dst_default_ip & 0xff);
    fflush(stdout);
    dst_default_ip = ReadUserInputAsIPAddr(dst_default_ip);
    printf("Pinging...");
    fflush(stdout);
    status = nx_icmp_ping(&ip_0, dst_default_ip, NULL, 0, &response_ptr, 1000);
    if (status) {
        printf_err("failed (0x%x)\r\n", status);
        return -1;
    } else {
        nx_packet_release(response_ptr);
        printf("done\r\n");
    }

    return 0;

err:
    nx_ip_delete(&ip_0);
    free(arp_cache_pointer);
err1:
    free(ip_mem_pointer);
    nx_packet_pool_delete(&pool_0);
err2:

    return -1;
}