#include "backup4k_sram.h"
#include "log_macros.h"
#include <string.h>

#define BACKUP_SRAM_ADDR    0x4902C000
#define BACKUP_SRAM_SIZE    0x1000

#define BACKUP_SRAM_MAGIC   0x12CD98BA

static struct backup_4k_t* backup_4k = (struct backup_4k_t *)(BACKUP_SRAM_ADDR);

/**
  \brief        Calculate 32-bit CRC (polynom: 0x04C11DB7, init value: 0xFFFFFFFF)
  \param[in]    data  pointer to data
  \param[in]    len   data length (in bytes)
  \return             CRC32 value
*/
uint32_t crc32(const uint8_t *data, uint32_t len)
{
  uint32_t crc32;
  uint32_t n;

  crc32 = 0xFFFFFFFFU;
  while (len != 0U) {
    crc32 ^= ((uint32_t)*data++) << 24U;
    for (n = 8U; n; n--) {
      if (crc32 & 0x80000000U) {
        crc32 <<= 1U;
        crc32  ^= 0x04C11DB7U;
      } else {
        crc32 <<= 1U;
      }
    }
    len--;
  }
  return (crc32);
}

static uint32_t backup4k_sram_get_crc_buf_size(void)
{
    return (uint8_t *)&backup_4k->crc32 - (uint8_t *)backup_4k;
}

void backup4k_sram_init(void)
{
    if (backup_4k->magic != BACKUP_SRAM_MAGIC ||
            crc32((uint8_t *)backup_4k, 
                backup4k_sram_get_crc_buf_size()) != backup_4k->crc32) {
        warn("Backup 4K SRAM is not valid -> initializing!\r\n");
        memset(backup_4k, 0x0, sizeof(*backup_4k));
        backup_4k->magic = BACKUP_SRAM_MAGIC;
    } else {
      ++backup_4k->power_cycle_cntr;
      warn("Wake up from STOP mode counter: %lu\r\n", backup_4k->power_cycle_cntr);
    }
}

struct backup_4k_t* backup4k_sram_get(void)
{
    return backup_4k;
}

void backup4k_sram_recalc_crc(void)
{
    backup_4k->crc32 = crc32((uint8_t *)backup_4k, 
        backup4k_sram_get_crc_buf_size());
}