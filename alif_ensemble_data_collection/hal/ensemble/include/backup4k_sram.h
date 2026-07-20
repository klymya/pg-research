#ifndef BACKUP4K_SRAM_H_
#define BACKUP4K_SRAM_H_

#ifdef  __cplusplus
extern "C"
{
#endif

#include <stdint.h>

struct backup_4k_t {
    uint32_t magic;
    /* add your data below */
    uint32_t power_cycle_cntr; /* Number of times we came out from the sleep mode, including initial POR */
    /* add your data above */
    uint32_t crc32;
};

uint32_t crc32(const uint8_t *data, uint32_t len);
void backup4k_sram_init(void);
struct backup_4k_t* backup4k_sram_get(void);
void backup4k_sram_recalc_crc(void);

#ifdef  __cplusplus
}
#endif

#endif /* BACKUP4K_SRAM_H_ */