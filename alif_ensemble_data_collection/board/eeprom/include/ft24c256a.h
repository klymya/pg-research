#ifndef FT24C256A_H_
#define FT24C256A_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define EEPROM_NV24C04LV 1

#if EEPROM_NV24C04LV == 0
#define EEPROM_PAGE_SIZE    64
#define EEPROM_PAGE_COUNT   512
#define EEPROM_SIZE_BYTES   (EEPROM_PAGE_COUNT * EEPROM_PAGE_SIZE)
#define EEPROM_WRITE_CYCLE_TIME_MAX_MS  5
#else
#define EEPROM_PAGE_SIZE    16
#define EEPROM_PAGE_COUNT   32
#define EEPROM_SIZE_BYTES   (EEPROM_PAGE_COUNT * EEPROM_PAGE_SIZE)
#define EEPROM_WRITE_CYCLE_TIME_MAX_MS  4
#endif

/* CPP guard */
#ifdef __cplusplus
extern "C" {
#endif

int8_t eeprom_init(void);
void eeprom_deinit(void);

bool eeprom_write_byte(uint16_t address, uint8_t data, uint32_t timeout_ms);
bool eeprom_write_buf(uint16_t address, uint8_t* data, size_t len, uint32_t timeout_ms);
bool eeprom_write_pages(uint16_t address, uint8_t* data, size_t len, uint32_t timeout_ms);

bool eeprom_read_byte(uint16_t address, uint8_t* data, uint32_t timeout_ms);
bool eeprom_read_buf(uint16_t address, uint8_t* data, size_t len, uint32_t timeout_ms);
bool eeprom_read_seq(uint16_t address, uint8_t* data, size_t len, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif /* End of CPP guard */
#endif /* FT24C256A_H_ */
