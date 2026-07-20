#include "TestEeprom.h"
#include "ft24c256a.h"
#include "log_macros.h"
#include "uart_tracelib.h"

int32_t test_eeprom(void)
{
    bool res = true;
    int counter;
    uint16_t cur_addr;
    uint8_t data_read[4] = {0};
    uint8_t data_to_write[] = { 0xaa, 0x55, 0xaa, 0x55 };

    eeprom_init();

    for (counter = 0; counter < 4 && res; ++counter) {
        cur_addr = counter * (EEPROM_SIZE_BYTES >> 2);
        res = eeprom_write_buf(cur_addr,
                    data_to_write, sizeof(data_to_write), 1000);
    }

    if (!res) {
        printf_err("Failed to write to EEPROM address (0x%x)\r\n", cur_addr);
        goto err;
    }

    for (counter = 0; counter < 4 && res; ++counter) {
        cur_addr = counter * (EEPROM_SIZE_BYTES >> 2);
        res = eeprom_read_seq(cur_addr,
                    data_read, sizeof(data_read), 1000);
        if (!res) {
            printf_err("Failed to read from EEPROM address (0x%x)\r\n",
                cur_addr);
            break;
        }
        if (*reinterpret_cast<uint32_t*>(&data_read[0]) !=
                *reinterpret_cast<uint32_t*>(&data_to_write[0])) {
            printf_err("Expected 0x%lx, read 0x%lx, address 0x%lx\r\n",
                *reinterpret_cast<uint32_t*>(&data_to_write[0]),
                *reinterpret_cast<uint32_t*>(&data_read[0]),
                cur_addr);
            res = false;
            break;
        }
    }

err:
    eeprom_deinit();

    return res ? 0 : -1;
}

int32_t test_eeprom_send(void)
{
    const uint8_t UART_START_EEPROM[] = {0xEE, 0x09};
    int size_bytes = 1;
    int32_t result = test_eeprom();
    uint8_t byte_result = result ? 0 : 1;

    send_str(const_cast<uint8_t*>(UART_START_EEPROM), sizeof(UART_START_EEPROM));
    send_str(&size_bytes, sizeof(size_bytes));
    send_str(&byte_result, 1);

    return result;
}