#include "ft24c256a.h"
#include "tx_api.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "Driver_I2C.h"
#include "pinconf.h"
#include "log_macros.h"

#define TX_I2C_MST_TRANSFER_DONE        0x01
#define TX_I2C_MST_TRANSFER_ADDR_NACK   0x02

#define I2C_ADDRS           (0x51)   /* 7 bit Target(Slave) Address, use by Master  */

#define RESTART             (0x01)
#define STOP                (0x00)

static uint8_t tx_buf[EEPROM_PAGE_SIZE];

extern ARM_DRIVER_I2C Driver_I2C0;
static ARM_DRIVER_I2C *I2C_MstDrv = &Driver_I2C0;
static TX_EVENT_FLAGS_GROUP event_flags_i2c;

static void hardware_init(void)
{
    /* I2C0_SDA_B */
    pinconf_set(PORT_3, PIN_5, PINMUX_ALTERNATE_FUNCTION_5,
         (PADCTRL_READ_ENABLE | PADCTRL_DRIVER_DISABLED_PULL_UP));

    /* I2C0_SCL_B */
    pinconf_set(PORT_3, PIN_4, PINMUX_ALTERNATE_FUNCTION_5,
         (PADCTRL_READ_ENABLE | PADCTRL_DRIVER_DISABLED_PULL_UP));
}

static void i2c_mst_transfer_callback(uint32_t event)
{
    if (event & ARM_I2C_EVENT_TRANSFER_DONE) {
        /* Master Transfer done. Wake-up Thread. */
        tx_event_flags_set(&event_flags_i2c, TX_I2C_MST_TRANSFER_DONE, TX_OR);
    }
    
    if (event & ARM_I2C_EVENT_ADDRESS_NACK) {
        tx_event_flags_set(&event_flags_i2c, TX_I2C_MST_TRANSFER_ADDR_NACK, TX_OR);
    }
}

int8_t eeprom_init(void)
{
    int ret;
    ARM_DRIVER_VERSION version;
    INT status;

    /* Pinmux */
    hardware_init();

    /* Create the event flags group used by I2C thread */
    status = tx_event_flags_create(&event_flags_i2c, "event flags EEPROM I2C");
    if (status != TX_SUCCESS) {
        printf_err("Could not create event flags\r\n");
        return -1;
    }

    version = I2C_MstDrv->GetVersion();
    info("\r\n I2C version api:0x%X driver:0x%X...\r\n", version.api, version.drv);

    /* Initialize I2C driver */
    ret = I2C_MstDrv->Initialize(i2c_mst_transfer_callback);
    if (ret != ARM_DRIVER_OK) {
        printf_err("EEPROM I2C master init failed\r\n");
        return -1;
    }

    /* Power control I2C */
    ret = I2C_MstDrv->PowerControl(ARM_POWER_FULL);
    if (ret != ARM_DRIVER_OK) {
        printf_err("EEPROM I2C Master Power up failed\r\n");
        goto error_poweroff;
    }

    /* I2C Master Control */
    ret = I2C_MstDrv->Control(ARM_I2C_BUS_SPEED, ARM_I2C_BUS_SPEED_STANDARD);
    if (ret != ARM_DRIVER_OK) {
        printf_err("EEPROM I2C Master Control failed\r\n");
        goto error_uninitialize;
    }

    return 0;

  error_poweroff:
      /* Power off I2C peripheral */
      ret = I2C_MstDrv->PowerControl(ARM_POWER_OFF);
      if(ret != ARM_DRIVER_OK) {
         printf_err("EEPROM I2C Master Power OFF failed.\r\n");
      }

  error_uninitialize:
      /* Un-initialize I2C driver */
      ret = I2C_MstDrv->Uninitialize();
      if (ret != ARM_DRIVER_OK) {
        printf_err("EEPROM I2C Master Uninitialize\r\n");
      }

    return -1;
}

void eeprom_deinit(void)
{
    int ret;

    /* Power off I2C peripheral */
    ret = I2C_MstDrv->PowerControl(ARM_POWER_OFF);
    if (ret != ARM_DRIVER_OK) {
        printf_err("EEPROM I2C Master Power OFF failed.\r\n");
    }

    /* Un-initialize I2C driver */
    ret = I2C_MstDrv->Uninitialize();
    if (ret != ARM_DRIVER_OK) {
        printf_err("EEPROM I2C Master Uninitialize\r\n");
    }

    tx_event_flags_delete(&event_flags_i2c);
}

bool eeprom_write_byte(uint16_t address, uint8_t data, uint32_t timeout_ms)
{
    int ret;
    ULONG events = 0;
    uint8_t byte_read;
    uint32_t write_num;

    if (!eeprom_read_byte(address, &byte_read, timeout_ms))
        return false;
    if (byte_read == data)
        return true;

#if EEPROM_NV24C04LV == 0
    tx_buf[0] = (uint8_t)((address >> 8) & 0xff);
    tx_buf[1] = (uint8_t)(address & 0xff);
    tx_buf[2] = data;
    write_num = 3;
#else
    tx_buf[0] = (uint8_t)(address & 0xff);
    tx_buf[1] = data;
    write_num = 2;
#endif

    tx_event_flags_set(&event_flags_i2c, 0, TX_AND);
    I2C_MstDrv->MasterTransmit(I2C_ADDRS, tx_buf, write_num, STOP);
    /* wait till master/slave isr callback */
    ret = tx_event_flags_get(&event_flags_i2c, TX_I2C_MST_TRANSFER_DONE,
                TX_OR_CLEAR, &events, timeout_ms / (1000 / TX_TIMER_TICKS_PER_SECOND));

    if (ret != TX_SUCCESS) {
        printf_err("EEPROM I2C tx_event_flags_get (%d)\r\n", ret);
        return false;
    }

    sys_busy_loop_us(EEPROM_WRITE_CYCLE_TIME_MAX_MS * 1000);

    return true;
}

bool eeprom_write_buf(uint16_t address, uint8_t* data, size_t len, uint32_t timeout_ms)
{
    bool res = true;

    while (len-- && res) {
        res = eeprom_write_byte(address++, *(data++), timeout_ms);
    }

    return res;
}

static bool eeprom_write_page(uint16_t address, uint8_t* data, size_t len, uint32_t timeout_ms)
{
    int ret;
    ULONG events = 0;
    uint32_t write_num;

#if EEPROM_NV24C04LV == 0
    tx_buf[0] = (uint8_t)((address >> 8) & 0xff);
    tx_buf[1] = (uint8_t)(address & 0xff);
    write_num = 2 + len;
    memcpy(&tx_buf[2], data, len);
#else
    tx_buf[0] = (uint8_t)(address & 0xff);
    write_num = 1 + len;
    memcpy(&tx_buf[1], data, len);
#endif

    tx_event_flags_set(&event_flags_i2c, 0, TX_AND);
    I2C_MstDrv->MasterTransmit(I2C_ADDRS, tx_buf, write_num, STOP);
    /* wait till master/slave isr callback */
    ret = tx_event_flags_get(&event_flags_i2c, TX_I2C_MST_TRANSFER_DONE,
                TX_OR_CLEAR, &events, timeout_ms / (1000 / TX_TIMER_TICKS_PER_SECOND));

    if (ret != TX_SUCCESS) {
        printf_err("EEPROM I2C tx_event_flags_get (%d)\r\n", ret);
        return false;
    }

    sys_busy_loop_us(EEPROM_WRITE_CYCLE_TIME_MAX_MS * 1000);

    return true;
}

bool eeprom_write_pages(uint16_t address, uint8_t* data, size_t len, uint32_t timeout_ms)
{
    uint8_t byte_read;
    size_t write_size;
    uint32_t idx;

    if (address + len > EEPROM_SIZE_BYTES) {
        printf_err("Writing %d bytes beyond EEPROM boundaries\r\n",
                    (address + len) - EEPROM_SIZE_BYTES);
        return false;
    }

    for (idx = 0; idx < len; ++idx) {
        if (!eeprom_read_byte(address + idx, &byte_read, timeout_ms))
            return false;
        if (byte_read != data[idx]) {
            break;
        } else if (idx >= (len - 1)) {
            return true;
        }
    }

    idx = 0;
    write_size = EEPROM_PAGE_SIZE - (address % EEPROM_PAGE_SIZE);
    write_size = write_size > len ? len : write_size;

    do {
        if (!eeprom_write_page(address, &data[idx], write_size, timeout_ms))
            return false;
        idx += write_size;
        address += write_size;
        len -= write_size;
        write_size = len > EEPROM_PAGE_SIZE ? EEPROM_PAGE_SIZE : len;
    } while (len);

    return true;
}

static bool eeprom_set_addr_cntr(uint16_t address, uint32_t timeout_ms)
{
    uint32_t write_num;
    ULONG events = 0;
    int ret;

#if EEPROM_NV24C04LV == 0
    tx_buf[0] = (uint8_t)((address >> 8) & 0xff);
    tx_buf[1] = (uint8_t)(address & 0xff);
    write_num = 2;
#else
    tx_buf[0] = (uint8_t)(address & 0xff);
    write_num = 1;
#endif

    tx_event_flags_set(&event_flags_i2c, 0, TX_AND);
    ret = I2C_MstDrv->MasterTransmit(I2C_ADDRS, tx_buf, write_num, STOP);
    ret = tx_event_flags_get(&event_flags_i2c, TX_I2C_MST_TRANSFER_DONE,
                TX_OR_CLEAR, &events, timeout_ms / (1000 / TX_TIMER_TICKS_PER_SECOND));

    if (ret != TX_SUCCESS) {
        printf_err("EEPROM I2C tx_event_flags_get (%d)\r\n", ret);
        return false;
    }

    return true;
}

bool eeprom_read_byte(uint16_t address, uint8_t* data, uint32_t timeout_ms)
{
    ULONG events = 0;
    int ret;

    if (!eeprom_set_addr_cntr(address, timeout_ms))
        return false;

    tx_event_flags_set(&event_flags_i2c, 0, TX_AND);
    ret = I2C_MstDrv->MasterReceive(I2C_ADDRS, data, 1, STOP);
    ret = tx_event_flags_get(&event_flags_i2c, TX_I2C_MST_TRANSFER_DONE,
                TX_OR_CLEAR, &events, timeout_ms / (1000 / TX_TIMER_TICKS_PER_SECOND));

    if (ret != TX_SUCCESS) {
        printf_err("EEPROM I2C tx_event_flags_get (%d)\r\n", ret);
        return false;
    }

    return true;
}

bool eeprom_read_buf(uint16_t address, uint8_t* data, size_t len, uint32_t timeout_ms)
{
    bool res = true;

    while (len-- && res) {
        res = eeprom_read_byte(address++, data++, timeout_ms);
    }

    return res;
}

bool eeprom_read_seq(uint16_t address, uint8_t* data, size_t len, uint32_t timeout_ms)
{
    ULONG events = 0;
    int ret;

    if (address + len > EEPROM_SIZE_BYTES) {
        printf_err("Reading %d bytes beyond EEPROM boundaries\r\n",
                    (address + len) - EEPROM_SIZE_BYTES);
        return false;
    }

    if (!eeprom_set_addr_cntr(address, timeout_ms))
        return false;

    tx_event_flags_set(&event_flags_i2c, 0, TX_AND);
    ret = I2C_MstDrv->MasterReceive(I2C_ADDRS, data, len, STOP);
    ret = tx_event_flags_get(&event_flags_i2c, TX_I2C_MST_TRANSFER_DONE,
                TX_OR_CLEAR, &events, timeout_ms / (1000 / TX_TIMER_TICKS_PER_SECOND));

    if (ret != TX_SUCCESS) {
        printf_err("EEPROM I2C tx_event_flags_get (%d)\r\n", ret);
        return false;
    }

    return true;
}