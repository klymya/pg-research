#include <stdio.h>
#include "uart_tracelib.h"
#include "TestEnvSensor.h"
#include "bme68x_common.h"
#include "tx_api.h"

/* Macro for count of samples to be displayed */
#define SAMPLE_COUNT  UINT16_C(5)

static int test_envsensor_init(struct bme68x_dev* bme,
        struct bme68x_conf* conf,
        struct bme68x_heatr_conf* heatr_conf)
{
    int8_t rslt;

    rslt = bme68x_interface_init(bme, BME68X_SPI_INTF);
    if (rslt != BME68X_OK) {
        goto err;
    }

    rslt = bme68x_init(bme);
    if (rslt != BME68X_OK) {
        goto err;
    }

    conf->filter = BME68X_FILTER_OFF;
    conf->odr = BME68X_ODR_NONE;
    conf->os_hum = BME68X_OS_16X;
    conf->os_pres = BME68X_OS_1X;
    conf->os_temp = BME68X_OS_2X;
    rslt = bme68x_set_conf(conf, bme);
    if (rslt != BME68X_OK) {
        goto err;
    }

    heatr_conf->enable = BME68X_ENABLE;
    heatr_conf->heatr_temp = 300;
    heatr_conf->heatr_dur = 100;
    rslt = bme68x_set_heatr_conf(BME68X_FORCED_MODE, heatr_conf, bme);
    if (rslt != BME68X_OK) {
        goto err;
    }

    return 0;

err:
    return rslt;
}

static int test_envsensor_get_data(struct bme68x_dev* bme,
        struct bme68x_conf* conf,
        struct bme68x_heatr_conf* heatr_conf,
        struct bme68x_data* data,
        uint8_t* n_fields,
        uint32_t* time_ms)
{
    int8_t rslt;
    uint32_t del_period;
    const uint8_t op_mode = BME68X_FORCED_MODE;

    rslt = bme68x_set_op_mode(op_mode, bme);
    if (rslt != BME68X_OK) {
        return rslt;
    }

    /* Calculate delay period in microseconds */
    del_period = bme68x_get_meas_dur(op_mode, conf, bme) + (heatr_conf->heatr_dur * 1000);
    bme->delay_us(del_period, bme->intf_ptr);

    if (time_ms != nullptr) {
        *time_ms = tx_time_get();
    }

    /* Check if rslt == BME68X_OK, report or handle if otherwise */
    rslt = bme68x_get_data(op_mode, data, n_fields, bme);

    return rslt;
}

int32_t test_envsensor(void)
{
    int8_t rslt;
    struct bme68x_dev bme;
    struct bme68x_conf conf;
    struct bme68x_heatr_conf heatr_conf;
    struct bme68x_data data;
    uint8_t n_fields;
    uint32_t time_ms = 0;
    uint16_t sample_count = 1;

    rslt = test_envsensor_init(&bme, &conf, &heatr_conf);
    bme68x_check_rslt("test_envsensor_init", rslt);
    if (rslt != BME68X_OK) {
        goto err;
    }

    printf("Sample, TimeStamp(ms), Temperature(deg C), Pressure(Pa), Humidity(%%), Gas resistance(ohm), Status\r\n");

    while (sample_count <= SAMPLE_COUNT) {
        rslt = test_envsensor_get_data(&bme, &conf, &heatr_conf, &data, &n_fields, &time_ms);

        if (rslt != BME68X_OK) {
            bme68x_check_rslt("test_envsensor_get_data", rslt);
            break;
        }

        if (n_fields) {
#ifdef BME68X_USE_FPU
            printf("%u, \t%lu, \t\t%.2f, \t\t   %.2f, \t %.2f, \t\t%.2f, \t0x%x\r\n",
                   sample_count,
                   (long unsigned int)time_ms,
                   data.temperature,
                   data.pressure,
                   data.humidity,
                   data.gas_resistance,
                   data.status);
#else
            printf("%u, %lu, %d, %lu, %lu, %lu, 0x%x\r\n",
                   sample_count,
                   (long unsigned int)time_ms,
                   (data.temperature / 100),
                   (long unsigned int)data.pressure,
                   (long unsigned int)(data.humidity / 1000),
                   (long unsigned int)data.gas_resistance,
                   data.status);
#endif
            sample_count++;
            if (sample_count <= SAMPLE_COUNT) {
                /* Sleep to cool the sensor off */
                tx_thread_sleep(800);
            }
        }
    }

err:
    bme68x_alif_deinit();

    return rslt;
}

int32_t test_envsensor_send(void)
{
    struct bme68x_dev bme;
    struct bme68x_conf conf;
    struct bme68x_heatr_conf heatr_conf;
    struct bme68x_data data;
    uint8_t n_fields;
    const uint8_t UART_START_ENV_SENSOR[] = {0x05, 0xFA};
    int size_bytes;
    int8_t rslt;
    char json_result_str[256];

    rslt = test_envsensor_init(&bme, &conf, &heatr_conf);
    if (rslt == BME68X_OK) {
        rslt = test_envsensor_get_data(&bme, &conf, &heatr_conf, &data, &n_fields, nullptr);
        if (rslt == BME68X_OK) {
            snprintf(json_result_str, sizeof(json_result_str) - 1,
                "{\"temperature\": %.2f, \"pressure\": %.2f, \"humidity\": %.2f, \"gas_resistance\": %.2f}",
                data.temperature,
                data.pressure,
                data.humidity,
                data.gas_resistance);
        }
    }

    if (rslt != BME68X_OK) {
        snprintf(json_result_str, sizeof(json_result_str) - 1, 
            "{\"error\": \"Failed to get env sensor data code = %d\"}", rslt);
    }

    size_bytes = strlen(json_result_str);
    send_str(const_cast<uint8_t*>(UART_START_ENV_SENSOR), sizeof(UART_START_ENV_SENSOR));
    send_str(&size_bytes, sizeof(size_bytes));
    send_str(&json_result_str, size_bytes);

err:
    bme68x_alif_deinit();

    return rslt;
}