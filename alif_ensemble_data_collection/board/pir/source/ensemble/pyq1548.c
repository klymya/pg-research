#include "pyq1548.h"
#include "tx_api.h"
#include "Driver_GPIO.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "pinconf.h"
#include "log_macros.h"
#include "board.h"

/* Serial In  */
extern  ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_PYQ1548_SERIAL_IN_PORT);
ARM_DRIVER_GPIO *gpioSerialIn = &ARM_Driver_GPIO_(BOARD_PYQ1548_SERIAL_IN_PORT);

/* DirectLink */
extern  ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_PYQ1548_DIRECT_LINK_PORT);
ARM_DRIVER_GPIO *gpioDirectLink = &ARM_Driver_GPIO_(BOARD_PYQ1548_DIRECT_LINK_PORT);

static uint32_t pir_cfg_reg_val;

on_motion_callback_t on_motion_callback = NULL;


#define CHECK_STATUS_AND_CALL(func, ...) \
    do { \
        int status = func(__VA_ARGS__); \
        if (status != ARM_DRIVER_OK) { \
            printf_err("Failed to call %s\r\n", #func); \
        } \
    } while (0)


static void direct_link_event(uint32_t event)
{
    UNUSED(event);
    gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
    sys_busy_loop_ns(PYQ1548_CLEAR_INTERRUPT);
    gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_INPUT);

    if(on_motion_callback) {
        on_motion_callback();
    }
}

int8_t pir_init(on_motion_callback_t on_motion_cb)
{
    int32_t ret1;
    int32_t ret2;

    on_motion_callback = on_motion_cb;

    uint32_t configSerialIn =
            PADCTRL_SLEW_RATE_FAST;

    uint32_t configDirectLink =
            PADCTRL_READ_ENABLE |
            PADCTRL_DRIVER_DISABLED_HIGH_Z |
            PADCTRL_SLEW_RATE_FAST;

    ret1 = gpioSerialIn->Initialize(BOARD_PYQ1548_SERIAL_IN_PIN, NULL);
    ret2 = gpioDirectLink->Initialize(BOARD_PYQ1548_DIRECT_LINK_PIN, direct_link_event);
    if ((ret1 != ARM_DRIVER_OK) || (ret2 != ARM_DRIVER_OK)) {
        printf_err("Failed to initialize a PIR gpio\r\n");
        return -1;
    }

    ret1 = gpioSerialIn->PowerControl(BOARD_PYQ1548_SERIAL_IN_PIN, ARM_POWER_FULL);
    ret2 = gpioDirectLink->PowerControl(BOARD_PYQ1548_DIRECT_LINK_PIN, ARM_POWER_FULL);
    if ((ret1 != ARM_DRIVER_OK) || (ret2 != ARM_DRIVER_OK)) {
        printf_err("Failed to power a PIR gpio\r\n");
        goto error_uninitialize;
    }

    ret1 = gpioSerialIn->SetDirection(BOARD_PYQ1548_SERIAL_IN_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    ret2 = gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    if ((ret1 != ARM_DRIVER_OK) || (ret2 != ARM_DRIVER_OK)) {
        printf_err("Failed to configure a PIR gpio\r\n");
        goto error_power_off;
    }

    pinconf_set(BOARD_PYQ1548_SERIAL_IN_PORT, BOARD_PYQ1548_SERIAL_IN_PIN, PINMUX_ALTERNATE_FUNCTION_0, configSerialIn);
    pinconf_set(BOARD_PYQ1548_DIRECT_LINK_PORT, BOARD_PYQ1548_DIRECT_LINK_PIN, PINMUX_ALTERNATE_FUNCTION_0, configDirectLink);

    gpioSerialIn->SetValue(BOARD_PYQ1548_SERIAL_IN_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
    gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_LOW);

    sys_busy_loop_init_ns();
    sys_busy_loop_ns(PYQ1548_CLEAR_INTERRUPT);

    return 0;

error_power_off:
    ret1 = gpioSerialIn->PowerControl(BOARD_PYQ1548_SERIAL_IN_PIN, ARM_POWER_OFF);
    ret2 = gpioDirectLink->PowerControl(BOARD_PYQ1548_DIRECT_LINK_PIN, ARM_POWER_OFF);
    if ((ret1 != ARM_DRIVER_OK) || (ret2 != ARM_DRIVER_OK)) {
        printf_err("Failed to power off a PIR gpio\r\n");
    } else {
        info("PIR gpios are power off\r\n");
    }

error_uninitialize:
    ret1 = gpioSerialIn->Uninitialize(BOARD_PYQ1548_SERIAL_IN_PIN);
    ret2 = gpioDirectLink->Uninitialize(BOARD_PYQ1548_DIRECT_LINK_PIN);
    if ((ret1 != ARM_DRIVER_OK) || (ret2 != ARM_DRIVER_OK)) {
        printf_err("Failed to Un-initialize a PIR gpio\r\n");
    } else {
        info("Un-initialized PIR gpio\r\n");
    }

    return -1;
}

int32_t pir_int_enable(bool enable){
    uint32_t control = ARM_GPIO_IRQ_SENSITIVE_EDGE
                    | ARM_GPIO_IRQ_EDGE_SENSITIVE_SINGLE
                    | ARM_GPIO_IRQ_POLARITY_HIGH;
    int32_t ret;
    if(enable){
        ret = gpioDirectLink->Control(BOARD_PYQ1548_DIRECT_LINK_PIN, ARM_GPIO_ENABLE_INTERRUPT, &control);
    } else {
        ret = gpioDirectLink->Control(BOARD_PYQ1548_DIRECT_LINK_PIN, ARM_GPIO_DISABLE_INTERRUPT, NULL);
    }
    return ret;
}

void pir_cfg_set(const uint32_t threshold, const enum blind_time_t blind_time,
                    const enum pulse_cntr_t pulse_cntr, const enum window_time_t window_time,
                    const enum op_modes_t op_mode, const enum signal_src_t signal_src,
                    const enum hpf_cutoff_t hpf_cutoff, const enum count_mode_t count_mode)
{
    pir_cfg_reg_val = PYQ1548_FIELD_SET(PYQ1548_THRESHOLD_OFFSET,
                        PYQ1548_THRESHOLD_MASK,
                        threshold);
    pir_cfg_reg_val |= PYQ1548_FIELD_SET(PYQ1548_BLIND_TIME_OFFSET,
                        PYQ1548_BLIND_TIME_MASK,
                        (uint32_t)blind_time);
    pir_cfg_reg_val |= PYQ1548_FIELD_SET(PYQ1548_PULSE_CNTR_OFFSET,
                        PYQ1548_PULSE_CNTR_MASK,
                        (uint32_t)pulse_cntr);
    pir_cfg_reg_val |= PYQ1548_FIELD_SET(PYQ1548_WINDOW_TIME_OFFSET,
                        PYQ1548_WINDOW_TIME_MASK,
                        (uint32_t)window_time);
    pir_cfg_reg_val |= PYQ1548_FIELD_SET(PYQ1548_OP_MODES_OFFSET,
                        PYQ1548_OP_MODES_MASK,
                        (uint32_t)op_mode);
    pir_cfg_reg_val |= PYQ1548_FIELD_SET(PYQ1548_SIGNAL_SRC_OFFSET,
                        PYQ1548_SIGNAL_SRC_MASK,
                        (uint32_t)signal_src);
    pir_cfg_reg_val |= 2U << 3; /* Reserved */
    pir_cfg_reg_val |= PYQ1548_FIELD_SET(PYQ1548_HPF_CUTOFF_OFFSET,
                        PYQ1548_HPF_CUTOFF_MASK,
                        (uint32_t)hpf_cutoff);
    pir_cfg_reg_val |= PYQ1548_FIELD_SET(PYQ1548_CNT_MODE_OFFSET,
                        PYQ1548_CNT_MODE_MASK,
                        (uint32_t)count_mode);
}

void pir_cfg_set_val(const uint32_t val)
{
    pir_cfg_reg_val = val;
}

uint32_t pir_cfg_get_val(void)
{
    return pir_cfg_reg_val;
}

void pir_cfg_apply(void)
{
    uint32_t msb = 1U << 24;

    //FIXMEAV: disable interrupts here
    gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
    gpioSerialIn->SetValue(BOARD_PYQ1548_SERIAL_IN_PIN, GPIO_PIN_OUTPUT_STATE_LOW);

    while (msb) {
        gpioSerialIn->SetValue(BOARD_PYQ1548_SERIAL_IN_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
        sys_busy_loop_ns(PYQ1548_DATA_T_SL_NS);
        gpioSerialIn->SetValue(BOARD_PYQ1548_SERIAL_IN_PIN, GPIO_PIN_OUTPUT_STATE_HIGH);
        sys_busy_loop_ns(PYQ1548_DATA_T_SH_NS);

        if (pir_cfg_reg_val & msb) {
            gpioSerialIn->SetValue(BOARD_PYQ1548_SERIAL_IN_PIN, GPIO_PIN_OUTPUT_STATE_HIGH);
        } else {
            gpioSerialIn->SetValue(BOARD_PYQ1548_SERIAL_IN_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
        }

        sys_busy_loop_ns(PYQ1548_DATA_IN_HOLD_TIME_NS);
        msb >>= 1;
    }
    gpioSerialIn->SetValue(BOARD_PYQ1548_SERIAL_IN_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
    gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_INPUT);
    //FIXMEAV: enable interrupts here
    sys_busy_loop_ns(PYQ1548_DATA_LOAD_TIME_NS);
}

void pir_read(int32_t* pir_val, uint32_t* stat_cfg)
{
    int i;
    uint32_t uibitmask;
    uint32_t ulbitmask;
    uint32_t value;

    gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_HIGH);
    sys_busy_loop_ns(PYQ1548_DATA_SETUP_TIME_NS);
    gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_LOW);

    // get first 15bit out-off-range and ADC value
    uibitmask = 0x4000;
    *pir_val = 0;

    for (i = 0; i < 15; i++) {
        // create low to high transition
        // Set DL = Low, duration must be > 200 ns (tL)
        // Configure DL as Output
        gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
        gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_OUTPUT);
        sys_busy_loop_ns(PYQ1548_DATA_T_DL_NS);
        // Set DL = High, duration must be > 200 ns (tH)
        // Configure DL as Input
        gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_HIGH);
        sys_busy_loop_ns(PYQ1548_DATA_T_DH_NS);
        gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_INPUT);
        // Wait for stable low signal
        sys_busy_loop_ns(PYQ1548_BIT_TIME_NS);

        // If DL High set masked bit in pir_val
        gpioDirectLink->GetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, &value);
        if (value) {
            *pir_val |= uibitmask;
        }
        uibitmask >>= 1;
    }

    // get 25bit status and config
    ulbitmask = 0x1000000;
    *stat_cfg = 0;
    for (i = 0; i < 25; i++) {
        // create low to high transition
        // Set DL = Low, duration must be > 200 ns (tL)
        // Configure DL as Output
        gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
        gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_OUTPUT);
        sys_busy_loop_ns(PYQ1548_DATA_T_DL_NS);
        // Set DL = High, duration must be > 200 ns (tH)
        // Configure DL as Input
        gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_HIGH);
        sys_busy_loop_ns(PYQ1548_DATA_T_DH_NS);
        gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_INPUT);
        // Wait for stable low signal, tbd empirically using scope
        sys_busy_loop_ns(PYQ1548_BIT_TIME_NS);

        gpioDirectLink->GetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, &value);
        if (value) {
            *stat_cfg |= ulbitmask;
        }
        ulbitmask >>= 1;
    }

    // Set DL = Low
    // Configure DL as Output
    gpioDirectLink->SetValue(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
    gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_OUTPUT);
    // Configure DL as Input
    gpioDirectLink->SetDirection(BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_INPUT);

    // clear unused bit
    *pir_val &= 0x3FFF;

    if (!(*stat_cfg & 0x60)) {
    // ADC source to PIR band pass
    // number in 14bit two's complement
        if (*pir_val & 0x2000) {
            *pir_val -= 0x4000;
        }
    }
}

bool isDirectLinkHigh(void)
{
    uint32_t value;

    CHECK_STATUS_AND_CALL(gpioDirectLink->GetValue, BOARD_PYQ1548_DIRECT_LINK_PIN, &value);

    if (value) {
        CHECK_STATUS_AND_CALL(gpioDirectLink->SetDirection, BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_OUTPUT);
        CHECK_STATUS_AND_CALL(gpioDirectLink->SetValue, BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_OUTPUT_STATE_LOW);
        sys_busy_loop_ns(PYQ1548_CLEAR_INTERRUPT);
        CHECK_STATUS_AND_CALL(gpioDirectLink->SetDirection, BOARD_PYQ1548_DIRECT_LINK_PIN, GPIO_PIN_DIRECTION_INPUT);
    }

    return value;
}

void pir_deinit(void)
{
    int32_t ret1;
    int32_t ret2;

    ret1 = gpioSerialIn->PowerControl(BOARD_PYQ1548_SERIAL_IN_PIN, ARM_POWER_OFF);
    ret2 = gpioDirectLink->PowerControl(BOARD_PYQ1548_DIRECT_LINK_PIN, ARM_POWER_OFF);
    if ((ret1 != ARM_DRIVER_OK) || (ret2 != ARM_DRIVER_OK)) {
        printf_err("Failed to power off a PIR gpio\r\n");
    } else {
        info("PIR gpios are power off\r\n");
    }

    ret1 = gpioSerialIn->Uninitialize(BOARD_PYQ1548_SERIAL_IN_PIN);
    ret2 = gpioDirectLink->Uninitialize(BOARD_PYQ1548_DIRECT_LINK_PIN);
    if ((ret1 != ARM_DRIVER_OK) || (ret2 != ARM_DRIVER_OK)) {
        printf_err("Failed to Un-initialize a PIR gpio\r\n");
    } else {
        info("Un-initialized PIR gpio\r\n");
    }
}
