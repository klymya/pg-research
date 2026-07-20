#ifndef PYQ1548_H_
#define PYQ1548_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define PYQ1548_THRESHOLD_OFFSET    17
#define PYQ1548_BLIND_TIME_OFFSET   13
#define PYQ1548_PULSE_CNTR_OFFSET   11
#define PYQ1548_WINDOW_TIME_OFFSET  9
#define PYQ1548_OP_MODES_OFFSET     7
#define PYQ1548_SIGNAL_SRC_OFFSET   5
#define PYQ1548_HPF_CUTOFF_OFFSET   2
#define PYQ1548_CNT_MODE_OFFSET     0

#define PYQ1548_THRESHOLD_MASK    0xff
#define PYQ1548_BLIND_TIME_MASK   0xf
#define PYQ1548_PULSE_CNTR_MASK   3
#define PYQ1548_WINDOW_TIME_MASK  3
#define PYQ1548_OP_MODES_MASK     3
#define PYQ1548_SIGNAL_SRC_MASK   3
#define PYQ1548_HPF_CUTOFF_MASK   1
#define PYQ1548_CNT_MODE_MASK     1

#define PYQ1548_FIELD_SET(offt, mask, val)  (((val) & (mask)) << (offt))

#define PYQ1548_DATA_IN_HOLD_TIME_NS    (100 * 1000)
#define PYQ1548_DATA_LOAD_TIME_NS       (650 * 1000)
#define PYQ1548_DATA_T_SL_NS            (500)
#define PYQ1548_DATA_T_SH_NS            (500)
#define PYQ1548_CLEAR_INTERRUPT         (170 * 1000)
#define PYQ1548_DATA_SETUP_TIME_NS      (120 * 1000)
#define PYQ1548_BIT_TIME_NS             (21 * 1000)
#define PYQ1548_DATA_T_DL_NS            (500)
#define PYQ1548_DATA_T_DH_NS            (500)

enum blind_time_t {
    PYQ1548_BLIND_TIME_500MS,
    PYQ1548_BLIND_TIME_1SEC,
    PYQ1548_BLIND_TIME_1SEC_500MS,
    PYQ1548_BLIND_TIME_2SEC,
    PYQ1548_BLIND_TIME_2SEC_500MS,
    PYQ1548_BLIND_TIME_3SEC,
    PYQ1548_BLIND_TIME_3SEC_500MS,
    PYQ1548_BLIND_TIME_4SEC,
    PYQ1548_BLIND_TIME_4SEC_500MS,
    PYQ1548_BLIND_TIME_5SEC,
    PYQ1548_BLIND_TIME_5SEC_500MS,
    PYQ1548_BLIND_TIME_6SEC,
    PYQ1548_BLIND_TIME_6SEC_500MS,
    PYQ1548_BLIND_TIME_7SEC,
    PYQ1548_BLIND_TIME_7SEC_500MS,
    PYQ1548_BLIND_TIME_8SEC,
};

enum pulse_cntr_t {
    PYQ1548_PULSE_CNTR_1,
    PYQ1548_PULSE_CNTR_2,
    PYQ1548_PULSE_CNTR_3,
    PYQ1548_PULSE_CNTR_4,
};

enum window_time_t {
    PYQ1548_WINDOW_TIME_2SEC,
    PYQ1548_WINDOW_TIME_4SEC,
    PYQ1548_WINDOW_TIME_6SEC,
    PYQ1548_WINDOW_TIME_8SEC,
};

enum op_modes_t {
    PYQ1548_OP_MODES_FORCED_READOUT,
    PYQ1548_OP_MODES_INTERRUPT_READOUT,
    PYQ1548_OP_MODES_WAKEUP,
};

enum signal_src_t {
    PYQ1548_SIGNAL_SRC_PIR_BFP,
    PYQ1548_SIGNAL_SRC_PIR_LFP,
    PYQ1548_SIGNAL_SRC_TEMP_SENSOR = 3,
};

enum hpf_cutoff_t {
    PYQ1548_HPF_CUTOFF_0_4HZ,
    PYQ1548_HPF_CUTOFF_0_2HZ,
};

enum count_mode_t {
    PYQ1548_COUNT_WITH_BPF_SIGN,
    PYQ1548_COUNT_WITHOUT_BPF_SIGN,
};

/* CPP guard */
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*on_motion_callback_t)(void);

int8_t pir_init(on_motion_callback_t on_motion_cb);
void pir_deinit(void);

void pir_cfg_set(const uint32_t threshold, const enum blind_time_t blind_time,
                    const enum pulse_cntr_t pulse_cntr, const enum window_time_t window_time,
                    const enum op_modes_t op_mode, const enum signal_src_t signal_src,
                    const enum hpf_cutoff_t hpf_cutoff, const enum count_mode_t count_mode);
void pir_cfg_set_val(const uint32_t val);
uint32_t pir_cfg_get_val(void);
int32_t pir_int_enable(bool enable);
void pir_cfg_apply(void);
/*
*  Keep in mind that DL line used both for force read and wake up mode
*  Because we need to be woken up by PIR sensor, we need to keep DL line connected to LPGPIO
*  But LPGPIO is clocked by 32kHz clock, wich makes it very slow
*  So slow that we can't use it for force read mode
*  Thus we can't read the configuration register
*/
void pir_read(int32_t* pir_val, uint32_t* stat_cfg);
bool isDirectLinkHigh(void);


#ifdef __cplusplus
}
#endif /* End of CPP guard */
#endif /* PYQ1548_H_ */
