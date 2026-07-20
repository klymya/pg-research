#ifndef LOW_POWER_H_
#define LOW_POWER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t lptimer_init(void);
int32_t set_lptimer(uint32_t timeout_sec);

#ifdef __cplusplus
}
#endif

#endif /* LOW_POWER_H_ */