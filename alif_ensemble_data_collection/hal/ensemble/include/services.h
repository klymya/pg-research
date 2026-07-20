#ifndef SERVICES_HAL_H
#define SERVICES_HAL_H

#include "services_lib_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @function    void mipi_clocks_init(void)
  \brief        Initializes MIPI clocks
  \param[in]    none
  \return       int
  */
int services_mipi_clocks_init(void);
SERVICES_version_data_t const* service_version_data_get(void);
char /*const*/ * services_alif_eui64_str_get(void);
void services_init(void);
void services_execute_aipm_stop_mode(const uint32_t wakeup_in_sec);
void pm_display_wakeup_reason(void);
void services_print_runp(void);
void services_print_offp(void);
#ifdef __cplusplus
}
#endif

#endif /* SERVICES_HAL_H */