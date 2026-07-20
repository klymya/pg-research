#ifndef AUDIO_DATA_PDM_H
#define AUDIO_DATA_PDM_H

#include <stdint.h>

#ifdef __cplusplus
/* Yes, C++ compiler is present.  Use standard C.  */
extern   "C" {
#endif

int audio_init_pdm(void);
int wait_for_audio_pdm(int16_t* sample_buf, uint32_t sample_num);
void audio_deinit_pdm(void);

#ifdef  __cplusplus
}
#endif

#endif /* AUDIO_DATA_PDM_H */