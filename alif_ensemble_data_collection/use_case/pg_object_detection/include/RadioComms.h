#ifndef _RADIO_COMMS_H
#define _RADIO_COMMS_H

#include "Detection.hpp"
#include "PGProto.h"
#include <vector>
#include <stdint.h>
#include "tx_api.h"

struct radio_frame_t {
    uint32_t crc32;      /* crc of the all fields below */
    uint32_t seq_num;    /* sequence number */
    uint8_t  pb_data[];  /* protobuf payload */
} __attribute__((packed));

#define RADIO_FRAME_HDR_SIZE (sizeof(struct radio_frame_t))

void radio_comms_init(void);
void radio_comms_thread(ULONG thread_input);
uint32_t radio_comms_seq_num_get(void);

/* Transmission functions */
PB_EVENTS radio_comms_xy_send(
    std::vector<arm::app::pg_object_detection::Detection>* detections, uint32_t timestampSec);
void radio_comms_ack_send(const uint32_t seq_num);

#endif /* _RADIO_COMMS_H */