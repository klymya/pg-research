#ifndef PG_PROTO_H_
#define PG_PROTO_H_

#include "Detection.hpp"
#include <vector>
#include <stdint.h>

typedef enum {
    PB_ACK_EVENT     = (1 << 0),
    PB_NACK_EVENT    = (1 << 1),
    PB_XY_EVENT      = (1 << 2),

    PB_ERROR_EVENT   = (1 << 30),
    PB_TIMEOUT_EVENT = (1 << 31)
} PB_EVENTS;

void pb_init(void);
void pb_set_wait_seq_num(const uint32_t seq_sum);
PB_EVENTS pb_wait_reply(const uint32_t timeout_ms);

/* Decoding */
bool pb_decode_msg(void* data, size_t data_len);

/* Encoding */
struct radio_frame_t* pb_encode_xy_msg(
        std::vector<arm::app::pg_object_detection::Detection>* detections, uint32_t timestampSec,
        size_t* encoded_buf_size);
struct radio_frame_t* pb_encode_ack_msg(uint32_t seq_num,
                                            size_t* encoded_buf_size);
#endif //PG_PROTO