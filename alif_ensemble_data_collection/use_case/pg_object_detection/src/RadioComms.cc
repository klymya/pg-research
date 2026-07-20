#include "RadioComms.h"
#ifndef TARGET_PLATFORM_NATIVE
#include "uart_radio.h"
#include "cobs_stuffing.h"
#include "log_macros.h"
#include "backup4k_sram.h"

#define DEBUG_DETECTIONS 0

#ifndef MAX_RECV_PACKET_SIZE
#define MAX_RECV_PACKET_SIZE    256
#endif

/* Input buffer */
static unsigned char radio_comms_recv_buf[MAX_RECV_PACKET_SIZE];
static uint32_t radio_comms_seq_num;
static TX_SEMAPHORE comms_semaphore;
static uint32_t recv_frame_crc_error_cntr;


static void radio_comms_print_buf(char* msg, uint8_t* buf, size_t buf_size)
{
#if DEBUG_DETECTIONS == 1
    printf("%s [%d]\r\npayload: [", msg, buf_size);
    for (int i = 0; i < buf_size; ++i) {
        printf("%02x", buf[i]);
    }
    printf("]\r\n");
#endif
}

static uint8_t* radio_comms_cobs_encode_pb(struct radio_frame_t* pb_encoded_frame,
                    size_t pb_encoded_frame_size, size_t* msg_len)
{
    if (pb_encoded_frame) {
        //FIXMEAV: use staic buffer instead?
        uint8_t* cobs_encoded_buf = new uint8_t[pb_encoded_frame_size * 2 + 2];

        radio_comms_print_buf("[send] protobuf encoded buf size:",
                reinterpret_cast<uint8_t*>(pb_encoded_frame), pb_encoded_frame_size);

        *msg_len = cobsEncode(pb_encoded_frame, pb_encoded_frame_size, cobs_encoded_buf + 1);
        cobs_encoded_buf[0] = 0; /* Prepend packet delimiter byte */
        cobs_encoded_buf[++(*msg_len)] = 0; /* Append packet delimiter byte */
        ++(*msg_len);
        delete [] pb_encoded_frame;

        radio_comms_print_buf("[send] COBS encoded buf size:", cobs_encoded_buf, *msg_len);

        return cobs_encoded_buf;
    }

    return nullptr;
}

void radio_comms_init(void)
{
    radio_comms_seq_num = backup4k_sram_get()->power_cycle_cntr << 16;
    if (tx_semaphore_create(&comms_semaphore,
            "radio comms semaphore", 1) != TX_SUCCESS) {
        printf_err("Failed to initialize radio comms semaphore\r\n");

        return;
    }
}

static uint32_t radio_comms_next_seq_num_get(void)
{
    if ((radio_comms_seq_num & 0xffffUL) == 0xffffUL) {
        radio_comms_seq_num &= ~0xffffUL;
    } else {
        ++radio_comms_seq_num;
    }

    return radio_comms_seq_num;
}

uint32_t radio_comms_seq_num_get(void)
{
    return radio_comms_seq_num;
}

/* To be called only inside radio_comms_send_wait_for_reply() */
static void radio_comms_send(struct radio_frame_t* pb_encoded_frame,
                    size_t pb_encoded_frame_size)
{
    size_t msg_len;
    uint8_t* cobs_encoded_buf;

    if (!pb_encoded_frame)
        return;

    pb_encoded_frame->seq_num = radio_comms_next_seq_num_get();
    pb_encoded_frame->crc32 = crc32(reinterpret_cast<uint8_t*>(&pb_encoded_frame->seq_num),
                    pb_encoded_frame_size - (RADIO_FRAME_HDR_SIZE - 4));

    cobs_encoded_buf = radio_comms_cobs_encode_pb(pb_encoded_frame,
                                            pb_encoded_frame_size, &msg_len);
    if (cobs_encoded_buf) {
        pb_set_wait_seq_num(radio_comms_seq_num_get());
        radio_send_data(cobs_encoded_buf, msg_len);
        delete [] cobs_encoded_buf;
    }
}

static PB_EVENTS radio_comms_send_wait_for_reply(
                    struct radio_frame_t* pb_encoded_frame,
                    size_t pb_encoded_frame_size, bool wait_for_reply,
                    uint32_t timeout_ms)
{
    PB_EVENTS status = PB_ERROR_EVENT;

    if (pb_encoded_frame) {
        tx_semaphore_get(&comms_semaphore, TX_WAIT_FOREVER);
        radio_comms_send(pb_encoded_frame, pb_encoded_frame_size);
        if (wait_for_reply) {
            status = pb_wait_reply(timeout_ms);
        }
        tx_semaphore_put(&comms_semaphore);
    }

    return status;
}

void radio_comms_thread(ULONG thread_input)
{
    (void)thread_input;
    int index = 0;
    int8_t c = 0xff;

    /* Waiting for initial 0 byte, that indicates beginning of a packet */
    while (1) {
        if (!radio_recv_data(&c, 1) && !c) {
            break;
        }
        tx_thread_sleep(1);
    }

    while (1) {
        if (radio_recv_data(&c, 1)) {
            printf_err("Failed to read byte from the radio uart\r\n");
            tx_thread_sleep(100);
            continue;
        }

        if (!c && !index)
            continue;

        if (index < sizeof(radio_comms_recv_buf)) {
            radio_comms_recv_buf[index++] = c;
        } else {
            printf_err("Receive buffer is too small - discarding frame\r\n");
            index = 0;
        }

        if (!c && index) {
            /* The end of a frame is detected */
            uint8_t *decoded_buf = new uint8_t[index]; //FIXMEAV: use staic buffer instead?
            size_t decoded_len = cobsDecode(radio_comms_recv_buf, index, decoded_buf);

            radio_comms_print_buf("[recv] COBS encoded buf size:", radio_comms_recv_buf, index);
            radio_comms_print_buf("[recv] protobuf encoded buf size:", decoded_buf, decoded_len);            
            
            if (decoded_len > RADIO_FRAME_HDR_SIZE) {
                uint32_t buf_calc_crc32;
                struct radio_frame_t* pb_encoded_frame = reinterpret_cast<struct radio_frame_t*>(decoded_buf);

                buf_calc_crc32 = crc32(reinterpret_cast<uint8_t*>(&pb_encoded_frame->seq_num),
                                        decoded_len - (RADIO_FRAME_HDR_SIZE - 4));

                if (pb_encoded_frame->crc32 == buf_calc_crc32) {
                    if (!pb_decode_msg(&pb_encoded_frame->pb_data[0],
                            decoded_len - RADIO_FRAME_HDR_SIZE)) {
                        warn("Failed to decode pb message\r\n");
                    }
                } else {
                    ++recv_frame_crc_error_cntr;
                    warn("Received frame CRC error, expected 0x%x, calculated 0x%x, counter %lu\r\n",
                            pb_encoded_frame->crc32, buf_calc_crc32, recv_frame_crc_error_cntr);
                }
            }
            delete [] decoded_buf;
            index = 0;
        }
    }
}

/*==============================================================*/
/* Transfer functions for different packets to be defined below */
/*==============================================================*/
PB_EVENTS radio_comms_xy_send(
    std::vector<arm::app::pg_object_detection::Detection>* detections, uint32_t timestampSec)
{
    size_t pb_encoded_frame_size;

    return radio_comms_send_wait_for_reply(
                    pb_encode_xy_msg(detections, timestampSec, &pb_encoded_frame_size),
                    pb_encoded_frame_size,
                    true, 1000);
}

void radio_comms_ack_send(const uint32_t seq_num)
{
    size_t pb_encoded_frame_size;

    radio_comms_send_wait_for_reply(
                    pb_encode_ack_msg(seq_num, &pb_encoded_frame_size),
                    pb_encoded_frame_size,
                    false, 0);
}

#else
PB_EVENTS radio_comms_xy_send(
    std::vector<arm::app::pg_object_detection::Detection>* detections, uint32_t timestampSec)
{
    (void)detections;
    (void)timestampSec;
}
#endif //TARGET_PLATFORM_NATIVE