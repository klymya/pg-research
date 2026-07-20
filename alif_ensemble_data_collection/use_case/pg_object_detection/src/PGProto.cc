#include <pb_encode.h>
#include <pb_decode.h>
#include <pb_common.h>
#include "pg.pb.h"
#include "Detection.hpp"
#include <vector>
#include <cstdio>
#include <tuple>
#include "services.h"
#include "backup4k_sram.h"
#include "log_macros.h"
#include "PGProto.h"
#include "RadioComms.h"

typedef std::tuple<std::vector<arm::app::pg_object_detection::Detection>*, uint32_t> XyData;

struct pb_event_t {
    PB_EVENTS id;
    void* data;
};

static TX_EVENT_FLAGS_GROUP decode_event_flags;
static TX_MUTEX decode_event_mutex;
static uint32_t decode_wait_seq_num_reply;

void pb_init(void)
{
    decode_wait_seq_num_reply = ~0UL;
    if (tx_event_flags_create(&decode_event_flags,
            "pb decode event flags") != TX_SUCCESS) {
        printf_err("Failed to initialize pb decode event flags\r\n");
        return;
    }

    if (tx_mutex_create(&decode_event_mutex,
            "pb decode event mutex", TX_NO_INHERIT) != TX_SUCCESS) {
        printf_err("Failed to initialize pb decode event mutex\r\n");
        return;
    }
}

void pb_set_wait_seq_num(const uint32_t seq_sum)
{
    UINT status;

    tx_mutex_get(&decode_event_mutex, TX_WAIT_FOREVER);
    decode_wait_seq_num_reply = seq_sum;
    status = tx_event_flags_set(&decode_event_flags, 0, TX_AND);
    tx_mutex_put(&decode_event_mutex);

    if (status != TX_SUCCESS) {
        printf_err("Failed to clear pb decode event flags (0x%x)\r\n", status);
        return;
    }
}

PB_EVENTS pb_wait_reply(const uint32_t timeout_ms)
{
    ULONG actual_events = 0;
    UINT status;

    status = tx_event_flags_get(&decode_event_flags,    \
            PB_ACK_EVENT | PB_NACK_EVENT,               \
            TX_OR_CLEAR,                                \
            &actual_events,                             \
            timeout_ms / (1000 / TX_TIMER_TICKS_PER_SECOND));

    switch (status) {
        case TX_SUCCESS:
            if (actual_events & PB_ACK_EVENT)
                return PB_ACK_EVENT;
            else if (actual_events & PB_NACK_EVENT)
                return PB_NACK_EVENT;
        break;
        case TX_NO_EVENTS:
            return PB_TIMEOUT_EVENT;
        break;
        default:
            printf_err("Failed to geet pb decode event flags (0x%x)\r\n", status);
        break;
    }

    return PB_ERROR_EVENT;
}
/*====================================================*/
/*                Encode functions                    */
/*====================================================*/
static bool fields_encode_ReportingObject(pb_ostream_t* stream, const pb_field_t* field, void* const* arg)
{
    telemetrypayload_ReportingObject ReportingObject = telemetrypayload_ReportingObject_init_default;
    auto detections = static_cast<std::vector<arm::app::pg_object_detection::Detection>*>(*arg);

	for (auto detection: *detections) {
        ReportingObject.position_x = detection.position.x;
        ReportingObject.position_y = detection.position.y;
        ReportingObject.box_x = detection.bbox.x;
        ReportingObject.box_y = detection.bbox.y;
        ReportingObject.box_w = detection.bbox.w;
        ReportingObject.box_h = detection.bbox.h;
        ReportingObject.confidence = detection.confidence;

	    if (!pb_encode_tag_for_field(stream, field)) {
			return false;
		}

		if (!pb_encode_submessage(stream, telemetrypayload_ReportingObject_fields, &ReportingObject)) {
			return false;
		}
	}

	return true;
}

static struct radio_frame_t* pb_encode_msg(
                                struct pb_event_t* event,
                                size_t* encoded_buf_size)
{
    telemetrypayload_Notification Notification = telemetrypayload_Notification_init_zero;

    switch (event->id) {
        case PB_XY_EVENT:
            Notification.which_notification = telemetrypayload_Notification_telemetry_tag;
            Notification.notification.telemetry.objects.arg = std::get<0>(*static_cast<XyData*>(event->data));
            Notification.notification.telemetry.timestamp = std::get<1>(*static_cast<XyData*>(event->data));
        break;
        case PB_ACK_EVENT:
            Notification.which_notification = telemetrypayload_Notification_ack_tag;
            Notification.notification.ack.seq_num = *static_cast<uint32_t*>(event->data);
        break;
    }

    if (pb_get_encoded_size(encoded_buf_size,
                telemetrypayload_Notification_fields, &Notification)) {
        uint8_t* buffer;
        bool status;
        pb_ostream_t stream;

        //FIXMEAV: use static buffer or implement sbrk for multithreading?
        buffer = new uint8_t[*encoded_buf_size + RADIO_FRAME_HDR_SIZE];
        stream = pb_ostream_from_buffer(buffer + RADIO_FRAME_HDR_SIZE,
                                *encoded_buf_size);
        status = pb_encode(&stream,
                    telemetrypayload_Notification_fields, &Notification);

        if (!status || (*encoded_buf_size != stream.bytes_written)) {
            printf_err("pg_encode_msg: error: encoded buffer size doesn't match\r\n");
            delete [] buffer;

            return nullptr;
        }

        *encoded_buf_size += RADIO_FRAME_HDR_SIZE;
        return reinterpret_cast<struct radio_frame_t*>(buffer);
    }

    return nullptr;
}

struct radio_frame_t* pb_encode_xy_msg(
        std::vector<arm::app::pg_object_detection::Detection>* detections, uint32_t timestampSec,
        size_t* encoded_buf_size)
{
    if (detections->size()) {
        XyData data = std::make_tuple(detections, timestampSec);
        struct pb_event_t event = {.id = PB_XY_EVENT, .data = &data};

        return pb_encode_msg(&event, encoded_buf_size);
    }

    return nullptr;
}

struct radio_frame_t* pb_encode_ack_msg(uint32_t seq_num,
                                            size_t* encoded_buf_size)
{
    struct pb_event_t event = {.id = PB_ACK_EVENT, .data = &seq_num};

    return pb_encode_msg(&event, encoded_buf_size);
}

/*====================================================*/
/*                Decode functions                    */
/*====================================================*/
static bool fields_decode_string(pb_istream_t *stream, const pb_field_t *field, void **arg)
{
    uint8_t buffer[124] = {0}; //FIXMEAV: set proper size of the buffer

    /* We could read block-by-block to avoid the large buffer... */
    if (stream->bytes_left > sizeof(buffer) - 1)
        return false;

    if (!pb_read(stream, buffer, stream->bytes_left))
        return false;

    return true;
}

bool pb_decode_msg(void* data, size_t data_len)
{
    pb_istream_t stream;
    bool status;

    stream = pb_istream_from_buffer(static_cast<pb_byte_t *>(data), data_len);

    telemetrypayload_Notification Notification = telemetrypayload_Notification_init_zero;

    if (!pb_decode(&stream, telemetrypayload_Notification_fields, &Notification))
        return false;

    switch (Notification.which_notification) {
    case telemetrypayload_Notification_telemetry_tag:
        break;
    case telemetrypayload_Notification_ack_tag: {
        tx_mutex_get(&decode_event_mutex, TX_WAIT_FOREVER);
        if (decode_wait_seq_num_reply == Notification.notification.ack.seq_num) {
            tx_event_flags_set(&decode_event_flags,
                    PB_ACK_EVENT, TX_OR);
            }
        tx_mutex_put(&decode_event_mutex);
        break;
    }
    case telemetrypayload_Notification_nack_tag:
        tx_mutex_get(&decode_event_mutex, TX_WAIT_FOREVER);
        if (decode_wait_seq_num_reply == Notification.notification.nack.seq_num) {
            tx_event_flags_set(&decode_event_flags,
                    PB_NACK_EVENT, TX_OR);
            }
        tx_mutex_put(&decode_event_mutex);
        break;
    }

    return true;
}