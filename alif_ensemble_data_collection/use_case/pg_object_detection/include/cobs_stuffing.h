/*
 * cobs_stuffing.h
 *
 *  Created on: Feb 20, 2024
 *      Author: avoloshyn
 */

#ifndef PG_COBS_STUFFING_H_
#define PG_COBS_STUFFING_H_

#include <stddef.h>
#include <stdint.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t cobsEncode(const void *data, size_t length, uint8_t *buffer);
size_t cobsDecode(const uint8_t *buffer, size_t length, void *data);

#ifdef __cplusplus
}
#endif

#endif /* PG_COBS_STUFFING_H_ */
