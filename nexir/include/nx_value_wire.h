#ifndef NEXIR_NX_VALUE_WIRE_H
#define NEXIR_NX_VALUE_WIRE_H

/* Portable scalar wire format. It never serializes process-local pointers.
 * Bytes 0..3: little-endian tag. Bytes 4..7: zero. Bytes 8..15: payload.
 * This header needs C99 uint64_t and an IEEE-754 binary64 double for f64.
 */
#include <stdint.h>
#include <string.h>
#include <float.h>

enum { NX_WIRE_SIZE = 16, NX_TAG_NULL = 0, NX_TAG_I64 = 1, NX_TAG_F64 = 2 };

static inline void nx_wire_put_u64(uint8_t *out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out[i] = (uint8_t)(value >> (8u * i));
}
static inline uint64_t nx_wire_get_u64(const uint8_t *in) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= ((uint64_t)in[i]) << (8u * i);
    return value;
}
static inline void nx_wire_i64(uint8_t out[NX_WIRE_SIZE], int64_t value) {
    memset(out, 0, NX_WIRE_SIZE);
    out[0] = NX_TAG_I64;
    uint64_t bits;
    memcpy(&bits, &value, 8);
    nx_wire_put_u64(out + 8, bits);
}
static inline int nx_wire_read_i64(const uint8_t in[NX_WIRE_SIZE], int64_t *value) {
    if (in[0] != NX_TAG_I64 || in[1] || in[2] || in[3] || in[4] || in[5] || in[6] || in[7]) return 0;
    uint64_t bits = nx_wire_get_u64(in + 8);
    memcpy(value, &bits, 8);
    return 1;
}
static inline int nx_wire_f64(uint8_t out[NX_WIRE_SIZE], double value) {
    if (sizeof(double) != 8 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024) return 0;
    memset(out, 0, NX_WIRE_SIZE);
    out[0] = NX_TAG_F64;
    uint64_t bits;
    memcpy(&bits, &value, 8);
    nx_wire_put_u64(out + 8, bits);
    return 1;
}
static inline int nx_wire_read_f64(const uint8_t in[NX_WIRE_SIZE], double *value) {
    if (sizeof(double) != 8 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || in[0] != NX_TAG_F64 || in[1] || in[2] || in[3] || in[4] || in[5] || in[6] || in[7]) return 0;
    uint64_t bits = nx_wire_get_u64(in + 8);
    memcpy(value, &bits, 8);
    return 1;
}

#endif
