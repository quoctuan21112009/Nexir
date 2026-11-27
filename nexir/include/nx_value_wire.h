#ifndef NEXIR_NX_VALUE_WIRE_H
#define NEXIR_NX_VALUE_WIRE_H

/* Portable scalar wire format. It never serializes process-local pointers.
 * Bytes 0..3: little-endian tag. Bytes 4..7: flags/zero. Bytes 8..23 (or 8..15 for 16B): payload.
 * Scalar wire records remain 16 bytes for NULL/I64/F64/BOOL; extended records use NX_WIRE_EXT_SIZE (24).
 */
#include <stdint.h>
#include <string.h>
#include <float.h>

enum {
    NX_WIRE_SIZE = 16,
    NX_WIRE_EXT_SIZE = 24,
    NX_TAG_NULL = 0,
    NX_TAG_I64 = 1,
    NX_TAG_F64 = 2,
    NX_TAG_BOOL = 3,
    NX_TAG_I128 = 4,
    NX_TAG_U128 = 5
};

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
static inline void nx_wire_bool(uint8_t out[NX_WIRE_SIZE], int value) {
    memset(out, 0, NX_WIRE_SIZE);
    out[0] = NX_TAG_BOOL;
    nx_wire_put_u64(out + 8, value ? 1ULL : 0ULL);
}
static inline int nx_wire_read_bool(const uint8_t in[NX_WIRE_SIZE], int *value) {
    if (in[0] != NX_TAG_BOOL || in[1] || in[2] || in[3] || in[4] || in[5] || in[6] || in[7]) return 0;
    *value = nx_wire_get_u64(in + 8) ? 1 : 0;
    return 1;
}
static inline void nx_wire_i128(uint8_t out[NX_WIRE_EXT_SIZE], uint64_t lo, int64_t hi) {
    memset(out, 0, NX_WIRE_EXT_SIZE);
    out[0] = NX_TAG_I128;
    nx_wire_put_u64(out + 8, lo);
    uint64_t hi_bits;
    memcpy(&hi_bits, &hi, 8);
    nx_wire_put_u64(out + 16, hi_bits);
}
static inline int nx_wire_read_i128(const uint8_t in[NX_WIRE_EXT_SIZE], uint64_t *lo, int64_t *hi) {
    if (in[0] != NX_TAG_I128 || in[1] || in[2] || in[3] || in[4] || in[5] || in[6] || in[7]) return 0;
    *lo = nx_wire_get_u64(in + 8);
    uint64_t hi_bits = nx_wire_get_u64(in + 16);
    memcpy(hi, &hi_bits, 8);
    return 1;
}

#endif
