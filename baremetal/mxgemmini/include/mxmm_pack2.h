#ifndef MXMM_PACK2_H
#define MXMM_PACK2_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
/* Exact integer RNE conversion of BF16 * 2^-block_exponent. The target codes
 * use direct E2M1, E3M2, E4M3FN and preserve signed zero like mxmm_encode.
 * Finite inputs never produce E4M3's reserved magnitude127 code. */
static inline uint8_t mxmm_pack2_encode(uint16_t bits, int block_exponent, int f) {
  unsigned sign = (bits >> 15) << (f == 4 ? 3 : f == 6 ? 5 : 7), mag = bits & 0x7fff;
  unsigned be = mag >> 7, frac = mag & 127,
           mb = f == 4   ? 1
                : f == 6 ? 2
                         : 3,
           bias = f == 4   ? 1
                  : f == 6 ? 3
                           : 7;
  unsigned maximum = f == 4 ? 7 : f == 6 ? 31 : 126;
  if (!mag) {
#ifdef __FAST_MATH__
    return 0; /* Match the fast-math signbit convention for scaled negative zero. */
#else
    return (uint8_t)sign;
#endif
  }
  if (be == 255)
    return (uint8_t)(sign | (frac ? 0 : maximum)); /* matches old encode */
  int e;
  unsigned sig;
  if (be) {
    e = (int)be - 127 - block_exponent;
    sig = 128 + frac;
  } else {
    int h = 31 - __builtin_clz(frac);
    e = h - 133 - block_exponent;
    sig = frac << (7 - h);
  }
#ifdef __FAST_MATH__
  /* Only fast-math signed-zero handling observes FP32 scaling underflow.
  * Preserve the frozen platform libm behavior in this extremely small tail;
  * ordinary activation conversion remains entirely integer. */
  if (sign && e < -126) {
    uint32_t raw = (uint32_t)bits << 16;
    float x;
    memcpy(&x, &raw, 4);
    if (ldexpf(x, -block_exponent) == 0.0f)
      sign = 0;
  }
#endif
  int emin = 1 - (int)bias;
  unsigned q, shift;
  if (e >= emin) {
    shift = 7 - mb;
    q = sig >> shift;
    unsigned rem = sig & ((1u << shift) - 1), half = 1u << (shift - 1);
    q += (rem > half) || (rem == half && (q & 1));
    if (q == (1u << (mb + 1))) {
      q >>= 1;
      e++;
    }
    unsigned code = ((unsigned)(e + (int)bias) << mb) + (q - (1u << mb));
    if (code > maximum)
      code = maximum;
    return (uint8_t)(sign | code);
  }
  shift = 7 + emin - (int)mb - e;
  if (shift > 8)
    return (uint8_t)sign;
  q = sig >> shift;
  unsigned rem = sig & ((1u << shift) - 1), half = 1u << (shift - 1);
  q += (rem > half) || (rem == half && (q & 1));
  return (uint8_t)(sign | q);
}
/* The existing BF16 API chooses floor(log2(amax)) with no epsilon floor and
 * exponent0 for an all-zero block. Preserve its scale-byte cast even for tiny
 * subnormal-only blocks; actual model inputs have ordinary legal E8M0 scales. */
static inline int mxmm_pack2_blockexp(const uint16_t *x) {
  unsigned maximum = 0;
  for (int j = 0; j < 32; j++) {
    unsigned m = x[j] & 0x7fff;
    if (m > maximum)
      maximum = m;
  }
  if (!maximum)
    return 0;
  unsigned e = maximum >> 7;
  if (e)
    return (int)e - 127;
  return (31 - __builtin_clz(maximum)) - 133;
}
typedef uint64_t mxmm_pack2_u64 __attribute__((may_alias));
/* All legal spans have length multiple32. Fast aligned path uses unrolled
 * word copies; unaligned callers keep a byte-safe path. */
static inline void mxmm_pack2_copy(uint8_t *dst, const uint8_t *src, size_t bytes) {
  if (!(((uintptr_t)dst | (uintptr_t)src) & 7)) {
    mxmm_pack2_u64 *d = (mxmm_pack2_u64 *)dst;
    const mxmm_pack2_u64 *s = (const mxmm_pack2_u64 *)src;
    for (size_t j = 0; j < bytes / 8; j += 4) {
      d[j] = s[j];
      d[j + 1] = s[j + 1];
      d[j + 2] = s[j + 2];
      d[j + 3] = s[j + 3];
    }
  } else
    for (size_t j = 0; j < bytes; j++)
      dst[j] = src[j];
}
/* GENERATED_LUT_START */
#ifdef __FAST_MATH__
#define MXMM_PACK2_NZ4 0
#define MXMM_PACK2_NZ6 0
#else
#define MXMM_PACK2_NZ4 8
#define MXMM_PACK2_NZ6 32
#endif
static const uint8_t mxmm_pack2_lut[2][256] = {
    {0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  1,
     1,  1,  1,
     1,  1,  1,
     1,  1,  1,
     1,  2,  2,
     2,  2,  2,
     2,  2,  3,
     3,  3,  4,
     4,  4,  4,
     4,  5,  5,
     5,  6,  6,
     6,  6,  6,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  7,
     7,  7,  MXMM_PACK2_NZ4,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  8,  8,
     8,  9,  9,
     9,  9,  9,
     9,  9,  9,
     9,  9,  9,
     10, 10, 10,
     10, 10, 10,
     10, 11, 11,
     11, 12, 12,
     12, 12, 12,
     13, 13, 13,
     14, 14, 14,
     14, 14, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15, 15, 15,
     15},
    {0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  1,
     1,  1,  1,
     1,  1,  1,
     1,  1,  1,
     1,  2,  2,
     2,  2,  2,
     2,  2,  3,
     3,  3,  4,
     4,  4,  4,
     5,  6,  6,
     6,  7,  8,
     8,  8,  9,
     10, 10, 10,
     11, 12, 12,
     12, 13, 14,
     14, 14, 15,
     16, 16, 16,
     17, 18, 18,
     18, 19, 20,
     20, 20, 21,
     22, 22, 22,
     23, 24, 24,
     24, 25, 26,
     26, 26, 27,
     28, 28, 28,
     29, 30, 30,
     30, 31, 31,
     31, 31, 31,
     31, 31, 31,
     31, 31, 31,
     31, 31, 31,
     31, 31, 31,
     31, 31, 31,
     31, 31, 31,
     31, 31, 31,
     31, 31, 31,
     31, 31, 31,
     31, 31, MXMM_PACK2_NZ6,
     32, 32, 32,
     32, 32, 32,
     32, 32, 32,
     32, 32, 32,
     32, 32, 32,
     32, 33, 33,
     33, 33, 33,
     33, 33, 33,
     33, 33, 33,
     34, 34, 34,
     34, 34, 34,
     34, 35, 35,
     35, 36, 36,
     36, 36, 37,
     38, 38, 38,
     39, 40, 40,
     40, 41, 42,
     42, 42, 43,
     44, 44, 44,
     45, 46, 46,
     46, 47, 48,
     48, 48, 49,
     50, 50, 50,
     51, 52, 52,
     52, 53, 54,
     54, 54, 55,
     56, 56, 56,
     57, 58, 58,
     58, 59, 60,
     60, 60, 61,
     62, 62, 62,
     63, 63, 63,
     63, 63, 63,
     63, 63, 63,
     63, 63, 63,
     63, 63, 63,
     63, 63, 63,
     63, 63, 63,
     63, 63, 63,
     63, 63, 63,
     63, 63, 63,
     63, 63, 63,
     63},
};
static inline const uint8_t *mxmm_pack2_source(const uint8_t *in, int rows, int k, int block, int r,
                                               int col) {
  return in + (block ? (size_t)(col / block) * rows * block + (size_t)r * block + col % block
                     : (size_t)r * k + col);
}
static void mxmm_pack2_group(const void *input, const uint8_t *sc, int rows, int m, int k,
                             int block, int g, int kb, int nb, uint8_t *out, uint8_t *as) {
  int kg = nb * 32;
  if (!kg)
    return;
  if (rows < m) {
    for (int b = 0; b < nb; b++)
      memset(as + (kb + b) * m + rows, 127, m - rows);
  }
  if (sc) {
    if (rows == m)
      mxmm_pack2_copy(as + (size_t)kb * m, sc + (size_t)kb * rows, (size_t)nb * rows);
    else
      for (int b = 0; b < nb; b++)
        memcpy(as + (kb + b) * m, sc + (kb + b) * rows, rows);
    if (g == 0) {
      for (int r = 0; r < rows; r += 2)
        for (int off = 0; off < kg;) {
          int col = kb * 32 + off, n = block ? block - col % block : kg - off;
          if (n > kg - off)
            n = kg - off;
          const uint8_t *a = mxmm_pack2_source((const uint8_t *)input, rows, k, block, r, col);
          const uint8_t *b =
              r + 1 < rows ? mxmm_pack2_source((const uint8_t *)input, rows, k, block, r + 1, col)
                           : NULL;
          uint8_t *dst = out + (size_t)(r / 2) * kg + off;
          if (b)
            for (int j = 0; j < n; j++)
              dst[j] = mxmm_pack2_lut[0][a[j]] | (mxmm_pack2_lut[0][b[j]] << 4);
          else
            for (int j = 0; j < n; j++)
              dst[j] = mxmm_pack2_lut[0][a[j]];
          off += n;
        }
    } else
      for (int r = 0; r < rows; r++)
        for (int off = 0; off < kg;) {
          int col = kb * 32 + off, n = block ? block - col % block : kg - off;
          if (n > kg - off)
            n = kg - off;
          const uint8_t *src = mxmm_pack2_source((const uint8_t *)input, rows, k, block, r, col);
          uint8_t *dst = out + (size_t)r * kg + off;
          if (g == 2)
            mxmm_pack2_copy(dst, src, n);
          else
            for (int j = 0; j < n; j++)
              dst[j] = mxmm_pack2_lut[1][src[j]];
          off += n;
        }
  } else if (g == 0) {
    for (int r = 0; r < rows; r += 2)
      for (int b = 0; b < nb; b++) {
        const uint16_t *a = (const uint16_t *)input + (size_t)r * k + (kb + b) * 32;
        const uint16_t *z = r + 1 < rows ? a + k : NULL;
        int ea = mxmm_pack2_blockexp(a), ez = z ? mxmm_pack2_blockexp(z) : 0;
        as[(kb + b) * m + r] = (uint8_t)(ea + 127);
        if (z)
          as[(kb + b) * m + r + 1] = (uint8_t)(ez + 127);
        uint8_t *dst = out + (size_t)(r / 2) * kg + b * 32;
        if (z)
          for (int j = 0; j < 32; j++)
            dst[j] = mxmm_pack2_encode(a[j], ea, 4) | (mxmm_pack2_encode(z[j], ez, 4) << 4);
        else
          for (int j = 0; j < 32; j++)
            dst[j] = mxmm_pack2_encode(a[j], ea, 4);
      }
  } else
    for (int r = 0; r < rows; r++)
      for (int b = 0; b < nb; b++) {
        const uint16_t *a = (const uint16_t *)input + (size_t)r * k + (kb + b) * 32;
        int e = mxmm_pack2_blockexp(a);
        as[(kb + b) * m + r] = (uint8_t)(e + 127);
        uint8_t *dst = out + (size_t)r * kg + b * 32;
        if (g == 1)
          for (int j = 0; j < 32; j++)
            dst[j] = mxmm_pack2_encode(a[j], e, 6);
        else
          for (int j = 0; j < 32; j++)
            dst[j] = mxmm_pack2_encode(a[j], e, 8);
      }
  if (rows < m) {
    int valid = g == 0 ? (rows + 1) / 2 : rows, total = g == 0 ? m / 2 : m;
    memset(out + (size_t)valid * kg, 0, (size_t)(total - valid) * kg);
  }
}

#endif
