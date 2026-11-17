#ifndef MX_MICROMIX_NUMERIC_H
#define MX_MICROMIX_NUMERIC_H
/* Full-vector BF16 comparison, also suitable for a host test compiled with
 * -ffast-math. Inspect exponent bits before converting or doing arithmetic. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef struct {
  uint64_t relative_l2_ppm;
  size_t bit_mismatches;
  int finite, pass;
} mxmm_numeric_result;
static inline int mxmm_numeric_finite(uint16_t x) { return (x & 0x7f80u) != 0x7f80u; }
static inline double mxmm_numeric_value(uint16_t x) {
  /* Build a normal binary64 value directly. A BF16 subnormal is a binary32
   * subnormal; casting through float would lose it under host DAZ/FTZ with
   * -ffast-math. Even the smallest BF16 remains normal in binary64. */
  unsigned e = (x >> 7) & 255u, m = x & 127u;
  uint64_t u = (uint64_t)(x & 0x8000u) << 48;
  if (e)
    u |= (uint64_t)(e + 896u) << 52 | (uint64_t)m << 45;
  else if (m) {
    unsigned p = 31u - (unsigned)__builtin_clz(m);
    u |= (uint64_t)(p + 890u) << 52 | (uint64_t)(m - (1u << p)) << (52u - p);
  }
  double d;
  memcpy(&d, &u, sizeof(d));
  return d;
}
/* hpre!=NULL compares residual updates (got-hpre) and (ref-hpre).
 * Their difference is got-ref; the denominator is specifically ref-hpre.
 * Zero reference norm passes only if the complete numeric difference is zero.
 * Signed-zero differences remain visible in bit_mismatches but have zero norm. */
static mxmm_numeric_result mxmm_numeric_compare(const uint16_t *got, const uint16_t *ref,
                                                const uint16_t *hpre, size_t n) {
  mxmm_numeric_result r = {0, 0, 1, 0};
  double error2 = 0, reference2 = 0;
  for (size_t i = 0; i < n; i++) {
    r.bit_mismatches += got[i] != ref[i];
    if (!mxmm_numeric_finite(got[i]) || !mxmm_numeric_finite(ref[i]) ||
        (hpre && !mxmm_numeric_finite(hpre[i]))) {
      r.finite = 0;
      continue;
    }
    double a = mxmm_numeric_value(got[i]), b = mxmm_numeric_value(ref[i]);
    double d = a - b, v = hpre ? b - mxmm_numeric_value(hpre[i]) : b;
    error2 += d * d;
    reference2 += v * v;
  }
  if (!r.finite) {
    r.relative_l2_ppm = UINT64_MAX;
    return r;
  }
  if (reference2 == 0) {
    r.pass = error2 == 0;
    r.relative_l2_ppm = r.pass ? 0 : UINT64_MAX;
    return r;
  }
  r.pass = error2 <= reference2 * 0.0001;
  double ppm = sqrt(error2 / reference2) * 1000000.0;
  r.relative_l2_ppm = ppm >= 18446744073709551615.0 ? UINT64_MAX : (uint64_t)(ppm + 0.5);
  return r;
}
#endif
