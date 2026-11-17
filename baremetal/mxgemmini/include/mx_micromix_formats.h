#ifndef MX_MICROMIX_FORMATS_H
#define MX_MICROMIX_FORMATS_H
#include <stdint.h>
#include <math.h>
static inline float mxmm_decode(unsigned c, int f) {
  if (f == 4) {
    static const float v[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return (c & 8) ? -v[c & 7] : v[c & 7];
  }
  int mb = f == 6 ? 2 : 3, eb = f == 6 ? 3 : 4, bias = f == 6 ? 3 : 7;
  unsigned mag = c & ((1u << (mb + eb)) - 1), e = mag >> mb, q = mag & ((1u << mb) - 1);
  float v = e ? ldexpf(1.0f + q / (float)(1 << mb), (int)e - bias)
              : ldexpf(q / (float)(1 << mb), 1 - bias);
  return (c & (1u << (mb + eb))) ? -v : v;
}
static inline uint8_t mxmm_encode(float x, int f) {
  unsigned sign = signbit(x) ? 1u << (f == 4 ? 3 : f == 6 ? 5 : 7) : 0;
  float v = fabsf(x);
  unsigned hi = f == 4 ? 7 : f == 6 ? 31 : 126, lo = 0;
  while (lo < hi) {
    unsigned mid = (lo + hi) / 2;
    if (mxmm_decode(mid, f) < v)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo) {
    float a = v - mxmm_decode(lo - 1, f), b = mxmm_decode(lo, f) - v;
    if (a < b || (a == b && !((lo - 1) & 1)))
      --lo;
  }
  return (uint8_t)(lo | sign);
}
#endif
