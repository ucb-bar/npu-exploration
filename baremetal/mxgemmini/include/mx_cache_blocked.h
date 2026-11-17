#ifndef MX_CACHE_BLOCKED_H
#define MX_CACHE_BLOCKED_H
#include "mx_cache_lut.h"
static int mx_cache_blocked_impl(const uint8_t *kt, const uint8_t *ks, const uint8_t *v,
                                 const uint8_t *vs, int heads, int hd, int capture_keys,
                                 int capacity, int prefix, int time_tile, float *kc, float *vc) {
  if (!kt || !ks || !v || !vs || !kc || !vc || heads < 1 || hd < 32 || hd % 32 || prefix < 0 ||
      prefix > capacity || capacity > capture_keys || capture_keys % 32 ||
      (time_tile != 1 && time_tile != 16 && time_tile != 32 && time_tile != 64))
    return 1;
  mx_cache_lut table;
  mx_cache_lut_init(&table);
  for (int h = 0; h < heads; h++) {
    for (int tb = 0; tb < prefix; tb += time_tile) {
      int count = prefix - tb < time_tile ? prefix - tb : time_tile;
      for (int db = 0; db < hd; db += 32) {
        float scale[64];
        const uint8_t *ss = ks + ((size_t)h * (hd / 32) + db / 32) * capture_keys + tb;
        for (int t = 0; t < count; t++)
          scale[t] = table.scale[ss[t]];
        for (int d = db; d < db + 32; d++) {
          const uint8_t *src = kt + ((size_t)h * hd + d) * capture_keys + tb;
          float *dst = kc + ((size_t)h * capacity + tb) * hd + d;
          for (int t = 0; t < count; t++)
            dst[(size_t)t * hd] = table.element[src[t]] * scale[t];
        }
      }
    }
    for (int t = 0; t < prefix; t++) {
      const uint8_t *src = v + ((size_t)h * capture_keys + t) * hd;
      const uint8_t *sc = vs + ((size_t)h * (capture_keys / 32) + t / 32) * hd;
      float *dst = vc + ((size_t)h * capacity + t) * hd;
      for (int d = 0; d < hd; d++)
        dst[d] = table.element[src[d]] * table.scale[sc[d]];
    }
    for (int t = prefix; t < capacity; t++)
      for (int d = 0; d < hd; d++) {
        size_t out = ((size_t)h * capacity + t) * hd + d;
        kc[out] = NAN;
        vc[out] = NAN;
      }
  }
  return 0;
}
/* Caller provides disjoint complete input/output spans. Time16 is an opt-in
 * layout transform; profile adoption is independently gated by full hardware. */
static inline int mx_cache_init_blocked(const uint8_t *kt, const uint8_t *ks, const uint8_t *v,
                                        const uint8_t *vs, int heads, int hd, int capture_keys,
                                        int capacity, int prefix, int time_tile, float *kc,
                                        float *vc) {
  if (!kt || !ks || !v || !vs || !kc || !vc || heads < 1 || hd < 32 || hd % 32 ||
      capture_keys < 1 || capture_keys % 32 || capacity < 1 || capacity > capture_keys ||
      prefix < 0 || prefix > capacity || time_tile != 16)
    return -1;
  if ((size_t)heads > SIZE_MAX / (size_t)hd / (size_t)capture_keys ||
      (size_t)heads > SIZE_MAX / sizeof(float) / (size_t)hd / (size_t)capacity)
    return -1;
  return mx_cache_blocked_impl(kt, ks, v, vs, heads, hd, capture_keys, capacity, prefix, time_tile,
                               kc, vc)
             ? -1
             : 0;
}

#endif
