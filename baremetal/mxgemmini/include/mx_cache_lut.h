#ifndef MX_CACHE_LUT_H
#define MX_CACHE_LUT_H
/* Lookup tables for the existing FP8 element and E8M0 scale decoders.
 * Construct during each cache reset so initialization timing includes it. */
#include "mx_host.h"
#ifdef __FAST_MATH__
#error "Cache LUT validation and use require strict FP"
#endif
typedef struct {
  float element[256], scale[256];
} llama7b_cache_lut;
static void llama7b_cache_lut_init(llama7b_cache_lut *table) {
  for (int i = 0; i < 256; i++) {
    table->element[i] = mx_e4m3_decode((uint8_t)i);
    table->scale[i] = mx_e8m0_decode((uint8_t)i);
  }
}
static inline float llama7b_cache_value(const llama7b_cache_lut *table, uint8_t code,
                                        uint8_t scale) {
  return table->element[code] * table->scale[scale];
}
static void llama7b_init_cache_lut(const uint8_t *kt, const uint8_t *ks, const uint8_t *v,
                                   const uint8_t *vs, int heads, int hd, int capture_keys,
                                   int capacity, int prefix, float *kc, float *vc) {
  llama7b_cache_lut table;
  llama7b_cache_lut_init(&table);
  for (int h = 0; h < heads; h++)
    for (int t = 0; t < capacity; t++)
      for (int d = 0; d < hd; d++) {
        size_t out = ((size_t)h * capacity + t) * hd + d;
        kc[out] = t < prefix
                      ? llama7b_cache_value(&table, kt[((size_t)h * hd + d) * capture_keys + t],
                                            ks[((size_t)h * (hd / 32) + d / 32) * capture_keys + t])
                      : NAN;
        vc[out] = t < prefix
                      ? llama7b_cache_value(&table, v[((size_t)h * capture_keys + t) * hd + d],
                                            vs[((size_t)h * (capture_keys / 32) + t / 32) * hd + d])
                      : NAN;
      }
}
/* Public checked adapter. Inputs and output mirrors must not overlap. */
typedef llama7b_cache_lut mx_cache_lut;
static inline void mx_cache_lut_init(mx_cache_lut *table) { llama7b_cache_lut_init(table); }
static inline float mx_cache_lut_value(const mx_cache_lut *table, uint8_t code, uint8_t scale) {
  return llama7b_cache_value(table, code, scale);
}
static inline int mx_cache_init_lut(const uint8_t *kt, const uint8_t *ks, const uint8_t *v,
                                    const uint8_t *vs, int heads, int hd, int capture_keys,
                                    int capacity, int prefix, float *kc, float *vc) {
  if (!kt || !ks || !v || !vs || !kc || !vc || heads < 1 || hd < 32 || hd % 32 ||
      capture_keys < 1 || capture_keys % 32 || capacity < 1 || capacity > capture_keys ||
      prefix < 0 || prefix > capacity)
    return -1;
  if ((size_t)heads > SIZE_MAX / (size_t)hd / (size_t)capture_keys ||
      (size_t)heads > SIZE_MAX / sizeof(float) / (size_t)hd / (size_t)capacity)
    return -1;
  llama7b_init_cache_lut(kt, ks, v, vs, heads, hd, capture_keys, capacity, prefix, kc, vc);
  return 0;
}
#endif
