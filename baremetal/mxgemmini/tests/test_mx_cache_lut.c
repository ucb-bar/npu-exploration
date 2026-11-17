#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "mx_cache_lut.h"
#include "mx_cache_blocked.h"
#define H 2
#define HD 64
#define CK 64
#define CAP 37
static uint8_t kt[H * HD * CK], ks[H * (HD / 32) * CK], vv[H * CK * HD], vs[H * (CK / 32) * HD];
static float kc[H * CAP * HD + 16], vc[H * CAP * HD + 16], bk[H * CAP * HD + 16],
    bv[H * CAP * HD + 16];
static int checks, failed;
static uint64_t hash = 1469598103934665603ULL;
static uint32_t bits(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return u;
}
/* Transport checksum canonicalizes only NaN payload/sign; raw comparisons below remain exact to the original decode on this processor. */
static uint32_t checksum_bits(float x) {
  uint32_t u = bits(x);
  return (u & 0x7fffffff) > 0x7f800000 ? 0x7fc00000 : u;
}
static void check(int ok) {
  checks++;
  failed += !ok;
}
int main(void) {
  mx_cache_lut table;
  mx_cache_lut_init(&table);
  for (int c = 0; c < 256; c++)
    for (int s = 0; s < 256; s++) {
      float got = mx_cache_lut_value(&table, (uint8_t)c, (uint8_t)s),
            ref = mx_e4m3_decode((uint8_t)c) * mx_e8m0_decode((uint8_t)s);
      check(bits(got) == bits(ref));
      hash = (hash ^ checksum_bits(got)) * 1099511628211ULL;
    }
  for (size_t i = 0; i < sizeof(kt); i++) {
    kt[i] = (uint8_t)(i * 17);
    vv[i] = (uint8_t)(i * 23);
  }
  for (size_t i = 0; i < sizeof(ks); i++)
    ks[i] = (uint8_t)(122 + i % 11);
  for (size_t i = 0; i < sizeof(vs); i++)
    vs[i] = (uint8_t)(122 + i % 11);
  int prefixes[] = {0, 1, 17, 33, 37};
  for (int p = 0; p < 5; p++) {
    int prefix = prefixes[p];
    for (int i = 0; i < H * CAP * HD + 16; i++) {
      uint32_t z = 0x4a55aa55;
      memcpy(kc + i, &z, 4);
      memcpy(vc + i, &z, 4);
    }
    check(mx_cache_init_lut(kt, ks, vv, vs, H, HD, CK, CAP, prefix, kc, vc) == 0);
    for (int i = 0; i < H * CAP * HD + 16; i++) {
      uint32_t z = 0x4a55aa55;
      memcpy(bk + i, &z, 4);
      memcpy(bv + i, &z, 4);
    }
    check(mx_cache_init_blocked(kt, ks, vv, vs, H, HD, CK, CAP, prefix, 16, bk, bv) == 0);
    for (int i = 0; i < H * CAP * HD; i++) {
      check(bits(kc[i]) == bits(bk[i]));
      check(bits(vc[i]) == bits(bv[i]));
    }
    for (int i = H * CAP * HD; i < H * CAP * HD + 16; i++)
      check(bits(bk[i]) == 0x4a55aa55 && bits(bv[i]) == 0x4a55aa55);

    for (int h = 0; h < H; h++)
      for (int t = 0; t < CAP; t++)
        for (int d = 0; d < HD; d++) {
          size_t i = ((size_t)h * CAP + t) * HD + d;
          float a = t < prefix ? mx_e4m3_decode(kt[((size_t)h * HD + d) * CK + t]) *
                                     mx_e8m0_decode(ks[((size_t)h * (HD / 32) + d / 32) * CK + t])
                               : NAN;
          float b = t < prefix ? mx_e4m3_decode(vv[((size_t)h * CK + t) * HD + d]) *
                                     mx_e8m0_decode(vs[((size_t)h * (CK / 32) + t / 32) * HD + d])
                               : NAN;
          check(bits(kc[i]) == bits(a));
          check(bits(vc[i]) == bits(b));
          hash = (hash ^ checksum_bits(kc[i])) * 1099511628211ULL;
          hash = (hash ^ checksum_bits(vc[i])) * 1099511628211ULL;
        }
    for (int i = H * CAP * HD; i < H * CAP * HD + 16; i++)
      check(bits(kc[i]) == 0x4a55aa55 && bits(vc[i]) == 0x4a55aa55);
  }
  uint32_t before = bits(kc[0]);
  check(mx_cache_init_lut(NULL, ks, vv, vs, H, HD, CK, CAP, 17, kc, vc) == -1);
  check(mx_cache_init_lut(kt, ks, vv, vs, H, HD - 1, CK, CAP, 17, kc, vc) == -1);
  check(mx_cache_init_lut(kt, ks, vv, vs, H, HD, CK, CAP, CAP + 1, kc, vc) == -1);
  check(bits(kc[0]) == before);
  check(mx_cache_init_blocked(kt, ks, vv, vs, H, HD, CK, CAP, 17, 32, bk, bv) == -1);
  check(mx_cache_init_blocked(NULL, ks, vv, vs, H, HD, CK, CAP, 17, 16, bk, bv) == -1);
  printf("MX_CACHE_TEST %s checks=%d checksum=%llu\n", failed ? "FAIL" : "PASS", checks,
         (unsigned long long)hash);
  return failed != 0;
}
