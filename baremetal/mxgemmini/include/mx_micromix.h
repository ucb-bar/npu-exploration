#ifndef MX_MICROMIX_H
#define MX_MICROMIX_H
/* Direct FP4/6/8 projection, DIM16 shared accumulator. All groups are contiguous
 * K32 blocks, issued FP4 then FP6 then FP8. Workspace is M*(K+K/32) bytes.
 * The fp8 entry requantizes the VPU's E4M3 result using its existing E8M0 scale;
 * bf16 entry determines one power-of-two scale per actual row and K32 block.
 * Output, scratch, packed weight groups and weight scales require 64-byte alignment.
 * Packing is synchronous and part of the caller's projection timing. */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "include/gemmini.h"
#include "mx_native_state.h"
#include "mxmm_pack2.h"
#define MXMM_HAS_ROUND2_PACK 1
static int mxmm_round2_pack = 0;
/* Descriptive alias; legacy spelling remains source compatible. Default off. */
#define mxmm_integer_pack mxmm_round2_pack
/* Local compiler barriers: frozen rocc helpers do not declare memory effects. */
static inline uint64_t mxmm_cycles(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x)::"memory");
  return x;
}
static inline void mxmm_device_fence(void) { asm volatile("fence" ::: "memory"); }
typedef struct {
  int k, n, nb4, nb6, nb8;
  const uint8_t *b4, *b6, *b8, *scales;
} mxmm_weight;
typedef struct {
  int nc, run, fence_groups;
} mxmm_schedule;
/* Aggregate counters let a complete layer expose CPU packing separately. */
static uint64_t mxmm_pack_cycles, mxmm_mesh_cycles;
#define MXMM_HAS_FAST_PACK 1
static int mxmm_fast_pack = 1;
static const mxmm_schedule mxmm_default_schedule = {0, 16, 1};
static inline size_t mxmm_workspace_bytes(int m, int k) { return (size_t)m * (k + k / 32); }
#include "mx_micromix_formats.h"
static inline int mxmm_legal(int m, const mxmm_weight *w, mxmm_schedule s, int *nc_out) {
  if (!w || DIM < 16 || m < 1 || w->k < 32 || w->n < 2 * DIM || m % (2 * DIM) || w->k % 32 ||
      w->n % (2 * DIM) || w->nb4 < 0 || w->nb6 < 0 || w->nb8 < 0 ||
      w->nb4 + w->nb6 + w->nb8 != w->k / 32 || s.run < 1)
    return 0;
  for (int nc = s.nc ? s.nc : 512; nc >= 2 * DIM; nc = s.nc ? 0 : nc / 2) {
    if (nc % (2 * DIM) || w->n % nc || (size_t)m * nc / (2 * DIM) > ACC_ROWS / 2 ||
        (size_t)(m + nc) * 32 * s.run / DIM > MXN_HALF_ROWS || (size_t)s.run * m > 4096 ||
        (size_t)s.run * nc > 4096)
      continue;
    *nc_out = nc;
    return 1;
  }
  return 0;
}
/* fp8 input may be row major (block=0) or [K/block][M][block].
 * Its scales are always [K/32][M]. bf16 input is [rows][K]. */
static int mxmm_project_impl(const void *input, const uint8_t *input_scales, int rows, int m,
                             int block, const mxmm_weight *w, uint16_t *out, uint8_t *scratch,
                             mxmm_schedule s) {
  int nc;
  if (!input || !out || !scratch || rows < 1 || rows > m || !mxmm_legal(m, w, s, &nc))
    return 1;
  if (input_scales && block && (block < 32 || block % 32))
    return 1;
  if (!w->scales || (w->nb4 && !w->b4) || (w->nb6 && !w->b6) || (w->nb8 && !w->b8))
    return 1;
  /* The RTL DMA/scale path requires aligned base pointers; Spike accepts
   * misalignment and therefore cannot qualify this property on its own. */
  if (((uintptr_t)out | (uintptr_t)scratch | (uintptr_t)w->scales | (uintptr_t)w->b4 |
       (uintptr_t)w->b6 | (uintptr_t)w->b8) &
      63u)
    return 1;
  const uint64_t pack_start = mxmm_cycles();
  uint8_t *as = scratch + (size_t)m * w->k, *ag[3];
  int ns[] = {w->nb4, w->nb6, w->nb8}, fs[] = {4, 6, 8}, kb = 0;
  size_t off = 0;
  const uint8_t *bs[] = {w->b4, w->b6, w->b8};
  /* The two lookup tables make FP8 -> mixed conversion independent of libm. */
  uint8_t lut[2][256];
  static uint8_t fast_lut[2][256];
  static unsigned fast_lut_valid;
  const uint8_t(*lookup)[256] = mxmm_fast_pack ? fast_lut : lut;
  if (input_scales && !mxmm_round2_pack)
    for (int j = 0; j < 2; j++) {
      if (mxmm_fast_pack && (!ns[j] || (fast_lut_valid & (1u << j))))
        continue;
      for (int c = 0; c < 256; c++)
        (mxmm_fast_pack ? fast_lut : lut)[j][c] = mxmm_encode(mxmm_decode(c, 8), fs[j]);
      if (mxmm_fast_pack)
        fast_lut_valid |= 1u << j;
    }
  if (mxmm_fast_pack && !mxmm_round2_pack)
    __builtin_memset(as, 127, (size_t)m * (w->k / 32));
  for (int g = 0; g < 3; g++) {
    int kg = ns[g] * 32;
    ag[g] = scratch + off;
    off += (size_t)m * kg / (g == 0 ? 2 : 1);
    if (mxmm_round2_pack) {
      mxmm_pack2_group(input, input_scales, rows, m, w->k, block, g, kb, ns[g], ag[g], as);
    } else if (mxmm_fast_pack) {
      if (rows < m)
        __builtin_memset(ag[g], 0, (size_t)m * kg / (g == 0 ? 2 : 1));
      for (int b = 0; b < ns[g]; b++)
        for (int r = 0; r < rows; r++) {
          const int k0 = (kb + b) * 32;
          int exponent = 0;
          const uint8_t *src = NULL;
          if (input_scales) {
            as[(kb + b) * m + r] = input_scales[(kb + b) * rows + r];
            src = (const uint8_t *)input +
                  (block ? (size_t)(k0 / block) * rows * block + (size_t)r * block + k0 % block
                         : (size_t)r * w->k + k0);
            if (g == 2) {
              __builtin_memcpy(ag[g] + (size_t)r * kg + b * 32, src, 32);
              continue;
            }
          } else {
            float amax = 0;
            for (int j = 0; j < 32; j++) {
              float x;
              uint32_t u = (uint32_t)((const uint16_t *)input)[(size_t)r * w->k + k0 + j] << 16;
              __builtin_memcpy(&x, &u, 4);
              if (fabsf(x) > amax)
                amax = fabsf(x);
            }
            if (amax > 0) {
              frexpf(amax, &exponent);
              --exponent;
            }
            as[(kb + b) * m + r] = (uint8_t)(exponent + 127);
          }
          uint8_t *dst = ag[g] + (size_t)(g == 0 ? r / 2 : r) * kg + b * 32;
          for (int j = 0; j < 32; j++) {
            uint8_t c;
            if (input_scales)
              c = lookup[g][src[j]];
            else {
              float x;
              uint32_t u = (uint32_t)((const uint16_t *)input)[(size_t)r * w->k + k0 + j] << 16;
              __builtin_memcpy(&x, &u, 4);
              c = mxmm_encode(ldexpf(x, -exponent), fs[g]);
            }
            if (g == 0) {
              if (r & 1)
                dst[j] |= c << 4;
              else
                dst[j] = c;
            } else
              dst[j] = c;
          }
        }
    } else
      for (int b = 0; b < ns[g]; b++)
        for (int r = 0; r < m; r++) {
          int exponent = 0;
          if (r < rows) {
            if (input_scales)
              exponent = (int)input_scales[(kb + b) * rows + r] - 127;
            else {
              float amax = 0;
              for (int j = 0; j < 32; j++) {
                float x;
                uint32_t u =
                    (uint32_t)((const uint16_t *)input)[(size_t)r * w->k + (kb + b) * 32 + j] << 16;
                __builtin_memcpy(&x, &u, 4);
                if (fabsf(x) > amax)
                  amax = fabsf(x);
              }
              if (amax > 0) {
                frexpf(amax, &exponent);
                --exponent;
              }
            }
          }
          as[(kb + b) * m + r] = (uint8_t)(exponent + 127);
          for (int j = 0; j < 32; j++) {
            uint8_t c = 0;
            if (r < rows) {
              int k = (kb + b) * 32 + j;
              if (input_scales) {
                size_t ix = block
                                ? (size_t)(k / block) * rows * block + (size_t)r * block + k % block
                                : (size_t)r * w->k + k;
                uint8_t v = ((const uint8_t *)input)[ix];
                c = g == 2 ? v : lookup[g][v];
              } else {
                float x;
                uint32_t u = (uint32_t)((const uint16_t *)input)[(size_t)r * w->k + k] << 16;
                __builtin_memcpy(&x, &u, 4);
                c = mxmm_encode(ldexpf(x, -exponent), fs[g]);
              }
            }
            size_t ix = (size_t)(g == 0 ? r / 2 : r) * kg + b * 32 + j;
            if (g == 0) {
              if (!(r & 1))
                ag[g][ix] = c;
              else
                ag[g][ix] |= c << 4;
            } else
              ag[g][ix] = c;
          }
        }
    kb += ns[g];
  }
  const uint64_t mesh_start = mxmm_cycles();
  mxmm_pack_cycles += mesh_start - pack_start;
  int first = -1, last = -1;
  for (int g = 0; g < 3; g++)
    if (ns[g]) {
      if (first < 0)
        first = g;
      last = g;
    }
  for (int col = 0; col < w->n; col += nc) {
    kb = 0;
    for (int g = 0; g < 3; kb += ns[g++]) {
      if (!ns[g])
        continue;
      int kg = ns[g] * 32, half = g == 0 ? 2 : 1, code = g == 0 ? 2 : g == 1 ? 1 : 0;
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                                  code, code, 3, 0);
      gemmini_extended3_config_ld(kg, MVIN_SCALE_IDENTITY, false, 0);
      gemmini_extended3_config_ld(w->n / half, MVIN_SCALE_IDENTITY, false, 1);
      gemmini_config_st(half * w->n * sizeof(uint16_t));
      for (int b = 0; b < ns[g]; b += s.run) {
        int run = ns[g] - b < s.run ? ns[g] - b : s.run, h = 1 + (mxn_loops++ & 1);
        int accum = !(g == first && !b), store = g == last && b + run == ns[g];
        uint16_t *dst = store ? out + col : NULL;
        if (g == 0) {
          gemmini_loop_ws_mx(m / (2 * DIM), nc / (2 * DIM), run * 32 / DIM, ag[g] + b * 32,
                             bs[g] + (size_t)b * 32 * (w->n / 2) + col / 2, dst, kg, w->n / 2, w->n,
                             as + (kb + b) * m, w->scales + (size_t)(kb + b) * w->n + col, m, w->n,
                             accum, h, h)
        } else {
          gemmini_loop_ws_mx_shared(m / DIM, nc / DIM, run * 32 / DIM, ag[g] + b * 32,
                                    bs[g] + (size_t)b * 32 * w->n + col, dst, kg, w->n, w->n,
                                    as + (kb + b) * m, w->scales + (size_t)(kb + b) * w->n + col, m,
                                    w->n, accum, h, h)
        }
      }
      if (s.fence_groups)
        mxmm_device_fence();
    }
  }
  mxmm_device_fence();
  mxmm_mesh_cycles += mxmm_cycles() - mesh_start;
  return 0;
}
static inline int mxmm_project_fp8(const uint8_t *a, const uint8_t *sc, int m, int block,
                                   const mxmm_weight *w, uint16_t *out, uint8_t *scratch,
                                   mxmm_schedule s) {
  return mxmm_project_impl(a, sc, m, m, block, w, out, scratch, s);
}
static inline int mxmm_project_bf16(const uint16_t *a, int rows, int m, const mxmm_weight *w,
                                    uint16_t *out, uint8_t *scratch, mxmm_schedule s) {
  return mxmm_project_impl(a, NULL, rows, m, 0, w, out, scratch, s);
}
#endif
