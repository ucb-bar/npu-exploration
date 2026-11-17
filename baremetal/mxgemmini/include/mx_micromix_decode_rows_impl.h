#ifndef MX_MICROMIX_DECODE_ROWS_IMPL_H
#define MX_MICROMIX_DECODE_ROWS_IMPL_H
/* Internal mixed decode kernel. FP4 issues 32 rows and FP6/FP8 issue 16
 * through the shared-accumulator layout. Storage and scale pitches remain 32.
 * The first 16 output rows are written; rows 16..31 are untouched.
 * Pure-FP8 weights use the canonical API instead. The public wrapper supplies
 * the required scale-retirement sequence before entering this kernel. */
#include "mx_micromix.h"
#if DIM != 16
#error "Mixed band-row prototype requires the exact DIM16 image"
#endif
static inline size_t mxdr_band_workspace_bytes(int k) { return (size_t)32 * (k + k / 32); }
static inline int mxdr_band_legal(const mxmm_weight *w, mxmm_schedule s, int *nc) {
  return w && w->nb4 > 0 && (w->nb6 > 0 || w->nb8 > 0) && s.fence_groups == 1 &&
         mxmm_legal(32, w, s, nc) && (size_t)s.run * (*nc) <= 2048;
}
static int mxdr_band_project_bf16(const uint16_t *a, int rows, const mxmm_weight *w, uint16_t *out,
                                  uint8_t *scratch, mxmm_schedule s) {
  int nc;
  if (!a || !out || !scratch || rows < 1 || rows > 16 || !mxdr_band_legal(w, s, &nc))
    return 1;
  if (!w->scales || !w->b4 || (w->nb6 && !w->b6) || (w->nb8 && !w->b8) ||
      (((uintptr_t)out | (uintptr_t)scratch | (uintptr_t)w->scales | (uintptr_t)w->b4 |
        (uintptr_t)w->b6 | (uintptr_t)w->b8) &
       63))
    return 1;
  uint64_t begin = mxmm_cycles();
  int ns[] = {w->nb4, w->nb6, w->nb8}, kb = 0;
  uint8_t *ag[3], *as = scratch + (size_t)32 * w->k;
  const uint8_t *bs[] = {w->b4, w->b6, w->b8};
  size_t off = 0;
  for (int g = 0; g < 3; g++) {
    int kg = ns[g] * 32;
    ag[g] = scratch + off;
    off += (size_t)32 * kg / (g == 0 ? 2 : 1);
    if (ns[g])
      mxmm_pack2_group(a, NULL, rows, 32, w->k, 0, g, kb, ns[g], ag[g], as);
    kb += ns[g];
  }
  uint64_t mesh = mxmm_cycles();
  mxmm_pack_cycles += mesh - begin;
  int last = w->nb8 ? 2 : 1;
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
        int run = ns[g] - b < s.run ? ns[g] - b : s.run, h = 1 + (mxn_loops++ & 1),
            accum = !(g == 0 && !b), store = g == last && b + run == ns[g];
        uint16_t *dst = store ? out + col : NULL;
        if (g == 0) {
          gemmini_loop_ws_mx(1, nc / (2 * DIM), run * 32 / DIM, ag[g] + b * 32,
                             bs[g] + (size_t)b * 32 * (w->n / 2) + col / 2, dst, kg, w->n / 2, w->n,
                             as + (kb + b) * 32, w->scales + (size_t)(kb + b) * w->n + col, 32,
                             w->n, accum, h, h)
        } else {
          /* Issue 16 single-format rows with 32-row storage/scale pitches.
           * The shared flag preserves FP4 lower-half accumulator locations;
           * I=1 makes the store unit write 16 output rows. */
          gemmini_loop_ws_mx_shared(1, nc / DIM, run * 32 / DIM, ag[g] + b * 32,
                                    bs[g] + (size_t)b * 32 * w->n + col, dst, kg, w->n, w->n,
                                    as + (kb + b) * 32, w->scales + (size_t)(kb + b) * w->n + col,
                                    32, w->n, accum, h, h)
        }
      }
      mxmm_device_fence();
    }
  }
  mxmm_device_fence();
  mxmm_mesh_cycles += mxmm_cycles() - mesh;
  return 0;
}
#endif
