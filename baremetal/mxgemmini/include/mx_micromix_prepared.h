#ifndef MICRO_MX_PREPARED_V2_H
#define MICRO_MX_PREPARED_V2_H
/* Mixed-format activation reuse, API version 2.
 * FP8 VPU intermediates use a full row count divisible by 64 and dedicated
 * caller-owned workspace with a 64-byte base. Sources stay immutable for their
 * nonzero capture epoch and phase. One lease may be live; every emit drains
 * before returning. Prepare checks the first consumer; each emit checks its
 * consumer's workspace/output overlap. Callers must prevalidate workspace
 * against future consumers before prepare; the layer dispatcher does this.
 * Pure-FP8 projections and batch-one decode use the canonical API. */
#include "mx_micromix.h"
typedef struct {
  const uint8_t *input, *input_scales;
  uint8_t *workspace, *ag[3], *as;
  size_t workspace_bytes;
  int rows, m, k, block, ns[3], expected, used, valid;
  uint64_t capture_epoch;
  unsigned phase;
} mxmp_prepared;
static mxmp_prepared *mxmp_live;
static uint64_t mxmp_last_epoch;
static unsigned mxmp_last_phase;
static unsigned mxmp_prepares, mxmp_emits, mxmp_reuses;
static inline int mxmp_overlap(const void *a, size_t an, const void *b, size_t bn) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x <= y ? y - x < an : x - y < bn;
}
static inline int mxmp_mixed_legal(int m, const mxmm_weight *w, mxmm_schedule s, int *nc) {
  if (!w || m < 64 || m % 64 || !w->nb4 || !w->nb6 || !w->nb8 || !mxmm_legal(m, w, s, nc))
    return 0;
  /* Each managed scale half holds 2,048 bytes. */
  if ((size_t)s.run * m > 2048 || (size_t)s.run * (*nc) > 2048)
    return 0;
  return w->b4 && w->b6 && w->b8 && w->scales &&
         !(((uintptr_t)w->b4 | (uintptr_t)w->b6 | (uintptr_t)w->b8 | (uintptr_t)w->scales) & 63u);
}
/* Called only after mxmp_mixed_legal has validated nonnegative dimensions. */
static inline int mxmp_weight_overlap(const void *ptr, size_t bytes, const mxmm_weight *w) {
  return mxmp_overlap(ptr, bytes, w->b4, (size_t)w->nb4 * 32 * (w->n / 2)) ||
         mxmp_overlap(ptr, bytes, w->b6, (size_t)w->nb6 * 32 * w->n) ||
         mxmp_overlap(ptr, bytes, w->b8, (size_t)w->nb8 * 32 * w->n) ||
         mxmp_overlap(ptr, bytes, w->scales, (size_t)(w->k / 32) * w->n);
}
static int mxmp_prepare_fp8(mxmp_prepared *p, const uint8_t *input, const uint8_t *sc, int m,
                            int block, const mxmm_weight *w, uint8_t *workspace, size_t capacity,
                            mxmm_schedule schedule, uint64_t epoch, unsigned phase, int consumers) {
  int nc;
  if (!p || !input || !sc || !workspace || mxmp_live || !epoch || !phase ||
      epoch < mxmp_last_epoch || (epoch == mxmp_last_epoch && phase <= mxmp_last_phase) ||
      (consumers != 2 && consumers != 3) || (block && (block < 32 || block % 32)) ||
      !mxmp_mixed_legal(m, w, schedule, &nc) || ((uintptr_t)workspace & 63) ||
      capacity < mxmm_workspace_bytes(m, w->k) ||
      mxmp_overlap(input, (size_t)m * w->k, workspace, capacity) ||
      mxmp_overlap(sc, (size_t)m * (w->k / 32), workspace, capacity) ||
      mxmp_overlap(workspace, capacity, w->b4, (size_t)w->nb4 * 32 * (w->n / 2)) ||
      mxmp_overlap(workspace, capacity, w->b6, (size_t)w->nb6 * 32 * w->n) ||
      mxmp_overlap(workspace, capacity, w->b8, (size_t)w->nb8 * 32 * w->n) ||
      mxmp_overlap(workspace, capacity, w->scales, (size_t)(w->k / 32) * w->n))
    return 1;
  uint64_t begin = mxmm_cycles();
  p->input = input;
  p->input_scales = sc;
  p->workspace = workspace;
  p->workspace_bytes = capacity;
  p->rows = m;
  p->m = m;
  p->k = w->k;
  p->block = block;
  p->expected = consumers;
  p->used = 0;
  p->capture_epoch = epoch;
  p->phase = phase;
  p->valid = 0;
  p->as = workspace + (size_t)m * w->k;
  p->ns[0] = w->nb4;
  p->ns[1] = w->nb6;
  p->ns[2] = w->nb8;
  int kb = 0;
  size_t off = 0;
  for (int g = 0; g < 3; g++) {
    p->ag[g] = workspace + off;
    mxmm_pack2_group(input, sc, m, m, w->k, block, g, kb, p->ns[g], p->ag[g], p->as);
    off += (size_t)m * p->ns[g] * 32 / (g == 0 ? 2 : 1);
    kb += p->ns[g];
  }
  /* readcycle has a compiler memory clobber, so all packed stores precede DMA. */
  mxmm_pack_cycles += mxmm_cycles() - begin;
  p->valid = 1;
  mxmp_live = p;
  mxmp_last_epoch = epoch;
  mxmp_last_phase = phase;
  mxmp_prepares++;
  return 0;
}
static int mxmp_emit_fp8(mxmp_prepared *p, const uint8_t *input, const uint8_t *sc, int m,
                         int block, const mxmm_weight *w, uint16_t *out, mxmm_schedule s,
                         uint64_t epoch, unsigned phase) {
  int nc;
  if (!p || mxmp_live != p || !p->valid || p->used >= p->expected || !input || !sc ||
      input != p->input || sc != p->input_scales || m != p->m || m != p->rows ||
      block != p->block || epoch != p->capture_epoch || phase != p->phase ||
      !mxmp_mixed_legal(m, w, s, &nc) || w->k != p->k || w->nb4 != p->ns[0] || w->nb6 != p->ns[1] ||
      w->nb8 != p->ns[2] || !out || ((uintptr_t)out & 63) ||
      mxmp_weight_overlap(p->workspace, p->workspace_bytes, w) ||
      mxmp_weight_overlap(out, (size_t)m * w->n * 2, w) ||
      mxmp_overlap(out, (size_t)m * w->n * 2, p->workspace, p->workspace_bytes) ||
      mxmp_overlap(out, (size_t)m * w->n * 2, input, (size_t)m * w->k) ||
      mxmp_overlap(out, (size_t)m * w->n * 2, sc, (size_t)m * (w->k / 32)))
    return 1;
  uint64_t begin = mxmm_cycles();
  int ns[] = {w->nb4, w->nb6, w->nb8};
  const uint8_t *bs[] = {w->b4, w->b6, w->b8};
  for (int col = 0; col < w->n; col += nc) {
    int kb = 0;
    for (int g = 0; g < 3; kb += ns[g++]) {
      int kg = ns[g] * 32, half = g == 0 ? 2 : 1, code = g == 0 ? 2 : g == 1 ? 1 : 0;
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                                  code, code, 3, 0);
      gemmini_extended3_config_ld(kg, MVIN_SCALE_IDENTITY, false, 0);
      gemmini_extended3_config_ld(w->n / half, MVIN_SCALE_IDENTITY, false, 1);
      gemmini_config_st(half * w->n * sizeof(uint16_t));
      for (int b = 0; b < ns[g]; b += s.run) {
        int run = ns[g] - b < s.run ? ns[g] - b : s.run, h = 1 + (mxn_loops++ & 1);
        int accum = !(g == 0 && !b), store = g == 2 && b + run == ns[g];
        uint16_t *dst = store ? out + col : NULL;
        if (g == 0) {
          gemmini_loop_ws_mx(m / (2 * DIM), nc / (2 * DIM), run * 32 / DIM, p->ag[g] + b * 32,
                             bs[g] + (size_t)b * 32 * (w->n / 2) + col / 2, dst, kg, w->n / 2, w->n,
                             p->as + (size_t)(kb + b) * m,
                             w->scales + (size_t)(kb + b) * w->n + col, m, w->n, accum, h, h);
        } else {
          gemmini_loop_ws_mx_shared(m / DIM, nc / DIM, run * 32 / DIM, p->ag[g] + b * 32,
                                    bs[g] + (size_t)b * 32 * w->n + col, dst, kg, w->n, w->n,
                                    p->as + (size_t)(kb + b) * m,
                                    w->scales + (size_t)(kb + b) * w->n + col, m, w->n, accum, h,
                                    h);
        }
      }
      if (s.fence_groups)
        mxmm_device_fence();
    }
  }
  mxmm_device_fence();
  mxmm_mesh_cycles += mxmm_cycles() - begin;
  if (p->used)
    mxmp_reuses++;
  p->used++;
  mxmp_emits++;
  return 0;
}
static int mxmp_release(mxmp_prepared *p) {
  if (!p || mxmp_live != p || !p->valid || p->used != p->expected)
    return 1;
  uint64_t t = mxmm_cycles();
  mxmm_device_fence();
  mxmm_mesh_cycles += mxmm_cycles() - t;
  p->valid = 0;
  mxmp_live = NULL;
  return 0;
}
#endif
