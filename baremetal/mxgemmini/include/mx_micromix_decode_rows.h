#ifndef MX_MICROMIX_DECODE_ROWS_H
#define MX_MICROMIX_DECODE_ROWS_H
/* BF16 decode projection with an explicit row-layout selector.
 * Mixed16 accepts 1..16 valid rows and requires FP4 plus FP6 and/or FP8.
 * Packing and scale pitches remain 32 rows. Output capacity is 32*N BF16
 * words: rows 0..15 are written, including padding; rows 16..31 are untouched.
 * Workspace needs 32*(K+K/32) bytes. Output, workspace, weight groups and scales
 * require 64-byte bases; the schedule must use group fences and fit 2,048
 * weight-scale bytes per run. Shape/pointer failures return before device work.
 * Callers provide finite inputs, supported encoded weights, valid capacities
 * and disjoint buffers; this entrypoint does not check arbitrary alias spans.
 * Calls own the synchronous device stream and cannot overlap a prepared lease.
 * The mixed16 wrapper retires scale halves before packing. Its select/fence
 * sequence is included in projection/body/mesh counters. Use the public
 * dispatcher below rather than calling the internal row kernel directly. */
#include "mx_micromix_decode_rows_impl.h"
static uint8_t mxdr_scale_sink[64] __attribute__((aligned(64)));
static int mxdr_project_bf16(const uint16_t *a, int rows, const mxmm_weight *w, uint16_t *out,
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
  gemmini_mxquant_config_mvout((uint64_t)mxdr_scale_sink, 1, 1, 1, 1, 1, 1);
  mxmm_device_fence();
  gemmini_mxquant_config_mvout((uint64_t)mxdr_scale_sink, 1, 1, 1, 0, 0, 1);
  mxmm_device_fence();
  mxmm_mesh_cycles += mxmm_cycles() - begin;
  return mxdr_band_project_bf16(a, rows, w, out, scratch, s);
}
/* Row layout is independent of attention mode. Zero uses the canonical
 * 32-row projection; mixed16 is opt-in and is not selected by measured profiles. */
typedef enum { MXMM_DECODE_ISSUE_M32 = 0, MXMM_DECODE_ISSUE_MIXED16 = 1 } mxmm_decode_issue_mode;
static int mxmm_decode_project_bf16(const uint16_t *a, int rows, const mxmm_weight *w,
                                    uint16_t *out, uint8_t *scratch, mxmm_schedule schedule,
                                    mxmm_decode_issue_mode mode) {
  if (mode == MXMM_DECODE_ISSUE_M32)
    return mxmm_project_bf16(a, rows, 32, w, out, scratch, schedule);
  if (mode == MXMM_DECODE_ISSUE_MIXED16)
    return mxdr_project_bf16(a, rows, w, out, scratch, schedule);
  return 1;
}
#endif
