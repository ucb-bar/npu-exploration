#ifndef MX_MICROMIX_LAYER_H
#define MX_MICROMIX_LAYER_H
#include "mx_micromix_prepared.h"
/* Seven-projection order Q,K,V,O,gate,up,down. A new nonzero epoch denotes
 * an immutable captured layer invocation, not merely a matching address.
 * The caller retains ownership of source/scales until each phase releases. */
typedef struct {
  const mxmm_weight *weights;
  uint8_t *scratch, *prepared_workspace;
  size_t prepared_capacity;
  mxmm_schedule schedule;
  uint64_t epoch;
  int prepared_reuse;
  mxmp_prepared phase;
} mxmm_layer_projections;
static inline int mxmm_layer_begin(mxmm_layer_projections *p, uint64_t epoch) {
  if (!p || !p->weights || !p->scratch || !epoch || epoch <= p->epoch || mxmp_live)
    return 1;
  p->epoch = epoch;
  return 0;
}
/* Unlike prepare's single-consumer interface, the layer knows every phase
 * consumer. Reject workspace/future-weight overlap BEFORE CPU packing writes
 * anything. Recheck the whole phase on emit, including the known output span. */
static int mxmm_layer_phase_safe(const mxmm_layer_projections *p, int projection, const uint8_t *a,
                                 const uint8_t *sc, int m, int block, uint16_t *out) {
  int first = projection <= 2 ? 0 : 4, last = first == 0 ? 2 : 5, nc;
  const mxmm_weight *initial = p->weights + first;
  if (!a || !sc || !out || !p->prepared_workspace || ((uintptr_t)out & 63) ||
      ((uintptr_t)p->prepared_workspace & 63) || (block && (block < 32 || block % 32)) ||
      !mxmp_mixed_legal(m, initial, p->schedule, &nc) ||
      p->prepared_capacity < mxmm_workspace_bytes(m, initial->k) || (block && initial->k % block))
    return 0;
  if (mxmp_live && (mxmp_live != &p->phase || p->phase.workspace != p->prepared_workspace ||
                    p->phase.workspace_bytes != p->prepared_capacity))
    return 0;
  for (int i = first; i <= last; i++) {
    const mxmm_weight *w = p->weights + i;
    if (!mxmp_mixed_legal(m, w, p->schedule, &nc) || w->k != initial->k || w->nb4 != initial->nb4 ||
        w->nb6 != initial->nb6 || w->nb8 != initial->nb8)
      return 0;
  }
  size_t output_bytes = (size_t)m * p->weights[projection].n * 2;
  size_t input_bytes = (size_t)m * initial->k, scale_bytes = (size_t)m * (initial->k / 32);
  if (mxmp_overlap(p->prepared_workspace, p->prepared_capacity, a, input_bytes) ||
      mxmp_overlap(p->prepared_workspace, p->prepared_capacity, sc, scale_bytes) ||
      mxmp_overlap(out, output_bytes, p->prepared_workspace, p->prepared_capacity) ||
      mxmp_overlap(out, output_bytes, a, input_bytes) ||
      mxmp_overlap(out, output_bytes, sc, scale_bytes))
    return 0;
  for (int i = first; i <= last; i++) {
    if (mxmp_weight_overlap(p->prepared_workspace, p->prepared_capacity, p->weights + i) ||
        mxmp_weight_overlap(out, output_bytes, p->weights + i))
      return 0;
  }
  return 1;
}
static int mxmm_layer_project_fp8(mxmm_layer_projections *p, int projection, const uint8_t *a,
                                  const uint8_t *sc, int m, int block, uint16_t *out) {
  if (!p || !p->weights || !p->scratch || !p->epoch || projection < 0 || projection >= 7)
    return 1;
  const mxmm_weight *w = p->weights + projection;
  unsigned phase = projection <= 2 ? 1 : (projection == 4 || projection == 5) ? 2 : 0;
  if (p->prepared_reuse && phase) {
    if (!mxmm_layer_phase_safe(p, projection, a, sc, m, block, out))
      return 1;
    int rc = 0;
    if (projection == 0 || projection == 4)
      rc = mxmp_prepare_fp8(&p->phase, a, sc, m, block, w, p->prepared_workspace,
                            p->prepared_capacity, p->schedule, p->epoch, phase,
                            projection == 0 ? 3 : 2);
    if (!rc)
      rc = mxmp_emit_fp8(&p->phase, a, sc, m, block, w, out, p->schedule, p->epoch, phase);
    if (!rc && (projection == 2 || projection == 5))
      rc = mxmp_release(&p->phase);
    return rc;
  }
  return mxmm_project_fp8(a, sc, m, block, w, out, p->scratch, p->schedule);
}
#endif
