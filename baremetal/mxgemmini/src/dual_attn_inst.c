// dual_attn_inst.c -- one Gemmini's share of src/dual_attn_llama.c, compiled once per instance (-iquote dual/<inst>,
// -DINST=16|8). src/attn_flash.c is the engine (its pipelined pass: mesh QK / PV in key blocks, online softmax on the VPU,
// SPAD_REQUANT of P); this file re-issues that pass one key block per step() call, with NO fences, and chains passes
// back to back: the next pass's Q scales are a GATED load into act half 1, issued right after the gated weight-scale
// load of this pass's last PV (it starts only once act half 1 is FREE, i.e. once a PV config has superseded the last
// QK config -- all of this pass's QKs have drained), and the next pass's QK0 config waits for it (ATTN_QK0_WAIT_Q).
#include <stdint.h>
#include <stddef.h>
#include "dual_attn_api.h"

#define ATTN_HEADER "attn_dual_kv0_shape.h"
#define ATTN_LAYER 1
#define ATTN_PARAMS_DIM 1
#define ATTN_QK0_WAIT_Q 1
#ifndef BK
#define BK 128
#endif
#include ATTN_HEADER
static const uint8_t (*p_q)[ATTN_D], (*p_qs)[ATTN_SQ], (*p_kt)[ATTN_SK], (*p_kts)[ATTN_SK], (*p_v)[ATTN_D],
                     (*p_vs)[ATTN_D];
static uint16_t *o_dst;
#define ATTN_Q    p_q
#define ATTN_QS   p_qs
#define ATTN_KT   p_kt
#define ATTN_KTS  p_kts
#define ATTN_V    p_v
#define ATTN_VS   p_vs
#define ATTN_OREF p_q   /* unused: graded by hash in the driver */
#define ATTN_STORE_O() mvout_rows(o_dst, SP_O, ROWS16(SQ, D))
#include "attn_flash.c"

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define FN(x) CAT(x, CAT(_g, INST))

static const da_bufs_t *dab;
static int p_first, p_end, pass, blk, stage_, q_next_issued;
enum { ST_BEGIN, ST_BODY, ST_TAIL, ST_DONE };

static void set_pass(int p) {
  p_q = (const uint8_t (*)[ATTN_D]) (dab->q + (size_t) p * SQ * D);
  p_qs = (const uint8_t (*)[ATTN_SQ]) (dab->q_fold + (size_t) p * (D / 32) * SQ);
  o_dst = dab->o + (size_t) p * SQ * D;
}
static void q_scales_gated(int p) {   // pass p's (1/sqrt(d)-folded) Q scales -> act half 1, once it is FREE
  gemmini_mx_load_scales_2d_gated((uint64_t) (dab->q_fold + (size_t) p * (D / 32) * SQ), SQ, D / 32, SQ, 4096, 0);
}
static void issue(void) {
  issue_mm();
  // issue_mm(n_mm - 2) just issued the last matmul's gated weight-scale load: the next pass's Q scales queue behind it
  if (mm_next == n_mm - 1 && pass + 1 < p_end && !q_next_issued) { q_scales_gated(pass + 1); q_next_issued = 1; }
}

void FN(da_init)(const da_bufs_t *b, int first_pass, int n_passes) {
  dab = b;
  p_kt = (const uint8_t (*)[ATTN_SK]) b->kt;
  p_kts = (const uint8_t (*)[ATTN_SK]) b->kts;
  p_v = (const uint8_t (*)[ATTN_D]) b->v;
  p_vs = (const uint8_t (*)[ATTN_D]) b->vs;
  p_first = pass = first_pass;
  p_end = first_pass + n_passes;
  stage_ = n_passes > 0 ? ST_BEGIN : ST_DONE;
}

#define BLK_CYC (2u * SQ * BK * D / (DIM * DIM))   // ideal mesh cycles of one key block (QK + PV)

int FN(da_step)(void) {
  switch (stage_) {
  case ST_BEGIN:
    set_pass(pass);
    n_mm = 0; mm_next = 0; q_next_issued = 0;
    sched[n_mm++] = (mm_t){0, 0}; sched[n_mm++] = (mm_t){0, 1};
    for (int j = 0; j < NB; j++) {
      if (ATTN_QK_FIRST && j + 2 < NB) sched[n_mm++] = (mm_t){0, j + 2};
      sched[n_mm++] = (mm_t){1, j};
      if (!ATTN_QK_FIRST && j + 2 < NB) sched[n_mm++] = (mm_t){0, j + 2};
    }
    if (pass == p_first) {   // the format before any scale load; later passes' Q scales were queued by the previous pass
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, OUT_BF16, 0);
      q_scales_gated(pass);
    }
    if (ATTN_STREAM) { load_q(); load_k(0); load_k(1); for (int v = 0; v < V_BUFS - 1; v++) load_v(v); } else loads();
    issue(); issue();                                   // QK0, QK1
    softmax_block(0); requant_block(0);
    if (ATTN_STREAM) load_k(2);
    blk = 0; stage_ = ST_BODY;
    return BLK_CYC;
  case ST_BODY: {   // run_pipelined's loop body for block j, fences removed
    const int j = blk;
    if (ATTN_QK_FIRST && j + 2 < NB) issue();
    if (ATTN_VPU_EARLY && j + 1 < NB) softmax_block(j + 1);
    issue();                                            // PV(j)
    if (ATTN_VPU_EARLY && ATTN_STREAM) { load_k(j + 3); load_v(j + V_BUFS - 1); }
    if (ATTN_VPU_EARLY && j > ATTN_UPD_LATE) update_block(j - ATTN_UPD_LATE);
    if (!ATTN_VPU_EARLY && j + 1 < NB) softmax_block(j + 1);
    if (j + 1 < NB) requant_block(j + 1);
    if (!ATTN_VPU_EARLY && ATTN_STREAM) { load_k(j + 3); load_v(j + V_BUFS - 1); }
    if (!ATTN_QK_FIRST && j + 2 < NB) issue();
    if (!ATTN_VPU_EARLY && j > 0) update_block(j);
    if (++blk == NB) stage_ = ST_TAIL;
    return BLK_CYC;
  }
  case ST_TAIL:
#if ATTN_FOLD_FINAL
    finalize_folded();
#else
    if (ATTN_VPU_EARLY && ATTN_UPD_LATE && NB > 1) update_block(NB - 1);
    finalize();
#endif
    stage_ = (++pass < p_end) ? ST_BEGIN : ST_DONE;
    return SQ * D / VL;
  default:
    return -1;
  }
}

void FN(da_info)(da_info_t *i) {
  i->dim = DIM; i->opcode = XCUSTOM_ACC; i->bk = BK; i->nb = NB; i->stream = ATTN_STREAM; i->split = ATTN_SPLIT;
  i->pass_ideal = 2ull * SQ * SK * D / (DIM * DIM);
}
