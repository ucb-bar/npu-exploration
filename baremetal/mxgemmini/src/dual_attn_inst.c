// dual_attn_inst.c -- one Gemmini's share of src/dual_attn_llama.c, compiled once per instance (-iquote dual/<inst>,
// -DINST=16|8). src/attn_flash.c is the engine (its pipelined pass: mesh QK / PV in key blocks, online softmax on the VPU,
// SPAD_REQUANT of P); this file re-issues that pass one key block per step() call, with NO fences, and chains passes
// back to back: the next pass's Q scales are a GATED load into act half 1, issued right after the gated weight-scale
// load of this pass's last PV (it starts only once act half 1 is FREE, i.e. once a PV config has superseded the last
// QK config -- all of this pass's QKs have drained), and the next pass's QK0 config waits for it (ATTN_QK0_WAIT_Q).
//
// DA_KSPLIT: every pass covers only keys [DA_K0, DA_K0 + DA_SKN) of the cache (split-K flash attention across the two
// Gemminis): no final 1/l (ATTN_PARTIAL); the unnormalized O, the running max m and the row sums l go to DRAM (m / l as
// [SQ][8] BF16 rows, lane-replicated, so either DIM can read them), and the 16x16 merges the two partials per head with
// its VPU (da_merge_g16) after the driver's fence.
#include <stdint.h>
#include <stddef.h>
#include "dual_attn_api.h"
#include "dual_qfree.h"   // command count + QUEUE_FREE query (credit-based issue)

#ifndef DA_SHAPE
#define DA_SHAPE "attn_dual_kv0_shape.h"
#endif
#define ATTN_HEADER DA_SHAPE
#define ATTN_LAYER 1
#define ATTN_PARAMS_DIM 1
#define ATTN_QK0_WAIT_Q 1
#ifndef BK
#define BK 128
#endif
#ifndef DA_KSPLIT
#define DA_KSPLIT 0
#endif
#if DA_KSPLIT
#define ATTN_SK DA_SKN
#define ATTN_PARTIAL 1
#define ATTN_STORE_PARTIAL() store_partial()
static void store_partial(void);
#else
#define DA_K0 0
#endif
#include ATTN_HEADER
#ifndef ATTN_SK_FULL
#define ATTN_SK_FULL ATTN_SK
#endif
#define ATTN_KT_PITCH ATTN_SK_FULL
static const uint8_t (*p_q)[ATTN_D], (*p_qs)[ATTN_SQ], (*p_kt)[ATTN_SK_FULL], (*p_kts)[ATTN_SK_FULL], (*p_v)[ATTN_D],
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
#define DQ_IMPL
#include "dual_qfree.h"

#ifndef CAT
#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#endif
#define FN(x) CAT(x, CAT(_g, INST))
#define IDX (INST == 16 ? 0 : 1)

static const da_bufs_t *dab;
static int p_first, p_end, pass, blk, stage_, q_next_issued;
enum { ST_BEGIN, ST_BODY, ST_TAIL, ST_DONE };
static uint32_t dq_cap, dq_need[ST_DONE];   // command-queue size; most commands per step kind so far

static void set_pass(int p) {
  p_q = (const uint8_t (*)[ATTN_D]) (dab->q + (size_t) p * SQ * D);
  p_qs = (const uint8_t (*)[ATTN_SQ]) (dab->q_fold + (size_t) p * (D / 32) * SQ);
  o_dst = dab->o + (size_t) p * SQ * D;
}

#if DA_KSPLIT
// unnormalized O (flat BF16 [SQ][D]) and m / l as [SQ][8] BF16 rows (16 B DRAM stride; at DIM 8 two 8 B halves)
#ifndef DA_TAIL_CMDS
#define DA_TAIL_CMDS 64   // O mvouts per tail step, so a step fits the command queue (DIM 8: 256 mvouts in all)
#endif
static int tail_r;   // next O row of the current pass's partial store
static void store_partial_o(int r0, int r1) {   // O rows [r0, r1) of the unnormalized partial (mvout_rows' addressing)
  uint8_t *dst = (uint8_t *) (dab->o_part[IDX] + (size_t) pass * SQ * D);
  if (r0 == 0) gemmini_config_st(DIM);
  for (int r = r0; r < r1; r += DIM) gemmini_extended_mvout(dst + r * DIM, SP_O + r, DIM, DIM);
}
static void store_partial_ml(void) {   // m / l as [SQ][8] BF16 rows (16 B DRAM stride; at DIM 8 two 8 B halves)
  uint8_t *pm = (uint8_t *) (dab->m_part[IDX] + (size_t) pass * SQ * 8);
  uint8_t *pl = (uint8_t *) (dab->l_part[IDX] + (size_t) pass * SQ * 8);
  gemmini_config_st(16);
  for (int h = 0; h < 16 / DIM; h++)
    for (int r = 0; r < SQ; r += DIM) {
      gemmini_extended_mvout(pm + r * 16 + h * DIM, ST_M((NB - 1) & 1) + r, DIM, DIM);
      gemmini_extended_mvout(pl + r * 16 + h * DIM, ST_L + r, DIM, DIM);
    }
}
static void store_partial(void) { store_partial_o(0, ROWS16(SQ, D)); store_partial_ml(); }   // unchunked (finalize)
#endif

static void q_scales_gated(int p) {   // pass p's Q scales (1/sqrt(d) folded or carried by Q) -> act half 1, once FREE
  gemmini_mx_load_scales_2d_gated((uint64_t) (dab->q_fold + (size_t) p * (D / 32) * SQ), SQ, D / 32, SQ, 4096, 0);
}
static void issue(void) {
  issue_mm();
  // issue_mm(n_mm - 2) just issued the last matmul's gated weight-scale load: the next pass's Q scales queue behind it
  if (mm_next == n_mm - 1 && pass + 1 < p_end && !q_next_issued) { q_scales_gated(pass + 1); q_next_issued = 1; }
}

void FN(da_init)(const da_bufs_t *b, int first_pass, int n_passes) {
  dab = b;
  p_kt = (const uint8_t (*)[ATTN_SK_FULL]) (b->kt + DA_K0);
  p_kts = (const uint8_t (*)[ATTN_SK_FULL]) (b->kts + DA_K0);
  p_v = (const uint8_t (*)[ATTN_D]) (b->v + (size_t) DA_K0 * D);
  p_vs = (const uint8_t (*)[ATTN_D]) (b->vs + (size_t) (DA_K0 / 32) * D);
  p_first = pass = first_pass;
  p_end = first_pass + n_passes;
  stage_ = n_passes > 0 ? ST_BEGIN : ST_DONE;
  dq_cap = dq_free();   // nothing issued yet: the whole command queue
}

#define BLK_CYC (2u * SQ * BK * D / (DIM * DIM))   // ideal mesh cycles of one key block (QK + PV)

static int step_impl(void) {
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
#if DA_KSPLIT
    // the partial store in queue-sized steps: O rows in DA_TAIL_CMDS-mvout chunks, then m / l
    if (tail_r == 0 && ATTN_VPU_EARLY && ATTN_UPD_LATE && NB > 1) update_block(NB - 1);
    if (tail_r < ROWS16(SQ, D)) {
      const int r1 = tail_r + DA_TAIL_CMDS * DIM < ROWS16(SQ, D) ? tail_r + DA_TAIL_CMDS * DIM : ROWS16(SQ, D);
      store_partial_o(tail_r, r1);
      tail_r = r1;
      return DA_TAIL_CMDS * DIM / VL;
    }
    store_partial_ml();
    tail_r = 0;
#elif ATTN_FOLD_FINAL
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

// Credit-based issue: the most commands each step kind (begin / key block / tail) has issued so far; until a kind has
// run once it needs the whole queue. room() = this Gemmini's queue can take its next step without blocking the CPU.
int FN(da_room)(void) {
  if (stage_ == ST_DONE) return 1;
  const uint32_t need = dq_need[stage_] && dq_need[stage_] < dq_cap ? dq_need[stage_] : dq_cap;
  return dq_free() >= need;
}
int FN(da_step)(void) {
  const int st = stage_;
  const uint32_t n0 = dq_n;
  const int r = step_impl();
  if (st != ST_DONE && dq_n - n0 > dq_need[st]) dq_need[st] = dq_n - n0;
  return r;
}

#if DA_KSPLIT && INST == 16
// Merge the two key-range partials of passes [0, n) on this (16x16) Gemmini, issued after the driver's fence:
//   m = max(m1, m2); a_i = exp(m_i - m); l = l1*a1 + l2*a2; r = 1/l; c_i = a_i*r; O = O1*c1 + O2*c2   (VPU, BF16)
// Two spad buffer sets alternate, so head h+1's loads run under head h's VPU work.
#define MR (ROWS16(SQ, D))
#define MSET (2 * MR + 11 * SQ)
static void mvin_flat(const void *src, uint32_t sp, int rows) {   // 16 B DRAM rows (stride set by the caller)
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvin((const uint8_t *) src + r * 16, sp + r, DIM, DIM);
}
void da_merge_g16(int n) {
  gemmini_config_ld(16);
  for (int p = 0; p < n; p++) {
    const uint32_t b = (p & 1) * MSET, O1 = b, O2 = b + MR, m1 = b + 2 * MR, m2 = m1 + SQ, l1 = m2 + SQ, l2 = l1 + SQ,
                   m = l2 + SQ, a1 = m + SQ, a2 = a1 + SQ, t = a2 + SQ, l = t + SQ, c1 = l + SQ, c2 = c1 + SQ;
    mvin_flat(dab->o_part[0] + (size_t) p * SQ * D, O1, MR);
    mvin_flat(dab->o_part[1] + (size_t) p * SQ * D, O2, MR);
    mvin_flat(dab->m_part[0] + (size_t) p * SQ * 8, m1, SQ);
    mvin_flat(dab->m_part[1] + (size_t) p * SQ * 8, m2, SQ);
    mvin_flat(dab->l_part[0] + (size_t) p * SQ * 8, l1, SQ);
    mvin_flat(dab->l_part[1] + (size_t) p * SQ * 8, l2, SQ);
    gemmini_vpu_binary(VPU_MAX, m, m1, m2, SQ);
    gemmini_vpu_binary(VPU_SUB, a1, m1, m, SQ);
    gemmini_vpu_unary(VPU_EXP, a1, a1, SQ);
    gemmini_vpu_binary(VPU_SUB, a2, m2, m, SQ);
    gemmini_vpu_unary(VPU_EXP, a2, a2, SQ);
    gemmini_vpu_binary(VPU_MUL, l, l1, a1, SQ);
    gemmini_vpu_binary(VPU_MUL, t, l2, a2, SQ);
    gemmini_vpu_binary(VPU_ADD, l, l, t, SQ);
    gemmini_vpu_unary(VPU_RCP, l, l, SQ);
    gemmini_vpu_binary(VPU_MUL, c1, a1, l, SQ);
    gemmini_vpu_binary(VPU_MUL, c2, a2, l, SQ);
    gemmini_vpu_bcast(VPU_MUL, O1, O1, c1, MR, D / VL);
    gemmini_vpu_bcast(VPU_MUL, O2, O2, c2, MR, D / VL);
    gemmini_vpu_binary(VPU_ADD, O1, O1, O2, MR);
    mvout_rows(dab->o + (size_t) p * SQ * D, O1, MR);
  }
}
#endif

void FN(da_info)(da_info_t *i) {
  i->dim = DIM; i->opcode = XCUSTOM_ACC; i->bk = BK; i->nb = NB; i->stream = ATTN_STREAM; i->split = ATTN_SPLIT;
  i->k0 = DA_K0; i->skn = SK;
  i->pass_ideal = 2ull * SQ * SK * D / (DIM * DIM);
}
