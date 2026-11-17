/* Native one-layer prefill example. Operator bodies retained from qualified
 * TinyLlama two-packed-head path; capture and independent same-mix references are supplied by caller. */
// Full TinyLlama layer5 prefill: VPU normalization/attention/nonlinear stages,
// mixed FP4/6/8 mesh projections, and CPU activation packing included in timing.
// Adapted from llama_layer_e2e.c; frozen real 64-token input after a 2048-token KV cache.
// Weight allocation is fixed contiguous K32 groups (not a sensitivity-ranked MicroMix allocation).
//
//   VPU+SR  xn1 = MX(rmsnorm(h_pre) * w_in)                         16-token chunks, two 8-token streams (two VPUs)
//   mesh    Q, K, V = xn1 @ Wq / Wk / Wv                            native DRAM loops (Wq/Wk columns rotary-half split)
//   VPU+SR  Q, K = rope(Q), rope(K) -> MX;  V^T -> MX               new K/V written into the cache (keys 2048..2111)
//   attn    16 GQA-packed flash passes (2 heads each) over 2112 keys, causal on the last block; O -> MX per pass
//   mesh    Yattn = O @ Wo                                          K-tiles = heads (A blocked per head)
//   VPU+SR  h_mid = h_pre + Yattn;  xn2 = MX(rmsnorm(h_mid) * w_post)
//   mesh    G, U = xn2 @ Wg / Wu
//   VPU+SR  H = MX(silu(G) * U)                                     8-token chunks, two column-half streams
//   mesh    Ymlp = H @ Wd
//   VPU     h_out = h_mid + Ymlp
//
// Timed from the first command to h_out in DRAM; utilization = the layer's mesh-ideal cycles / that time. Phases are
// fenced (the CPU reorders scales between them). Outputs are hashed (vs Spike) and graded against fp64 references and
// against TinyLlama's own hidden_states[layer + 1].
#include "mx_micromix_dim16.h"
#include MICROMIX_CAPTURE_HEADER
#if MICROMIX_MODEL != 0
#error "TinyLlama packed-head path requires TinyLlama selector"
#endif
typedef char
    micromix_tiny_capture_shape[(E2E_M == 64 && E2E_SK == 2048 && E2E_D == 2048 && E2E_F == 5632 &&
                                 E2E_NH == 32 && E2E_NKV == 4 && E2E_HD == 64)
                                    ? 1
                                    : -1];

#define ATTN_HEADER MICROMIX_CAPTURE_HEADER
#define ATTN_SQ (2 * E2E_M)      // two packed query heads
#define ATTN_SK (E2E_SK + E2E_M) // cache + the chunk
#define ATTN_D E2E_HD
#define ATTN_HEADS 2
#define ATTN_CAUSAL 1
#define BK 128
#define BK_LAST E2E_M // the last key block = the chunk itself (one mask ADD)
#ifndef ATTN_EXPSUB
#define ATTN_EXPSUB 1
#endif
#ifndef ATTN_EXPSUM
#define ATTN_EXPSUM 1
#endif
#ifndef ATTN_ST_BANK2
#define ATTN_ST_BANK2 1
#endif
#define ATTN_LAYER 1
static int cur_pass, cur_kv;
static uint8_t q_sc_pass[ATTN_D / 32][ATTN_SQ] __attribute__((aligned(64)));
// the cache is written (keys E2E_SK..): through a pointer the compiler cannot trace to the const blob symbol, or it
// may drop the stores
static uint8_t *e2e_rw;
#define KTC_ (e2e_rw + E2E_OFF_KT_CACHE)
#define KTS_ (e2e_rw + E2E_OFF_KT_SCALES)
#define VC_ (e2e_rw + E2E_OFF_V_CACHE)
#define VS_ (e2e_rw + E2E_OFF_V_SCALES)
#define ATTN_Q ((const uint8_t *)0)
#define ATTN_QS q_sc_pass
#define ATTN_KT ((const uint8_t(*)[ATTN_SK])(KTC_ + (size_t)cur_kv * ATTN_D * ATTN_SK))
#define ATTN_KTS ((const uint8_t(*)[ATTN_SK])(KTS_ + (size_t)cur_kv * (ATTN_D / 32) * ATTN_SK))
#define ATTN_V ((const uint8_t(*)[ATTN_D])(VC_ + (size_t)cur_kv * ATTN_SK * ATTN_D))
#define ATTN_VS ((const uint8_t(*)[ATTN_D])(VS_ + (size_t)cur_kv * (ATTN_SK / 32) * ATTN_D))
#define ATTN_OREF ((const uint32_t *)E2E_AT(E2E_OFF_REF_O, uint32_t))
#define MASK_BF16 E2E_AT(E2E_OFF_MASK, uint16_t)
static void e2e_load_q(void);
static void e2e_store_o(void);
#define ATTN_LOAD_Q e2e_load_q
#define ATTN_STORE_O e2e_store_o
#include "attn_flash.c"
#include "mx_native.h"
#include "mx_micromix.h"
#include "mx_micromix_profiles.h"
#include "mx_micromix_layer.h"
#include MXMM_DATA_H
#ifdef MXMM_NUMERIC_MODE
#include "mx_micromix_numeric.h"
#include MXMM_NUMERIC_REF_H
#endif
/* Keep local CPU packing, fixture writes and CSR measurements ordered without
 * changing the frozen headers used by already prepared hardware jobs. */
#undef gemmini_fence
#define gemmini_fence() mxmm_device_fence()
#define read_cycles() mxmm_cycles()
#ifndef MXMM_NC
#define MXMM_NC 0
#endif
#ifndef MXMM_RUN
#define MXMM_RUN 8
#endif
#ifndef MXMM_FENCE_GROUPS
#define MXMM_FENCE_GROUPS 1
#endif
#ifdef MXMM_PHASE_PROGRESS
#define MXMM_PROGRESS(s)                                                                           \
  printf("MXMM_PROGRESS %s cycle=%llu\n", s, (unsigned long long)read_cycles())
#else
#define MXMM_PROGRESS(s)                                                                           \
  do {                                                                                             \
  } while (0)
#endif
static mxmm_schedule layer_schedule = {MXMM_NC, MXMM_RUN, MXMM_FENCE_GROUPS};
static uint8_t mxmm_scratch[E2E_M * (E2E_F + E2E_F / 32)] __attribute__((aligned(64)));
static uint8_t prepared_workspace[E2E_M * (E2E_D + E2E_D / 32)] __attribute__((aligned(64)));
static mxmm_layer_projections projections;
static uint64_t capture_epoch;

#define LM E2E_M
#define LD E2E_D
#define LF E2E_F
#define LHD E2E_HD
#define LNH E2E_NH
#define LKV (E2E_NKV * E2E_HD) // 256
#define QH (LD / 2)            // one rotary half of Q: x1 (or x2) of every head
#define KH (LKV / 2)
#define NPASS (LNH / ATTN_HEADS)
#define FH (LF / 2) // SwiGLU column-half stream
#define SPR(b, r) ((uint32_t)((b) * BANK_ROWS + (r)))

// ---- DRAM buffers ----
#define A64 __attribute__((aligned(64)))
static uint8_t xn1_c[LM * LD] A64, xn1_sr[LM * LD / 32] A64, xn1_s[LD / 32 * LM] A64;
static uint16_t Qb[LM * LD] A64, Kb[LM * LKV] A64, Vb[LM * LKV] A64;
static uint8_t q1_c[LM * QH] A64, q2_c[LM * QH] A64, q1_sr[LM * QH / 32] A64,
    q2_sr[LM * QH / 32] A64;
static uint8_t k1_c[LM * KH] A64, k2_c[LM * KH] A64, k1_sr[LM * KH / 32] A64,
    k2_sr[LM * KH / 32] A64;
static uint16_t vt_b[LKV * LM] A64;
static uint8_t vt_c[LKV * LM] A64, vt_sr[LKV * LM / 32] A64;
static uint8_t o_c[LNH * LM * LHD] A64, o_sr[NPASS][ATTN_SQ * ATTN_D / 32] A64,
    o_s[LD / 32 * LM] A64;
static uint16_t Ya[LM * LD] A64, Hmid[LM * LD] A64;
static uint8_t xn2_c[LM * LD] A64, xn2_sr[LM * LD / 32] A64, xn2_s[LD / 32 * LM] A64;
static uint16_t Gb[LM * LF] A64, Ub[LM * LF] A64;
static uint8_t h_c[LM * LF] A64, h_sr[2][LM * FH / 32] A64, h_s[LF / 32 * LM] A64;
static uint16_t Ym[LM * LD] A64, Hout[LM * LD] A64;

// contiguous DRAM <-> contiguous spad rows, 16 rows per command (config_ld / config_st = DIM set by the caller)
static void mvin16(const void *src, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM)
    gemmini_extended_mvin((const uint8_t *)src + r * DIM, sp + r, DIM, DIM);
}
static void mvout16(void *dst, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM)
    gemmini_extended_mvout((uint8_t *)dst + r * DIM, sp + r, DIM, DIM);
}
// byte of (row m, group g) in SR scales written by ch-row chunks, each chunk [groups][ch]
#define SRI(m, g, groups, ch) (((m) - (m) % (ch)) * (groups) + (g) * (ch) + (m) % (ch))
// chunked SR scales -> the native loop's A-scale layout [cols/32][rows]
static void scales_t(uint8_t *dst, const uint8_t *src, int rows, int groups, int ch) {
  for (int t = 0; t < rows; t += ch)
    for (int g = 0; g < groups; g++)
      for (int r = 0; r < ch; r++)
        dst[g * rows + t + r] = src[t * groups + g * ch + r];
}
// Free every scale half: software-managed passes (attention) and loop-managed native loops each leave halves INUSE
// that the other's gated loads would wait on. A legacy config frees the halves it does not select.
static void scale_halves_reset(void) {
  gemmini_mxquant_config_mvout((uint64_t)scale_sink, 1, 1, 1, 1, 1, 1);
  /* Archived Controller.scala 581-590 gives cfg events priority over retire:
   * consecutive selects 1,0 can cancel half0 retirement before its FREE cycle.
   * Drain each command before selecting the other half. */
  mxmm_device_fence();
  gemmini_mxquant_config_mvout((uint64_t)scale_sink, 1, 1, 1, 0, 0, 1);
  mxmm_device_fence();
}

// ---- RMSNorm (optionally after a residual add) -> MX, 16 tokens per chunk as two 8-token streams: stream s in banks
// 2s (x, its weight copy) / 2s+1 (scratch y, stats, codes), so the two VPUs never share a read bank ----
#define RX(s) SPR(2 * (s), 0)
#define RW(s) SPR(2 * (s), 2048)
#define RY(s) SPR(2 * (s) + 1, 0)
#define RS(s) SPR(2 * (s) + 1, 2048)
#define RC(s) SPR(2 * (s) + 1, 2064)
static void rms_phase(const uint16_t *x, const uint16_t *res, uint16_t *hsum, const uint16_t *w,
                      uint8_t *codes, uint8_t *sraw) {
  const uint16_t inv_d = vpu_f_to_bf16(1.0f / LD), eps = vpu_f_to_bf16(E2E_EPS);
  const int n = 8 * LD / 8; // rows per stream
  MXMM_PROGRESS("rms.config.before");
  gemmini_config_ld(DIM);
  gemmini_config_st(DIM);
  for (int s = 0; s < 2; s++)
    mvin16(w, RW(s), LD / 8);
  MXMM_PROGRESS("rms.weights_issued");
  for (int c = 0; c < LM / 16; c++) {
    int t[2] = {c * 16, c * 16 + 8};
    for (int s = 0; s < 2; s++) {
      mvin16(x + t[s] * LD, RX(s), n);
      if (res)
        mvin16(res + t[s] * LD, RY(s), n);
    }
    if (!c)
      MXMM_PROGRESS("rms.input_issued");
    if (res) {
      for (int s = 0; s < 2; s++)
        gemmini_vpu_binary(VPU_ADD, RX(s), RX(s), RY(s), n);
      for (int s = 0; s < 2; s++)
        mvout16(hsum + t[s] * LD, RX(s), n);
    }
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_MUL, RY(s), RX(s), RX(s), n);
    if (!c)
      MXMM_PROGRESS("rms.square_issued");
    for (int s = 0; s < 2; s++)
      gemmini_vpu_reduce(VPU_RSUM, RS(s), RY(s), n, LD / 8);
    if (!c)
      MXMM_PROGRESS("rms.reduce_issued");
    for (int s = 0; s < 2; s++)
      gemmini_vpu_scalar(VPU_MULS, RS(s), RS(s), inv_d, 8);
    for (int s = 0; s < 2; s++)
      gemmini_vpu_scalar(VPU_ADDS, RS(s), RS(s), eps, 8);
    if (!c)
      MXMM_PROGRESS("rms.scalars_issued");
    for (int s = 0; s < 2; s++)
      gemmini_vpu_unary(VPU_RSQRT, RS(s), RS(s), 8);
    if (!c)
      MXMM_PROGRESS("rms.rsqrt_issued");
    for (int s = 0; s < 2; s++)
      gemmini_vpu_bcast(VPU_MUL, RY(s), RX(s), RS(s), n, LD / 8);
    if (!c)
      MXMM_PROGRESS("rms.bcast_issued");
    for (int k = 0; k < 8; k++)
      for (int s = 0; s < 2; s++)
        gemmini_vpu_binary(VPU_MUL, RY(s) + k * (LD / 8), RY(s) + k * (LD / 8), RW(s), LD / 8);
    if (!c)
      MXMM_PROGRESS("rms.weights_mul_issued");
    for (int s = 0; s < 2; s++)
      gemmini_spad_requant(RC(s), RY(s), 8, LD, 0, (uint64_t)(sraw + t[s] * (LD / 32)), 0);
    if (!c)
      MXMM_PROGRESS("rms.requant_issued");
    for (int s = 0; s < 2; s++)
      mvout16(codes + t[s] * LD, RC(s), n / 2);
  }
}

// ---- RoPE on the two rotary halves (x1, x2 of every head, contiguous after the Wq/Wk column split):
//   x1' = x1 * cos - x2 * sin,  x2' = x2 * cos + x1 * sin  -> MX (SR per half; a 32-group = one head's half)
// two streams of tps tokens: bank 2s holds x1, x2 and the codes, bank 2s+1 cos, sin and two temporaries ----
static void rope_phase(const uint16_t *x, int ld, int half, int tps, int chunks, const uint16_t *cs,
                       const uint16_t *sn, uint8_t *c1, uint8_t *c2, uint8_t *s1, uint8_t *s2) {
  const int hr = half / 8, n = tps * hr;
  gemmini_config_ld(DIM);
  gemmini_config_st(DIM);
  for (int c = 0; c < chunks; c++) {
    int t[2];
    for (int s = 0; s < 2; s++) {
      t[s] = (2 * c + s) * tps;
      for (int m = 0; m < tps; m++) {
        mvin16(x + (size_t)(t[s] + m) * ld, SPR(2 * s, m * hr), hr);
        mvin16(x + (size_t)(t[s] + m) * ld + half, SPR(2 * s, n + m * hr), hr);
      }
      mvin16(cs + (size_t)t[s] * half, SPR(2 * s + 1, 0), n);
      mvin16(sn + (size_t)t[s] * half, SPR(2 * s + 1, n), n);
    }
#define X1 SPR(2 * s, 0)
#define X2 SPR(2 * s, n)
#define TC SPR(2 * s + 1, 0)
#define TS SPR(2 * s + 1, n)
#define TB SPR(2 * s + 1, 2 * n)
#define TT SPR(2 * s + 1, 3 * n)
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_MUL, TB, X2, TS, n);
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_MUL, TT, X1, TS, n);
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_MUL, X1, X1, TC, n);
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_SUB, X1, X1, TB, n);
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_MUL, X2, X2, TC, n);
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_ADD, X2, X2, TT, n);
    for (int s = 0; s < 2; s++) {
      gemmini_spad_requant(SPR(2 * s, 2 * n), X1, tps, half, 0, (uint64_t)(s1 + t[s] * (half / 32)),
                           0);
      gemmini_spad_requant(SPR(2 * s, 2 * n + n / 2), X2, tps, half, 0,
                           (uint64_t)(s2 + t[s] * (half / 32)), 0);
    }
#undef X1
#undef X2
#undef TC
#undef TS
#undef TB
#undef TT
    for (int s = 0; s < 2; s++) {
      mvout16(c1 + (size_t)t[s] * half, SPR(2 * s, 2 * n), n / 2);
      mvout16(c2 + (size_t)t[s] * half, SPR(2 * s, 2 * n + n / 2), n / 2);
    }
  }
}

// ---- V of the chunk -> the cache's MX layout (blocks of 32 keys per head dim): V^T rows through one SR ----
static void vt_transpose(void) { // V [64][256] -> V^T [256][64] BF16 (CPU)
  for (int t = 0; t < LM; t++)
    for (int c = 0; c < LKV; c++)
      vt_b[c * LM + t] = Vb[t * LKV + c];
}
static void vt_requant(void) { // V^T rows -> MX (blocks of 32 keys per head dim)
  gemmini_config_ld(DIM);
  gemmini_config_st(DIM);
  mvin16(vt_b, SPR(0, 0), LKV * LM / 8);
  gemmini_spad_requant(SPR(1, 0), SPR(0, 0), LKV, LM, 0, (uint64_t)vt_sr, 0);
  mvout16(vt_c, SPR(1, 0), LKV * LM / 16);
}
// the chunk's K / V codes and scales into cache keys E2E_SK.. (K^T: [d][key]; V: [key][d], scales [key/32][d])
static void cache_append(void) {
  for (int g = 0; g < E2E_NKV; g++) {
    uint8_t *kt = KTC_ + (size_t)g * ATTN_D * ATTN_SK,
            *kts = KTS_ + (size_t)g * (ATTN_D / 32) * ATTN_SK;
    uint8_t *v = VC_ + (size_t)g * ATTN_SK * ATTN_D,
            *vs = VS_ + (size_t)g * (ATTN_SK / 32) * ATTN_D;
    for (int t = 0; t < LM; t++) {
      for (int d = 0; d < 32; d++) {
        kt[d * ATTN_SK + E2E_SK + t] = k1_c[t * KH + g * 32 + d];
        kt[(32 + d) * ATTN_SK + E2E_SK + t] = k2_c[t * KH + g * 32 + d];
      }
      kts[E2E_SK + t] = k1_sr[SRI(t, g, KH / 32, LM / 2)];
      kts[ATTN_SK + E2E_SK + t] = k2_sr[SRI(t, g, KH / 32, LM / 2)];
      for (int d = 0; d < LHD; d++)
        v[(E2E_SK + t) * ATTN_D + d] = vt_c[(g * LHD + d) * LM + t];
    }
    for (int b = 0; b < LM / 32; b++)
      for (int d = 0; d < LHD; d++)
        vs[(E2E_SK / 32 + b) * ATTN_D + d] = vt_sr[b * LKV + g * LHD + d];
  }
}

// ---- attention hooks: Q of pass cur_pass (heads 2p, 2p+1) from the RoPE output; O -> MX, head-blocked [h][m][d] ----
static void e2e_load_q(void) {
  gemmini_config_ld(QH);
  for (int r = 0; r < ATTN_SQ / DIM; r++) { // 16-row tile r: head 2p + r/4, tokens 16 (r % 4) ..
    const int hh = ATTN_HEADS * cur_pass + r / (LM / DIM), m0 = (r % (LM / DIM)) * DIM;
    gemmini_extended_mvin(q1_c + m0 * QH + hh * 32, SP_Q + r * 4 * DIM, 2 * DIM, DIM); // dims 0..31
    gemmini_extended_mvin(q2_c + m0 * QH + hh * 32, SP_Q + r * 4 * DIM + 2 * DIM, 2 * DIM,
                          DIM); // dims 32..63
  }
}
static void e2e_store_o(void) {
  const uint32_t dst = P_BUF(NB & 1); // the P buffer the last block does not use
  gemmini_spad_requant(dst, SP_O, ATTN_SQ, ATTN_D, 0, (uint64_t)o_sr[cur_pass], 0);
  mvout_rows(o_c + (size_t)ATTN_HEADS * cur_pass * LM * LHD, dst, ROWS8(ATTN_SQ, ATTN_D));
}
static void
q_scales_of_pass(int p) { // [2][128]: row j = head 2p + j/64, token j % 64; group = the rotary half
  for (int j = 0; j < ATTN_SQ; j++) {
    const int hh = ATTN_HEADS * p + j / LM, m = j % LM;
    q_sc_pass[0][j] = q1_sr[SRI(m, hh, QH / 32, 8)];
    q_sc_pass[1][j] = q2_sr[SRI(m, hh, QH / 32, 8)];
  }
}

// ---- SwiGLU -> MX: 8 tokens per chunk, stream s = column half s (banks 0-1 / 2-3; codes between G and U) ----
//   U = U * G; G = 1 / (1 + exp(-G)); U = U * G   (= silu(g) * u)
#define SG_N (8 * FH / 8) // rows per region (8 tokens); codes SG_N / 2
static const uint32_t SG_G[2] = {0, 2 * BANK_ROWS}, SG_C[2] = {SG_N, 2 * BANK_ROWS + SG_N},
                      SG_U[2] = {SG_N + SG_N / 2, 2 * BANK_ROWS + SG_N + SG_N / 2};
typedef char e2e_swiglu_fits[(SG_N <= BANK_ROWS && SG_N + SG_N / 2 >= BANK_ROWS &&
                              2 * SG_N + SG_N / 2 <= 2 * BANK_ROWS && (8 * FH / 32) % 32 == 0)
                                 ? 1
                                 : -1]; // G in bank 2s, U in bank 2s+1
static void swiglu_phase(void) {
  const int n = SG_N;
  gemmini_config_ld(DIM);
  gemmini_config_st(DIM);
  for (int c = 0; c < LM / 8; c++) {
    const int t0 = c * 8;
    for (int s = 0; s < 2; s++)
      for (int m = 0; m < 8; m++) {
        mvin16(Gb + (size_t)(t0 + m) * LF + s * FH, SG_G[s] + m * (FH / 8), FH / 8);
        mvin16(Ub + (size_t)(t0 + m) * LF + s * FH, SG_U[s] + m * (FH / 8), FH / 8);
      }
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_MUL, SG_U[s], SG_U[s], SG_G[s], n);
    if (!c)
      MXMM_PROGRESS("rms.reduce_issued");
    for (int s = 0; s < 2; s++)
      gemmini_vpu_scalar(VPU_MULS, SG_G[s], SG_G[s], 0xBF80, n); // -1
    for (int s = 0; s < 2; s++)
      gemmini_vpu_unary(VPU_EXP, SG_G[s], SG_G[s], n);
    for (int s = 0; s < 2; s++)
      gemmini_vpu_scalar(VPU_ADDS, SG_G[s], SG_G[s], 0x3F80, n); // +1
    for (int s = 0; s < 2; s++)
      gemmini_vpu_unary(VPU_RCP, SG_G[s], SG_G[s], n);
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_MUL, SG_U[s], SG_U[s], SG_G[s], n);
    for (int s = 0; s < 2; s++)
      gemmini_spad_requant(SG_C[s], SG_U[s], 8, FH, 0, (uint64_t)(h_sr[s] + t0 * (FH / 32)), 0);
    for (int s = 0; s < 2; s++)
      for (int m = 0; m < 8; m++)
        mvout16(h_c + (size_t)(t0 + m) * LF + s * FH, SG_C[s] + m * (FH / 16), FH / 16);
  }
}

static void residual_phase(const uint16_t *a, const uint16_t *b, uint16_t *out) {
  const int n = 8 * LD / 8;
  gemmini_config_ld(DIM);
  gemmini_config_st(DIM);
  for (int c = 0; c < LM / 16; c++) {
    int t[2] = {c * 16, c * 16 + 8};
    for (int s = 0; s < 2; s++) {
      mvin16(a + t[s] * LD, RX(s), n);
      mvin16(b + t[s] * LD, RY(s), n);
    }
    for (int s = 0; s < 2; s++)
      gemmini_vpu_binary(VPU_ADD, RX(s), RX(s), RY(s), n);
    for (int s = 0; s < 2; s++)
      mvout16(out + t[s] * LD, RX(s), n);
  }
}

// ---- checks ----
static float rel_bf16(const uint16_t *x, const float *ref, const uint16_t *base,
                      int n) { // ||x - b - ref|| / ||ref||
  float num = 0, den = 0;
  for (int i = 0; i < n; i++) {
    const float r = ref[i], v = bf(x[i]) - (base ? bf(base[i]) : 0.0f);
    num += (v - r) * (v - r);
    den += r * r;
  }
  return sqrtf(num / den);
}
static float rel_upd(const uint16_t *x, const float *ref_out, const uint16_t *base,
                     int n) { // (x - b) vs (ref - b)
  float num = 0, den = 0;
  for (int i = 0; i < n; i++) {
    const float b0 = bf(base[i]), v = bf(x[i]) - b0, r = ref_out[i] - b0;
    num += (v - r) * (v - r);
    den += r * r;
  }
  return sqrtf(num / den);
}

static float e4m3(uint8_t c, uint8_t sc) {
  const int e = (c >> 3) & 15, m = c & 7;
  float v = e ? (8 + m) * ldexpf(1.0f, e - 10) : m * ldexpf(1.0f, -9);
  return ldexpf(c & 0x80 ? -v : v, (int)sc - 127);
}
static float rel_mx(const float *ref, int n, float (*get)(int)) {
  float num = 0, den = 0;
  for (int i = 0; i < n; i++) {
    const float r = ref[i], v = get(i);
    num += (v - r) * (v - r);
    den += r * r;
  }
  return sqrtf(num / den);
}
static float get_xn1(int i) {
  const int m = i / LD, k = i % LD;
  return e4m3(xn1_c[i], xn1_sr[SRI(m, k / 32, LD / 32, 8)]);
}
static float get_o(int i) { // REF_O [m][h*64 + d]
  const int m = i / LD, h = (i % LD) / LHD, d = i % LHD;
  return e4m3(o_c[((size_t)h * LM + m) * LHD + d],
              o_sr[h / 2][(d / 32) * ATTN_SQ + (h % 2) * LM + m]);
}

enum {
  PH_RMS1,
  PH_QKV,
  PH_ROPE,
  PH_ATTN,
  PH_OPROJ,
  PH_RMS2,
  PH_GU,
  PH_SWIGLU,
  PH_DOWN,
  PH_RES2,
  PH_N
};
static const char *ph_name[PH_N] = {"rmsnorm1 -> MX",   "Q/K/V proj",
                                    "rope + K/V cache", "attention (16 passes)",
                                    "o_proj",           "residual1 + rmsnorm2",
                                    "gate/up proj",     "swiglu -> MX",
                                    "down proj",        "residual2"};
#define PASS_IDEAL (2ULL * ATTN_SQ * ATTN_SK * ATTN_D / (DIM * DIM))
static const uint64_t ph_ideal[PH_N] = {
    0, MXN_IDEAL(LM, LD, LD + 2 * LKV), 0, NPASS *PASS_IDEAL,     MXN_IDEAL(LM, LD, LD),
    0, 2 * MXN_IDEAL(LM, LD, LF),       0, MXN_IDEAL(LM, LF, LD), 0};

#ifdef MXMM_EXPECT_H
#include MXMM_EXPECT_H
#define E2E_HAVE_EXPECT 1
#else
#define E2E_HAVE_EXPECT 0
#endif

#ifdef E2E_DEBUG
static void dbg1(const char *n, const void *p, size_t sz) {
  const uint8_t *b = (const uint8_t *)p;
  size_t nz = 0;
  for (size_t i = 0; i < sz; i++)
    nz += b[i] != 0;
  printf("  dbg %-6s nonzero %d/%d  first %02x%02x %02x%02x %02x%02x %02x%02x\n", n, (int)nz,
         (int)sz, b[1], b[0], b[3], b[2], b[5], b[4], b[7], b[6]);
}
static void e2e_dbg(int p) {
  printf(" after %s\n", ph_name[p]);
  if (p == PH_QKV) {
    dbg1("Q", Qb, sizeof(Qb));
    dbg1("V", Vb, sizeof(Vb));
  }
  if (p == PH_ROPE) {
    int c7 = 0, cs = 0, ks = 0, k7 = 0, v7 = 0, vs = 0;
    for (size_t i = 0; i < sizeof(q1_c); i++)
      c7 += (q1_c[i] & 0x7f) == 0x7f;
    for (size_t i = 0; i < sizeof(q1_sr); i++)
      cs += q1_sr[i] == 0xff;
    for (int g = 0; g < E2E_NKV; g++) {
      for (int d = 0; d < ATTN_D; d++)
        for (int t = 0; t < ATTN_SK; t++)
          k7 += (KTC_[((size_t)g * ATTN_D + d) * ATTN_SK + t] & 0x7f) == 0x7f;
      for (int d = 0; d < 2; d++)
        for (int t = 0; t < ATTN_SK; t++)
          ks += KTS_[((size_t)g * 2 + d) * ATTN_SK + t] == 0xff;
      for (size_t i = 0; i < (size_t)ATTN_SK * ATTN_D; i++)
        v7 += (VC_[(size_t)g * ATTN_SK * ATTN_D + i] & 0x7f) == 0x7f;
      for (size_t i = 0; i < (size_t)ATTN_SK / 32 * ATTN_D; i++)
        vs += VS_[(size_t)g * ATTN_SK / 32 * ATTN_D + i] == 0xff;
    }
    printf("  nan q1 codes %d, q1 scales %d, K codes %d, K scales %d, V codes %d, V scales %d\n",
           c7, cs, k7, ks, v7, vs);
    dbg1("q1", q1_c, sizeof(q1_c));
    dbg1("vt_b", vt_b, sizeof(vt_b));
    dbg1("vt_c", vt_c, sizeof(vt_c));
  }
  if (p == PH_ATTN) {
    dbg1("o", o_c, sizeof(o_c));
    dbg1("o_sr", o_sr, sizeof(o_sr));
  }
  if (p == PH_OPROJ)
    dbg1("Ya", Ya, sizeof(Ya));
  if (p == PH_RMS2) {
    dbg1("Hmid", Hmid, sizeof(Hmid));
    dbg1("xn2", xn2_c, sizeof(xn2_c));
  }
}
#endif

#ifndef MXMM_CASE_ID
#define MXMM_CASE_ID "prefill_micromix"
#endif
static uint64_t mxmm_instret(void) {
  uint64_t n;
  asm volatile("csrr %0,minstret" : "=r"(n)::"memory");
  return n;
}
static const char *mxmm_variant = "single";
static int mxmm_trial, mxmm_warmup;
static uint64_t mxmm_projection_elapsed[7], mxmm_projection_packing[7], mxmm_projection_mesh[7];
#ifdef MXMM_NUMERIC_MODE
static uint16_t mxmm_hardware_baseline[MXMM_NUMERIC_WORDS] A64;
static int mxmm_hardware_baseline_ready;
static int mxmm_validate_numeric(const char *case_id, const uint16_t *hpre) {
  const uint16_t *got[8] = {Qb, Kb, Vb, Ya, Gb, Ub, Ym, Hout};
  static const size_t sizes[8] = {LM * LD, LM * LKV, LM * LKV, LM * LD,
                                  LM * LF, LM * LF,  LM * LD,  LM * LD};
  static const char *stages[9] = {"q", "k", "v", "o", "gate", "up", "down", "hout", "update"};
  if (!mxmm_hardware_baseline_ready && mxmm_warmup && mxmm_trial == 0 && mxmm_variant[0] == 'b') {
    for (int i = 0; i < 8; i++)
      memcpy(mxmm_hardware_baseline + mxmm_numeric_ref_offsets[i] / 2, got[i],
             sizes[i] * sizeof(uint16_t));
    mxmm_hardware_baseline_ready = 1;
    printf("MXMM_HARDWARE_BASELINE_FROZEN {\"case_id\":\"%s\",\"words\":%u}\n", case_id,
           (unsigned)MXMM_NUMERIC_WORDS);
  }
  int bad = 0;
  for (int comparison = 0; comparison < 2; comparison++)
    for (int stage = 0; stage < 9; stage++) {
      int i = stage == 8 ? 7 : stage;
      const uint16_t *ref =
          comparison ? mxmm_hardware_baseline + mxmm_numeric_ref_offsets[i] / 2
                     : (const uint16_t *)(MXMM_NUMERIC_REF_BLOB + mxmm_numeric_ref_offsets[i]);
      mxmm_numeric_result r = mxmm_numeric_compare(got[i], ref, stage == 8 ? hpre : NULL, sizes[i]);
      if (comparison && !mxmm_hardware_baseline_ready) {
        r.finite = 0;
        r.pass = 0;
        r.relative_l2_ppm = UINT64_MAX;
      }
      bad |= !r.pass;
      printf(
          "MXMM_NUMERIC_GATE {\"case_id\":\"%s\",\"comparison\":\"%s\",\"stage\":\"%s\",\"relative_l2_ppm\":%llu,\"finite\":%s,\"pass\":%s,\"bit_mismatches\":%lu}\n",
          case_id, comparison ? "hardware_baseline" : "spike", stages[stage],
          (unsigned long long)r.relative_l2_ppm, r.finite ? "true" : "false",
          r.pass ? "true" : "false", (unsigned long)r.bit_mismatches);
    }
  return bad;
}
#endif
static int mxmm_layer_project(int i, const uint8_t *a, const uint8_t *sc, int m, int block,
                              uint16_t *out) {
  uint64_t pc = mxmm_pack_cycles, mc = mxmm_mesh_cycles, t = read_cycles();
  int rc = mxmm_layer_project_fp8(&projections, i, a, sc, m, block, out);
  mxmm_projection_elapsed[i] = read_cycles() - t;
  mxmm_projection_packing[i] = mxmm_pack_cycles - pc;
  mxmm_projection_mesh[i] = mxmm_mesh_cycles - mc;
  return rc;
}
static int mxmm_run_once(const char *case_id) {
  e2e_rw = (uint8_t *)E2E_AT(0, uint8_t);
  __asm__ volatile("" : "+r"(e2e_rw));
  const uint16_t *H_PRE = E2E_AT(E2E_OFF_H_PRE, uint16_t);
  printf(
      "llama_layer_micromix: TinyLlama layer, %d tokens after a %d-token cache (causal over %d keys), D=%d F=%d, "
      "%d heads / %d kv heads, VPU stages and mixed mesh projections; CPU activation packing included\n",
      LM, E2E_SK, ATTN_SK, LD, LF, LNH, E2E_NKV);
  printf(
      "MXMM schedule nc=%d run=%d fence_groups=%d; formats K32 groups4/6/8=%d/%d/%d; weights=raw TinyLlama layer5\n",
      layer_schedule.nc, layer_schedule.run, layer_schedule.fence_groups, mxmm_weights[0].nb4,
      mxmm_weights[0].nb6, mxmm_weights[0].nb8);
  attn_print_config();
  MXMM_PROGRESS("entry.flush.before");
  gemmini_flush(0);
  MXMM_PROGRESS("entry.flush.after");
#ifndef MXMM_SKIP_ENTRY_SCALE_RESET
  MXMM_PROGRESS("entry.scale_reset.before");
  scale_halves_reset();
  MXMM_PROGRESS("entry.scale_reset.issued");
#else
  MXMM_PROGRESS("entry.scale_reset.skipped");
#endif
  MXMM_PROGRESS("entry.scale_reset.fence.before");
  gemmini_fence();
  MXMM_PROGRESS("entry.scale_reset.fence.after");
  mxn_loops = 0;
  mxmm_pack_cycles = 0;
  mxmm_mesh_cycles = 0;
  cur_pass = cur_kv = 0;
  fmask = 0;
  n_mm = mm_next = 0;
  memset(mxmm_projection_elapsed, 0, sizeof(mxmm_projection_elapsed));
  memset(mxmm_projection_packing, 0, sizeof(mxmm_projection_packing));
  memset(mxmm_projection_mesh, 0, sizeof(mxmm_projection_mesh));
  uint64_t ph[PH_N], tm;
  int bad = 0, projmis = 0;
  gemmini_fence();
  const uint64_t inst_start = mxmm_instret();
  const uint64_t t_start = read_cycles();
  tm = t_start;
#ifdef E2E_DEBUG
#define PHASE_END(p)                                                                               \
  do {                                                                                             \
    gemmini_fence();                                                                               \
    const uint64_t t_ = read_cycles();                                                             \
    ph[p] = t_ - tm;                                                                               \
    tm = t_;                                                                                       \
    e2e_dbg(p);                                                                                    \
  } while (0)
#else
#define PHASE_END(p)                                                                               \
  do {                                                                                             \
    gemmini_fence();                                                                               \
    const uint64_t t_ = read_cycles();                                                             \
    ph[p] = t_ - tm;                                                                               \
    tm = t_;                                                                                       \
  } while (0)
#endif

  MXMM_PROGRESS("rms1.before");
  rms_phase(H_PRE, 0, 0, E2E_AT(E2E_OFF_W_IN_LN, uint16_t), xn1_c, xn1_sr);
  MXMM_PROGRESS("rms1.commands_issued");
  PHASE_END(PH_RMS1);
  MXMM_PROGRESS("rms1.after");
  scales_t(xn1_s, xn1_sr, LM, LD / 32, 8);
  bad |= mxmm_layer_project(0, xn1_c, xn1_s, LM, 0, Qb);
  bad |= mxmm_layer_project(1, xn1_c, xn1_s, LM, 0, Kb);
  bad |= mxmm_layer_project(2, xn1_c, xn1_s, LM, 0, Vb);
  MXMM_PROGRESS("qkv.commands_issued");
  PHASE_END(PH_QKV);
  MXMM_PROGRESS("qkv.after");

  // CPU work stays out of the matmul / attention phases: the CPU issues their commands and would delay the mesh
  rope_phase(Qb, LD, QH, 8, LM / 16, E2E_AT(E2E_OFF_ROPE_CQ, uint16_t),
             E2E_AT(E2E_OFF_ROPE_SQ, uint16_t), q1_c, q2_c, q1_sr, q2_sr);
  rope_phase(Kb, LKV, KH, LM / 2, 1, E2E_AT(E2E_OFF_ROPE_CK, uint16_t),
             E2E_AT(E2E_OFF_ROPE_SK, uint16_t), k1_c, k2_c, k1_sr, k2_sr);
  gemmini_fence(); // the RoPE banks are reused by the V requant
  vt_transpose();
  vt_requant();
  gemmini_fence();
  cache_append();
  MXMM_PROGRESS("rope.commands_issued");
  PHASE_END(PH_ROPE);
  MXMM_PROGRESS("rope.after");

  scale_halves_reset();
  for (int p = 0; p < NPASS; p++) {
    cur_pass = p;
    cur_kv = p / (NPASS / E2E_NKV);
    q_scales_of_pass(p);
    attn_load_q_scales();
    fmask = 0;
    run_pipelined();
  }
  MXMM_PROGRESS("attn.commands_issued");
  PHASE_END(PH_ATTN);
  MXMM_PROGRESS("attn.after");

  MXMM_PROGRESS("postattn.scales.before");
  for (int p = 0; p < NPASS; p++) // o_sr[p]: [2][128]; row = head (2p + row/64), token row % 64
    for (int hh = 0; hh < ATTN_HEADS; hh++)
      for (int g = 0; g < 2; g++)
        for (int m = 0; m < LM; m++)
          o_s[((ATTN_HEADS * p + hh) * 2 + g) * LM + m] = o_sr[p][g * ATTN_SQ + hh * LM + m];
  MXMM_PROGRESS("postattn.scales.after");
  MXMM_PROGRESS("postattn.scale_reset.before");
  scale_halves_reset();
  MXMM_PROGRESS("postattn.scale_reset.after_fences");
  MXMM_PROGRESS("oproj.before");
  bad |= mxmm_layer_project(3, o_c, o_s, LM, 64, Ya);
  MXMM_PROGRESS("oproj.commands_issued");
  PHASE_END(PH_OPROJ);
  MXMM_PROGRESS("oproj.after");

  rms_phase(H_PRE, Ya, Hmid, E2E_AT(E2E_OFF_W_POST_LN, uint16_t), xn2_c, xn2_sr);
  MXMM_PROGRESS("rms2.commands_issued");
  PHASE_END(PH_RMS2);
  MXMM_PROGRESS("rms2.after");
  scales_t(xn2_s, xn2_sr, LM, LD / 32, 8);
  bad |= mxmm_layer_project(4, xn2_c, xn2_s, LM, 0, Gb);
  bad |= mxmm_layer_project(5, xn2_c, xn2_s, LM, 0, Ub);
  MXMM_PROGRESS("gu.commands_issued");
  PHASE_END(PH_GU);
  MXMM_PROGRESS("gu.after");

  swiglu_phase();
  MXMM_PROGRESS("swiglu.commands_issued");
  PHASE_END(PH_SWIGLU);
  MXMM_PROGRESS("swiglu.after");
  for (int s = 0; s < 2; s++)
    scales_t(h_s + s * (FH / 32) * LM, h_sr[s], LM, FH / 32, 8);
  bad |= mxmm_layer_project(6, h_c, h_s, LM, 0, Ym);
  MXMM_PROGRESS("down.commands_issued");
  PHASE_END(PH_DOWN);
  MXMM_PROGRESS("down.after");

  residual_phase(Hmid, Ym, Hout);
  MXMM_PROGRESS("res2.commands_issued");
  PHASE_END(PH_RES2);
  MXMM_PROGRESS("res2.after");
  const uint64_t total = read_cycles() - t_start;
  const uint64_t inst_delta = mxmm_instret() - inst_start;
  if (bad)
    printf("a matmul was rejected (shape/alignment)\n");

  // ---- hashes (Spike is the bit-exact reference) ----
  struct {
    const char *n;
    const void *p;
    size_t sz;
  } hs[] = {
      {"xn1", xn1_c, sizeof(xn1_c)}, {"q", q1_c, sizeof(q1_c)},    {"q2", q2_c, sizeof(q2_c)},
      {"k", k1_c, sizeof(k1_c)},     {"v", vt_c, sizeof(vt_c)},    {"o", o_c, sizeof(o_c)},
      {"yattn", Ya, sizeof(Ya)},     {"hmid", Hmid, sizeof(Hmid)}, {"xn2", xn2_c, sizeof(xn2_c)},
      {"h", h_c, sizeof(h_c)},       {"ymlp", Ym, sizeof(Ym)},     {"hout", Hout, sizeof(Hout)}};
  const int nh = sizeof(hs) / sizeof(hs[0]);
  uint64_t hv[16];
  int hmis = 0;
  for (int i = 0; i < nh; i++) {
    hv[i] = fnv(hs[i].p, hs[i].sz);
#if E2E_HAVE_EXPECT
    const int ok = hv[i] == E2E_EXPECT[i];
    hmis += !ok;
    printf("hash %-6s %016llx %s\n", hs[i].n, (unsigned long long)hv[i],
           ok ? "match" : "MISMATCH vs Spike");
#else
    printf("hash %-6s %016llx\n", hs[i].n, (unsigned long long)hv[i]);
#endif
  }
#if !E2E_HAVE_EXPECT
  printf("static const uint64_t E2E_EXPECT[%d] = {", nh);
  for (int i = 0; i < nh; i++)
    printf("%s0x%016llxULL", i ? ", " : "", (unsigned long long)hv[i]);
  printf("};\n");
#endif

#ifdef E2E_DUMP_H // Spike: dump h for data/llama_layer_e2e_h_expect.h (gen/dump_h_expect.py)
  for (int i = 0; i < LM * LF; i += 64) {
    printf("HDUMP ");
    for (int k = 0; k < 64; k++)
      printf("%02x", h_c[i + k]);
    printf("\n");
  }
#endif
#if 0 /* Original FP8 h hashes do not describe this mixed run. */
  {   // every h code that differs from Spike's, with the G / U inputs that produced it
#include "llama_layer_e2e_h_expect.h"
    int nd = 0, nz = 0, shown = 0;
    for (int i = 0; i < LM * LF; i++) {
      if (h_c[i] == E2E_H_EXPECT[i]) continue;
      nd++;
      const int zero_sign = (h_c[i] | E2E_H_EXPECT[i]) == 0x80 && (h_c[i] & 0x7f) == 0 && (E2E_H_EXPECT[i] & 0x7f) == 0;
      nz += zero_sign;
      if (shown < 64 || (!zero_sign && shown < 128)) {
        const int t = i / LF, c = i % LF;
        printf("  h diff token %2d col %4d: device %02x spike %02x%s  G %04x (%g) U %04x (%g) scale %u\n", t, c, h_c[i],
               E2E_H_EXPECT[i], zero_sign ? " (+0/-0)" : "", Gb[i], bf(Gb[i]), Ub[i], bf(Ub[i]),
               h_sr[c / FH][SRI(t, (c % FH) / 32, FH / 32, 8)]);
        shown++;
      }
    }
    printf("h vs Spike: %d of %d codes differ, %d of them only in the sign of zero\n", nd, LM * LF, nz);
  }
#endif
  { // h: zero codes by sign, and a hash with -0 (0x80) folded into +0 (a sign-of-zero difference changes no product)
    static uint8_t hz[LM * LF] A64;
    int z0 = 0, z8 = 0;
    for (int i = 0; i < LM * LF; i++) {
      z0 += h_c[i] == 0x00;
      z8 += h_c[i] == 0x80;
      hz[i] = h_c[i] == 0x80 ? 0 : h_c[i];
    }
    printf("h codes: +0 %d, -0 %d; hash with -0 -> +0: %016llx\n", z0, z8,
           (unsigned long long)fnv(hz, sizeof(hz)));
  }
  {
    struct {
      const char *name;
      const uint8_t *p;
      size_t n;
    } xs[] = {{"xn1", xn1_c, sizeof(xn1_c)},
              {"o", o_c, sizeof(o_c)},
              {"xn2", xn2_c, sizeof(xn2_c)},
              {"h", h_c, sizeof(h_c)}};
    for (int z = 0; z < 4; z++) {
      unsigned max = 0;
      size_t over4 = 0, over6 = 0;
      for (size_t i = 0; i < xs[z].n; i++) {
        unsigned c = xs[z].p[i] & 127;
        if (c > max)
          max = c;
        over4 += c > 0x4c;
        over6 += c > 0x5e;
      }
      printf(
          "MXMM_ACTIVATION_RANGE %s max_code=%02x max_normalized_milli=%d above_fp4_max=%lu above_fp6_max=%lu\n",
          xs[z].name, max, (int)(1000 * mxmm_decode(max, 8)), (unsigned long)over4,
          (unsigned long)over6);
    }
  }
  // ---- accuracy: vs fp64 references (from the BF16 h_pre) and vs TinyLlama's own layer output ----
  const int n = LM * LD;
  printf("accuracy  xn1    rel_fro %6d ppm vs fp64 (MX dequantized)\n",
         (int)(1e6f * rel_mx(E2E_AT(E2E_OFF_REF_XN1, float), n, get_xn1)));
  printf("accuracy  O      rel_fro %6d ppm vs fp64 (MX dequantized, all heads)\n",
         (int)(1e6f * rel_mx(E2E_AT(E2E_OFF_REF_O, float), n, get_o)));
  printf("accuracy  Yattn  rel_fro %6d ppm vs fp64\n",
         (int)(1e6f * rel_bf16(Ya, E2E_AT(E2E_OFF_REF_YATTN, float), 0, n)));
  printf("accuracy  h_mid  rel_fro %6d ppm vs fp64 (update h_mid - h_pre: %d ppm)\n",
         (int)(1e6f * rel_bf16(Hmid, E2E_AT(E2E_OFF_REF_HMID, float), 0, n)),
         (int)(1e6f * rel_upd(Hmid, E2E_AT(E2E_OFF_REF_HMID, float), H_PRE, n)));
  printf("accuracy  Ymlp   rel_fro %6d ppm vs fp64\n",
         (int)(1e6f * rel_bf16(Ym, E2E_AT(E2E_OFF_REF_YMLP, float), 0, n)));
  printf(
      "accuracy  h_out  rel_fro %6d ppm vs fp64, %d ppm vs TinyLlama hidden_states (layer update h_out - h_pre: "
      "%d ppm vs fp64, %d ppm vs TinyLlama)\n",
      (int)(1e6f * rel_bf16(Hout, E2E_AT(E2E_OFF_REF_HOUT, float), 0, n)),
      (int)(1e6f * rel_bf16(Hout, E2E_AT(E2E_OFF_H_MODEL, float), 0, n)),
      (int)(1e6f * rel_upd(Hout, E2E_AT(E2E_OFF_REF_HOUT, float), H_PRE, n)),
      (int)(1e6f * rel_upd(Hout, E2E_AT(E2E_OFF_H_MODEL, float), H_PRE, n)));

  {
    struct {
      const char *name;
      const uint16_t *p;
      int len;
    } ps[] = {{"q_proj", Qb, LM * LD},   {"k_proj", Kb, LM * LKV},   {"v_proj", Vb, LM * LKV},
              {"o_proj", Ya, LM * LD},   {"gate_proj", Gb, LM * LF}, {"up_proj", Ub, LM * LF},
              {"down_proj", Ym, LM * LD}};
    for (int p = 0; p < 7; p++) {
      uint64_t phash = fnv(ps[p].p, ps[p].len * 2);
      printf("MXMM_PROJ_HASH %s %016llx\n", ps[p].name, (unsigned long long)phash);
#ifdef MXMM_HAVE_PROJECTION_EXPECT
      if (phash != MXMM_PROJ_EXPECT[p]) {
        projmis++;
        printf("MXMM_PROJ_MISMATCH %s\n", ps[p].name);
      }
#endif
#ifdef MXMM_DUMP_PROJECTIONS
      for (int i = 0; i < ps[p].len; i += 32) {
        printf("MXMM_PROJECTION %s %d ", ps[p].name, i);
        for (int j = 0; j < 32; j++)
          printf("%04x", ps[p].p[i + j]);
        printf("\n");
      }
#endif
    }
  }
#ifdef MXMM_DUMP_OUTPUT
  for (int i = 0; i < LM * LD; i += 32) {
    printf("MXMM_OUTPUT %d ", i);
    for (int j = 0; j < 32; j++)
      printf("%04x", Hout[i + j]);
    printf("\n");
  }
#endif
  uint64_t ideal = 0;
  for (int p = 0; p < PH_N; p++) {
    uint64_t phase_ideal = ph_ideal[p];
    if (p == PH_QKV || p == PH_OPROJ || p == PH_GU || p == PH_DOWN) {
      const mxmm_weight *w = &mxmm_weights[p == PH_DOWN ? 6 : 0];
      phase_ideal = phase_ideal * (w->nb4 + 4 * w->nb6 + 4 * w->nb8) / (4 * (w->k / 32));
    }
    ideal += phase_ideal;
    if (phase_ideal)
      printf("PERF phase %-22s %9llu cycles (%2llu%%); mesh ideal %9llu -> util %llu%%\n",
             ph_name[p], (unsigned long long)ph[p], (unsigned long long)(ph[p] * 100 / total),
             (unsigned long long)phase_ideal, (unsigned long long)(phase_ideal * 100 / ph[p]));
    else
      printf("PERF phase %-22s %9llu cycles (%2llu%%)\n", ph_name[p], (unsigned long long)ph[p],
             (unsigned long long)(ph[p] * 100 / total));
  }
  printf("PERF layer: %llu cycles from launch to h_out; mesh ideal %llu -> util %llu.%llu%%\n",
         (unsigned long long)total, (unsigned long long)ideal,
         (unsigned long long)(ideal * 100 / total),
         (unsigned long long)(ideal * 1000 / total % 10));
  printf("MXMM_TIMING packing_cycles=%llu projection_cycles=%llu packing_included=true\n",
         (unsigned long long)mxmm_pack_cycles, (unsigned long long)mxmm_mesh_cycles);
#ifdef MXMM_NUMERIC_MODE
  int numeric_bad = mxmm_validate_numeric(case_id, H_PRE);
  printf(
      "MXMM_HASH_DIAGNOSTIC {\"case_id\":\"%s\",\"stage_mismatches\":%d,\"projection_mismatches\":%d,\"failure_policy\":\"numeric_1_percent\"}\n",
      case_id, hmis, projmis);
  const int fail = bad || numeric_bad;
#else
  const int fail = bad || hmis || projmis;
#endif
#ifdef MX_ROCKET
#define MXMM_SOURCE "fpga"
#else
#define MXMM_SOURCE "spike"
#endif
  printf(
      "MXMM_TRIAL_RESULT {\"case_id\":\"%s\",\"variant\":\"%s\",\"trial\":%d,\"warmup\":%s,\"source\":\"%s\",\"total_cycles\":%llu,\"packing_cycles\":%llu,\"mesh_cycles\":%llu,\"fast_pack\":%d,\"nc\":%d,\"run\":%d,\"fence_groups\":%d,\"projection_cycles\":[",
      case_id, mxmm_variant, mxmm_trial, mxmm_warmup ? "true" : "false", MXMM_SOURCE,
      (unsigned long long)total, (unsigned long long)mxmm_pack_cycles,
      (unsigned long long)mxmm_mesh_cycles, mxmm_fast_pack, layer_schedule.nc, layer_schedule.run,
      layer_schedule.fence_groups);
  for (int i = 0; i < 7; i++)
    printf("%s%llu", i ? "," : "", (unsigned long long)mxmm_projection_elapsed[i]);
  printf("],\"projection_pack_cycles\":[");
  for (int i = 0; i < 7; i++)
    printf("%s%llu", i ? "," : "", (unsigned long long)mxmm_projection_packing[i]);
  printf("],\"projection_mesh_cycles\":[");
  for (int i = 0; i < 7; i++)
    printf("%s%llu", i ? "," : "", (unsigned long long)mxmm_projection_mesh[i]);
  printf("],\"pass\":%s}\n", fail ? "false" : "true");
  printf("llama_layer_micromix %s\n", fail              ? "FAILED"
                                      : E2E_HAVE_EXPECT ? "PASSED"
                                                        : "DONE (no Spike hashes yet)");
  printf(
      "IRIS_REGION {\"case_id\":\"%s\",\"region\":\"prefill\",\"hart\":0,\"start_cycle\":%llu,\"end_cycle\":%llu,\"instret_delta\":%llu,\"checksum\":%llu,\"pass\":%s}\n",
      case_id, (unsigned long long)t_start, (unsigned long long)(t_start + total),
      (unsigned long long)inst_delta, (unsigned long long)fnv(Hout, sizeof(Hout)),
      fail ? "false" : "true");
  return fail;
}

#include "micromix_prefill_main.h"
