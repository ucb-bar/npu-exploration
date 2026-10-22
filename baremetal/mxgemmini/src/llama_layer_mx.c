// One whole decoder layer of a 7-13B model on MxGemmini -- the llama_layer_e2e.c flow, generalized over the model
// (llama_mx_common.MODELS: hidden / FFN size, heads, GQA groups, head dim 128, Qwen2's Q/K/V biases), the projection
// format (LMX_FMT: E4M3, FP6 E3M2 through LUTs, FP4 E2M1; attention and the KV cache always E4M3) and the mesh (DIM 4,
// 8, 16, 32: -DATTN_PARAMS_DIM -iquote dim<N>/<cfg> for the params header). Every stage on the device (mesh, VPU,
// SPAD_REQUANT); the CPU only issues commands, reorders small scale arrays, transposes V and appends to the cache.
// A 64-token prefill chunk after a KV cache (gen/gen_llama_layer_mx.py):
//
//   VPU+SR  xn1 = Q(rmsnorm(h_pre) * w_in)                    8-token streams x2, column chunks of LMX_RMS_NC
//   mesh    Q, K, V = xn1 @ Wq / Wk / Wv                      native DRAM loops (Wq/Wk columns rotary-half split)
//   VPU     (+ bq, bk, bv)
//   VPU+SR  Q, K = rope(Q), rope(K) -> E4M3;  V^T -> E4M3     new K/V written into the cache (keys LMX_SK..)
//   attn    one flash pass per query head (E4M3) over LMX_SK + LMX_M keys, causal on the last block; O -> Q(.)
//   mesh    Yattn = O @ Wo
//   VPU+SR  h_mid = h_pre + Yattn;  xn2 = Q(rmsnorm(h_mid) * w_post)
//   mesh    G, U = xn2 @ Wg / Wu
//   VPU+SR  H = Q(silu(G) * U)
//   mesh    Ymlp = H @ Wd
//   VPU     h_out = h_mid + Ymlp
// Q(.) = the projection format's SPAD_REQUANT (E4M3; FP4; FP6: E3M2 -> the finder's index in the tensor's table).
//
// Timed from the first command to h_out in DRAM; utilization = the layer's mesh-ideal cycles at the format's rate /
// that time. Phases are fenced. 22 stage hashes are checked against the rtl_exact golden (gen/golden_llama_layer_mx.py,
// LMX_EXPECT), which also fixes the two chunkings that change numbers (LMX_RMS_NC, the attention key blocks).
#include LMX_DATA_H
#include LMX_EXPECT_H

#define ATTN_HEADER LMX_DATA_H
#define ATTN_SQ     LMX_M            // one query head per pass
#define ATTN_SK     (LMX_SK + LMX_M)  // cache + the chunk
#define ATTN_D      LMX_HD
#define ATTN_HEADS  1
#define ATTN_CAUSAL 1
#define ATTN_SCALE_IN_Q 1            // Wq / bq carry 1/sqrt(head_dim)
#define BK          LMX_BK
#define BK_FIRST    LMX_BK_FIRST
#define BK_LAST     LMX_BK_LAST      // the last key block = the chunk itself (one mask ADD)
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
static int cur_head, cur_kv;
static uint8_t q_sc_pass[ATTN_D / 32][ATTN_SQ] __attribute__((aligned(64)));
// the cache is written (keys LMX_SK..): through a pointer the compiler cannot trace to the const blob symbol
static uint8_t *lmx_rw;
#define KTC_  (lmx_rw + LMX_OFF_KT_CACHE)
#define KTS_  (lmx_rw + LMX_OFF_KT_SCALES)
#define VC_   (lmx_rw + LMX_OFF_V_CACHE)
#define VS_   (lmx_rw + LMX_OFF_V_SCALES)
#define ATTN_Q    ((const uint8_t *) 0)
#define ATTN_QS   q_sc_pass
#define ATTN_KT   ((const uint8_t (*)[ATTN_SK]) (KTC_ + (size_t) cur_kv * ATTN_D * ATTN_SK))
#define ATTN_KTS  ((const uint8_t (*)[ATTN_SK]) (KTS_ + (size_t) cur_kv * (ATTN_D / 32) * ATTN_SK))
#define ATTN_V    ((const uint8_t (*)[ATTN_D]) (VC_ + (size_t) cur_kv * ATTN_SK * ATTN_D))
#define ATTN_VS   ((const uint8_t (*)[ATTN_D]) (VS_ + (size_t) cur_kv * (ATTN_SK / 32) * ATTN_D))
#define ATTN_OREF ((const uint32_t *) LMX_AT(LMX_OFF_REF_O, uint32_t))
#define MASK_BF16 LMX_AT(LMX_OFF_MASK, uint16_t)
static void lmx_load_q(void);
static void lmx_store_o(void);
#define ATTN_LOAD_Q  lmx_load_q
#define ATTN_STORE_O lmx_store_o
#include "attn_flash.c"
#include "mx_native.h"

#if defined(MX_ROCKET) || defined(SPIKE_SIM)
typedef char lmx_expect_dim_ok[(LMX_EXPECT_DIM == DIM) ? 1 : -1];   // the golden was run for this mesh

#define LM   LMX_M
#define LD   LMX_D
#define LF   LMX_F
#define LHD  LMX_HD
#define LNH  LMX_NH
#define LKV  (LMX_NKV * LMX_HD)
#define H2   (LHD / 2)            // one rotary half of a head
#define QH   (LD / 2)             // one rotary half of Q: x1 (or x2) of every head
#define KH   (LKV / 2)
#define PQ   (LMX_FMT ? 2 : 1)    // projection operand rows per byte (FP4 / FP6 LUT: row pairs); attn_flash's QT is 1
#define RATE (PQ * PQ)            // mesh MACs per cycle relative to E4M3 single (the quad mesh)
#define SPR(b, r) ((uint32_t) ((b) * BANK_ROWS + (r)))
#define BANK_B (BANK_ROWS * DIM)  // bytes per bank
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ---- DRAM buffers ----
#define A64 __attribute__((aligned(64)))
static uint8_t  xn1_c[LM / PQ * LD] A64, xn1_sr[LM * LD / 32] A64, xn1_s[LD / 32 * LM] A64;
static uint16_t Qb[LM * LD] A64, Kb[LM * LKV] A64, Vb[LM * LKV] A64;
#if LMX_BIAS
static uint16_t Vbb[LM * LKV] A64;   // V + bv
#endif
static uint8_t  q1_c[LM * QH] A64, q2_c[LM * QH] A64, q1_sr[LM * QH / 32] A64, q2_sr[LM * QH / 32] A64;
static uint8_t  k1_c[LM * KH] A64, k2_c[LM * KH] A64, k1_sr[LM * KH / 32] A64, k2_sr[LM * KH / 32] A64;
static uint16_t vt_b[LKV * LM] A64;
static uint8_t  vt_c[LKV * LM] A64, vt_sr[LKV * LM / 32] A64;
static uint8_t  o_c[LM / PQ * LD] A64, o_s[LD / 32 * LM] A64;
static uint16_t Ya[LM * LD] A64, Hmid[LM * LD] A64;
static uint8_t  xn2_c[LM / PQ * LD] A64, xn2_sr[LM * LD / 32] A64, xn2_s[LD / 32 * LM] A64;
static uint16_t Gb[LM * LF] A64, Ub[LM * LF] A64;
static uint8_t  h_c[LM / PQ * LF] A64, h_sr[LM * LF / 32] A64, h_s[LF / 32 * LM] A64;
static uint16_t Ym[LM * LD] A64, Hout[LM * LD] A64;

// contiguous DRAM <-> contiguous spad rows, up to DIM rows per command (config_ld / config_st = DIM set by the caller)
static void mv_in(const void *src, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvin((const uint8_t *) src + (size_t) r * DIM, sp + r, DIM, MIN(DIM, rows - r));
}
static void mv_out(void *dst, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvout((uint8_t *) dst + (size_t) r * DIM, sp + r, DIM, MIN(DIM, rows - r));
}
// byte of (row m, group g) in SR scales written by 8-row commands: command (m / 8) covers every group, [groups][8]
#define SRI(m, g, groups) (((m) - (m) % 8) * (groups) + (g) * 8 + (m) % 8)
// 8-row SR scales -> the native loop's A-scale layout [cols/32][rows]
static void scales_t(uint8_t *dst, const uint8_t *src, int rows, int groups) {
  for (int t = 0; t < rows; t += 8)
    for (int g = 0; g < groups; g++)
      for (int r = 0; r < 8; r++) dst[g * rows + t + r] = src[t * groups + g * 8 + r];
}
static void scale_halves_reset(void) {   // free every scale half (as llama_layer_e2e.c)
  gemmini_mxquant_config_mvout((uint64_t) scale_sink, 1, 1, 1, 1, 1, 1);
  gemmini_mxquant_config_mvout((uint64_t) scale_sink, 1, 1, 1, 0, 0, 1);
}

// ---- the projection format: its SPAD_REQUANT, matmul and configuration ----
#if LMX_FMT == 0
#define SR_FMT(dst, src, m, n, tiled, sc) gemmini_spad_requant(dst, src, m, n, tiled, (uint64_t) (sc), 0)
#define MATMUL(A, B, C, ldc, As, Bs, Bl, Al, M, K, N) mxn_matmul(A, K, B, N, C, ldc, As, M, Bs, N, M, K, N)
static void fmt_cfg(void) {}
#elif LMX_FMT == 2
#define SR_FMT(dst, src, m, n, tiled, sc) gemmini_spad_requant_fp4(dst, src, m, n, tiled, (uint64_t) (sc), 0)
#define MATMUL(A, B, C, ldc, As, Bs, Bl, Al, M, K, N) mxn_matmul_fp4(A, K, B, (N) / 2, C, ldc, As, M, Bs, N, M, K, N)
static void fmt_cfg(void) {}
#else
#define LUT_BYTES 12   // one FP6 table: 16 x 6 bit
#define SR_FMT(dst, src, m, n, tiled, sc) gemmini_spad_requant_lut(dst, src, m, n, tiled, (uint64_t) (sc), 0, 1, 0, LMX_G)
#define MATMUL(A, B, C, ldc, As, Bs, Bl, Al, M, K, N) \
  mxn_matmul_lut_ex(A, K, B, (N) / 2, C, ldc, As, M, Bs, N, Bl, LUT_BYTES, LMX_G, M, K, N, Al, 1)
static void fmt_cfg(void) {   // FP6 E3M2 via LUT on A and B, BF16 out (as mlp_proj_fp6.c), and the matmuls' G
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16) | (3 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | ((WEIGHT_STATIONARY) << 2) | CONFIG_EX,
    ((uint64_t)(1) << 48), k_CONFIG);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 3, 1);
  gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, LMX_G);
}
#endif
// the C table the requantizer's finder reads (FP6), one copy per 2^G rows
#if LMX_FMT == 1
#define OUT_TABLE(n) gemmini_mx_load_lut((uint64_t) LMX_AT(LMX_OFF_L_##n, uint8_t), 4, 2)
#else
#define OUT_TABLE(n) ((void) 0)
#endif
#if LMX_FMT == 1
#define WLUT(n) LMX_AT(LMX_OFF_W##n##_L, uint8_t)
#define ALUT(n) LMX_AT(LMX_OFF_L_##n, uint8_t)
#else
#define WLUT(n) ((const uint8_t *) 0)
#define ALUT(n) ((const uint8_t *) 0)
#endif
#define WC(n) LMX_AT(LMX_OFF_W##n##_C, uint8_t)
#define WS(n) LMX_AT(LMX_OFF_W##n##_S, uint8_t)

// ---- RMSNorm (optionally after a residual add) -> Q(.): 8-token streams s = 0, 1 in bank pairs (2s, 2s+1), column
// chunks of NC: pass A sums x^2 per chunk (RSUM) and adds the partial sums (BF16), pass B normalizes and requantizes.
// bank 2s: x [8][NC], w [NC]; bank 2s+1: y [8][NC], sum, partial (8 rows each), codes ----
#define NC  LMX_RMS_NC
#define NCR (NC / VL)                   // spad rows of one token's chunk
#define RX(s) SPR(2 * (s), 0)
#define RW(s) SPR(2 * (s), 8 * NCR)
#define RY(s) SPR(2 * (s) + 1, 0)
#define RS(s) SPR(2 * (s) + 1, 8 * NCR)
#define RP(s) SPR(2 * (s) + 1, 8 * NCR + 8)
#define RC(s) SPR(2 * (s) + 1, 8 * NCR + 16)
typedef char lmx_rms_fits[(LD % NC == 0 && NC % 128 == 0 && 9 * NCR <= BANK_ROWS && NCR <= 1024 &&
                           8 * NCR + 16 + 8 * NC / (PQ * DIM) <= BANK_ROWS) ? 1 : -1];
static void rms_phase(const uint16_t *x, const uint16_t *res, uint16_t *hsum, const uint16_t *w, uint8_t *codes,
                      uint8_t *sraw) {
  const uint16_t inv_d = vpu_f_to_bf16(1.0f / LD), eps = vpu_f_to_bf16(LMX_EPS);
  const int nch = LD / NC, n = 8 * NCR;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  if (nch == 1)
    for (int s = 0; s < 2; s++) mv_in(w, RW(s), NCR);
  for (int c = 0; c < LM / 16; c++) {
    const int t[2] = {c * 16, c * 16 + 8};
    for (int k = 0; k < nch; k++) {   // pass A: sum of squares
      for (int s = 0; s < 2; s++)
        for (int m = 0; m < 8; m++) {
          mv_in(x + (size_t) (t[s] + m) * LD + k * NC, RX(s) + m * NCR, NCR);
          if (res) mv_in(res + (size_t) (t[s] + m) * LD + k * NC, RY(s) + m * NCR, NCR);
        }
      if (res) {
        for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_ADD, RX(s), RX(s), RY(s), n);
        for (int s = 0; s < 2; s++)
          for (int m = 0; m < 8; m++) mv_out(hsum + (size_t) (t[s] + m) * LD + k * NC, RX(s) + m * NCR, NCR);
      }
      for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, RY(s), RX(s), RX(s), n);
      for (int s = 0; s < 2; s++) gemmini_vpu_reduce(VPU_RSUM, k ? RP(s) : RS(s), RY(s), n, NCR);
      if (k)
        for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_ADD, RS(s), RS(s), RP(s), 8);
    }
    for (int s = 0; s < 2; s++) gemmini_vpu_scalar(VPU_MULS, RS(s), RS(s), inv_d, 8);
    for (int s = 0; s < 2; s++) gemmini_vpu_scalar(VPU_ADDS, RS(s), RS(s), eps, 8);
    for (int s = 0; s < 2; s++) gemmini_vpu_unary(VPU_RSQRT, RS(s), RS(s), 8);
    for (int k = 0; k < nch; k++) {   // pass B: x * (1/rms) * w -> Q(.)
      if (nch > 1) {
        for (int s = 0; s < 2; s++) {
          mv_in(w + k * NC, RW(s), NCR);
          for (int m = 0; m < 8; m++) {
            mv_in(x + (size_t) (t[s] + m) * LD + k * NC, RX(s) + m * NCR, NCR);
            if (res) mv_in(res + (size_t) (t[s] + m) * LD + k * NC, RY(s) + m * NCR, NCR);
          }
        }
        if (res)
          for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_ADD, RX(s), RX(s), RY(s), n);
      }
      for (int s = 0; s < 2; s++) gemmini_vpu_bcast(VPU_MUL, RY(s), RX(s), RS(s), n, NCR);
      for (int m = 0; m < 8; m++)
        for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, RY(s) + m * NCR, RY(s) + m * NCR, RW(s), NCR);
      for (int s = 0; s < 2; s++) SR_FMT(RC(s), RY(s), 8, NC, 0, sraw + t[s] * (LD / 32) + (k * NC / 32) * 8);
      for (int s = 0; s < 2; s++)   // code row r (a token, or a token pair): NC bytes
        for (int r = 0; r < 8 / PQ; r++) mv_out(codes + (size_t) (t[s] / PQ + r) * LD + k * NC, RC(s) + r * (NC / DIM), NC / DIM);
    }
  }
}

// largest multiple of 128 dividing n with per * it <= bytes (0: none)
static int chunk_div(int n, int per, int bytes) {
  int best = 0;
  for (int c = 128; c <= n; c += 128)
    if (n % c == 0 && per * c <= bytes) best = c;
  return best;
}

// ---- (Q / K bias and) RoPE on the two rotary halves (x1, x2 of every head, contiguous after the Wq/Wk column split):
//   x1' = x1 * cos - x2 * sin,  x2' = x2 * cos + x1 * sin  -> E4M3 (SR per half; a 32-group = half a head's half)
// 8-token streams s = 0, 1, column chunks of HC: bank 2s x1, x2, the codes (and the bias chunk); bank 2s+1 cos, sin
// and two temporaries ----
static int rope_phase(const uint16_t *x, int ld, int half, const uint16_t *cs, const uint16_t *sn, const uint16_t *bias,
                      uint8_t *c1, uint8_t *c2, uint8_t *s1, uint8_t *s2) {
  const int hc = chunk_div(half, 64, BANK_B), hr = hc / VL, n = 8 * hr, cr = 8 * hc / DIM;
  if (!hc || 2 * n + 2 * cr + 2 * hr > BANK_ROWS || hr > 1024) { printf("rope_phase: no chunk fits (half %d)\n", half); return 1; }
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
#define X1 SPR(2 * s, 0)
#define X2 SPR(2 * s, n)
#define C1 SPR(2 * s, 2 * n)
#define C2 SPR(2 * s, 2 * n + cr)
#define B1 SPR(2 * s, 2 * n + 2 * cr)
#define B2 SPR(2 * s, 2 * n + 2 * cr + hr)
#define TC SPR(2 * s + 1, 0)
#define TS SPR(2 * s + 1, n)
#define TB SPR(2 * s + 1, 2 * n)
#define TT SPR(2 * s + 1, 3 * n)
  for (int c0 = 0; c0 < half; c0 += hc) {
    if (bias)
      for (int s = 0; s < 2; s++) { mv_in(bias + c0, B1, hr); mv_in(bias + half + c0, B2, hr); }
    for (int tg = 0; tg < LM / 16; tg++) {
      int t[2];
      for (int s = 0; s < 2; s++) {
        t[s] = tg * 16 + 8 * s;
        for (int m = 0; m < 8; m++) {
          mv_in(x + (size_t) (t[s] + m) * ld + c0, X1 + m * hr, hr);
          mv_in(x + (size_t) (t[s] + m) * ld + half + c0, X2 + m * hr, hr);
          mv_in(cs + (size_t) (t[s] + m) * half + c0, TC + m * hr, hr);
          mv_in(sn + (size_t) (t[s] + m) * half + c0, TS + m * hr, hr);
        }
      }
      if (bias)
        for (int m = 0; m < 8; m++)
          for (int s = 0; s < 2; s++) {
            gemmini_vpu_binary(VPU_ADD, X1 + m * hr, X1 + m * hr, B1, hr);
            gemmini_vpu_binary(VPU_ADD, X2 + m * hr, X2 + m * hr, B2, hr);
          }
      for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, TB, X2, TS, n);
      for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, TT, X1, TS, n);
      for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, X1, X1, TC, n);
      for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_SUB, X1, X1, TB, n);
      for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, X2, X2, TC, n);
      for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_ADD, X2, X2, TT, n);
      for (int s = 0; s < 2; s++) {
        gemmini_spad_requant(C1, X1, 8, hc, 0, (uint64_t) (s1 + t[s] * (half / 32) + (c0 / 32) * 8), 0);
        gemmini_spad_requant(C2, X2, 8, hc, 0, (uint64_t) (s2 + t[s] * (half / 32) + (c0 / 32) * 8), 0);
      }
      for (int s = 0; s < 2; s++)
        for (int m = 0; m < 8; m++) {
          mv_out(c1 + (size_t) (t[s] + m) * half + c0, C1 + m * (hc / DIM), hc / DIM);
          mv_out(c2 + (size_t) (t[s] + m) * half + c0, C2 + m * (hc / DIM), hc / DIM);
        }
    }
  }
#undef X1
#undef X2
#undef C1
#undef C2
#undef B1
#undef B2
#undef TC
#undef TS
#undef TB
#undef TT
  return 0;
}

#if LMX_BIAS
// ---- V + bv (8 tokens at a time; bank 0 the bias row, bank 1 the tokens) ----
typedef char lmx_vbias_fits[(9 * (LKV / VL) <= BANK_ROWS) ? 1 : -1];
static void vbias_phase(void) {
  const int r = LKV / VL;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  mv_in(LMX_AT(LMX_OFF_BV, uint16_t), SPR(0, 0), r);
  for (int tg = 0; tg < LM / 8; tg++) {
    mv_in(Vb + (size_t) tg * 8 * LKV, SPR(1, 0), 8 * r);
    for (int m = 0; m < 8; m++) gemmini_vpu_binary(VPU_ADD, SPR(1, m * r), SPR(1, m * r), SPR(0, 0), r);
    mv_out(Vbb + (size_t) tg * 8 * LKV, SPR(1, 0), 8 * r);
  }
}
#define VSRC Vbb
#else
#define VSRC Vb
#endif

// ---- V of the chunk -> the cache's E4M3 layout (blocks of 32 keys per head dim): V^T rows through SR, VR rows at a
// time (VR * LM / 32 <= 2048 scale bytes per SR); scales chunk-major: chunk q's [LM/32][VR] at vt_sr + q * 2 * VR ----
#define VR_MAX ((3 * LM * 1024 <= BANK_NUM * BANK_B) ? 1024 : 512)
#define VR ((LKV % VR_MAX) == 0 ? VR_MAX : LKV < VR_MAX ? LKV : 256)
typedef char lmx_vt_fits[(LKV % VR == 0 && VR % 16 == 0 && VR * LM / 32 <= 2048 && 3 * VR * LM <= BANK_NUM * BANK_B) ? 1 : -1];
static void vt_transpose(void) {   // V [LM][LKV] -> V^T [LKV][LM] BF16 (CPU)
  for (int t = 0; t < LM; t++)
    for (int c = 0; c < LKV; c++) vt_b[c * LM + t] = VSRC[t * LKV + c];
}
static void vt_requant(void) {
  const uint32_t cod = VR * LM * 2 / DIM;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  for (int q = 0; q < LKV / VR; q++) {
    mv_in(vt_b + (size_t) q * VR * LM, 0, VR * LM * 2 / DIM);
    gemmini_spad_requant(cod, 0, VR, LM, 0, (uint64_t) (vt_sr + (size_t) q * (LM / 32) * VR), 0);
    mv_out(vt_c + (size_t) q * VR * LM, cod, VR * LM / DIM);
  }
}
#define VTS(b, row) vt_sr[((row) / VR) * (LM / 32) * VR + (b) * VR + (row) % VR]
// the chunk's K / V codes and scales into cache keys LMX_SK.. (K^T: [d][key]; V: [key][d], scales [key/32][d])
static void cache_append(void) {
  for (int g = 0; g < LMX_NKV; g++) {
    uint8_t *kt = KTC_ + (size_t) g * ATTN_D * ATTN_SK, *kts = KTS_ + (size_t) g * (ATTN_D / 32) * ATTN_SK;
    uint8_t *v = VC_ + (size_t) g * ATTN_SK * ATTN_D, *vs = VS_ + (size_t) g * (ATTN_SK / 32) * ATTN_D;
    for (int t = 0; t < LM; t++) {
      for (int d = 0; d < H2; d++) {
        kt[d * ATTN_SK + LMX_SK + t] = k1_c[t * KH + g * H2 + d];
        kt[(H2 + d) * ATTN_SK + LMX_SK + t] = k2_c[t * KH + g * H2 + d];
      }
      for (int b = 0; b < H2 / 32; b++) {
        kts[b * ATTN_SK + LMX_SK + t] = k1_sr[SRI(t, g * (H2 / 32) + b, KH / 32)];
        kts[(H2 / 32 + b) * ATTN_SK + LMX_SK + t] = k2_sr[SRI(t, g * (H2 / 32) + b, KH / 32)];
      }
      for (int d = 0; d < LHD; d++) v[(LMX_SK + t) * ATTN_D + d] = vt_c[(g * LHD + d) * LM + t];
    }
    for (int b = 0; b < LM / 32; b++)
      for (int d = 0; d < LHD; d++) vs[(LMX_SK / 32 + b) * ATTN_D + d] = VTS(b, g * LHD + d);
  }
}

// ---- attention hooks: Q of head cur_head from the RoPE output (x1 half from q1, x2 from q2), tiled for the mesh; O ->
// Q(.), tiled, stored into o_c [LM/PQ][LD] at the head's columns, its scales straight into o_s rows ----
static void lmx_load_q(void) {
  int w = MAX_BYTES / DIM;
  while ((H2 / DIM) % w) w >>= 1;
  gemmini_config_ld(QH);
  for (int r = 0; r < ATTN_SQ / DIM; r++)
    for (int x = 0; x < 2; x++)
      for (int c = 0; c < H2 / DIM; c += w)
        gemmini_extended_mvin((x ? q2_c : q1_c) + (size_t) r * DIM * QH + cur_head * H2 + c * DIM,
                              SP_Q + (r * (LHD / DIM) + x * (H2 / DIM) + c) * DIM, DIM * w, DIM);
}
typedef char lmx_o_fits[(ATTN_SQ * ATTN_D / (PQ * DIM) <= ROWS8(ATTN_SQ, BK) && ATTN_SQ % (PQ * DIM) == 0) ? 1 : -1];
static void lmx_store_o(void) {
  const uint32_t dst = P_BUF(NB & 1);   // the P buffer the last block does not use
  SR_FMT(dst, SP_O, ATTN_SQ, ATTN_D, 1, o_s + (size_t) cur_head * (LHD / 32) * LM);
  gemmini_config_st(LD);
  for (int pi = 0; pi < ATTN_SQ / (PQ * DIM); pi++)
    for (int ni = 0; ni < LHD / DIM; ni++)
      gemmini_extended_mvout(o_c + (size_t) pi * DIM * LD + cur_head * LHD + ni * DIM, dst + (pi * (LHD / DIM) + ni) * DIM,
                             DIM, DIM);
}
static void q_scales_of_head(int h) {   // [hd/32][LM]: groups 0..H2/32-1 from x1, the rest from x2
  for (int j = 0; j < ATTN_SQ; j++)
    for (int g = 0; g < H2 / 32; g++) {
      q_sc_pass[g][j] = q1_sr[SRI(j, h * (H2 / 32) + g, QH / 32)];
      q_sc_pass[H2 / 32 + g][j] = q2_sr[SRI(j, h * (H2 / 32) + g, QH / 32)];
    }
}

// ---- SwiGLU -> Q(.): 8 tokens, column chunks of FC alternating between streams s = 0, 1 (bank 2s: G and the codes;
// bank 2s+1: U);  U = U * G; G = 1 / (1 + exp(-G)); U = U * G  (= silu(g) * u) ----
static int swiglu_phase(void) {
  int fc = (BANK_B / (16 + 8 / PQ)) / 128 * 128;
  if (fc > LF) fc = LF;
  if (fc > 8192) fc = 8192;
  if (fc / VL > BANK_ROWS / 8) fc = BANK_ROWS / 8 * VL / 128 * 128;
  if (fc < 128) { printf("swiglu_phase: no chunk fits\n"); return 1; }
  const int nchk = (LF + fc - 1) / fc;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  for (int tg = 0; tg < LM / 8; tg++) {
    const int t0 = tg * 8;
    for (int k = 0; k < nchk; k += 2) {
      const int ns = MIN(2, nchk - k);
      int c0[2], wd[2], fr[2];
      uint32_t SG[2], SC[2], SU[2];
      for (int s = 0; s < ns; s++) {
        c0[s] = (k + s) * fc; wd[s] = MIN(fc, LF - c0[s]); fr[s] = wd[s] / VL;
        SG[s] = SPR(2 * s, 0); SC[s] = SPR(2 * s, 8 * fr[s]); SU[s] = SPR(2 * s + 1, 0);
        for (int m = 0; m < 8; m++) {
          mv_in(Gb + (size_t) (t0 + m) * LF + c0[s], SG[s] + m * fr[s], fr[s]);
          mv_in(Ub + (size_t) (t0 + m) * LF + c0[s], SU[s] + m * fr[s], fr[s]);
        }
      }
      for (int s = 0; s < ns; s++) gemmini_vpu_binary(VPU_MUL, SU[s], SU[s], SG[s], 8 * fr[s]);
      for (int s = 0; s < ns; s++) gemmini_vpu_scalar(VPU_MULS, SG[s], SG[s], 0xBF80, 8 * fr[s]);   // -1
      for (int s = 0; s < ns; s++) gemmini_vpu_unary(VPU_EXP, SG[s], SG[s], 8 * fr[s]);
      for (int s = 0; s < ns; s++) gemmini_vpu_scalar(VPU_ADDS, SG[s], SG[s], 0x3F80, 8 * fr[s]);   // +1
      for (int s = 0; s < ns; s++) gemmini_vpu_unary(VPU_RCP, SG[s], SG[s], 8 * fr[s]);
      for (int s = 0; s < ns; s++) gemmini_vpu_binary(VPU_MUL, SU[s], SU[s], SG[s], 8 * fr[s]);
      for (int s = 0; s < ns; s++) SR_FMT(SC[s], SU[s], 8, wd[s], 0, h_sr + t0 * (LF / 32) + (c0[s] / 32) * 8);
      for (int s = 0; s < ns; s++)
        for (int r = 0; r < 8 / PQ; r++)
          mv_out(h_c + (size_t) (t0 / PQ + r) * LF + c0[s], SC[s] + r * (wd[s] / DIM), wd[s] / DIM);
    }
  }
  return 0;
}

// ---- out = a + b: 8-token streams in bank pairs, column chunks ----
static void residual_phase(const uint16_t *a, const uint16_t *b, uint16_t *out) {
  int rc = (BANK_B / 16) / 128 * 128;
  if (rc > LD) rc = LD;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  for (int c0 = 0; c0 < LD; c0 += rc) {
    const int wd = MIN(rc, LD - c0), r = wd / VL;
    for (int tg = 0; tg < LM / 16; tg++) {
      for (int s = 0; s < 2; s++)
        for (int m = 0; m < 8; m++) {
          const size_t o = (size_t) (tg * 16 + 8 * s + m) * LD + c0;
          mv_in(a + o, SPR(2 * s, m * r), r);
          mv_in(b + o, SPR(2 * s + 1, m * r), r);
        }
      for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_ADD, SPR(2 * s, 0), SPR(2 * s, 0), SPR(2 * s + 1, 0), 8 * r);
      for (int s = 0; s < 2; s++)
        for (int m = 0; m < 8; m++) mv_out(out + (size_t) (tg * 16 + 8 * s + m) * LD + c0, SPR(2 * s, m * r), r);
    }
  }
}

// ---- checks ----
static float rel_bf16(const uint16_t *x, const float *ref, const uint16_t *base, int n) {   // ||x - b - ref|| / ||ref||
  float num = 0, den = 0;
  for (int i = 0; i < n; i++) {
    const float r = ref[i] - (base ? bf(base[i]) : 0.0f), v = bf(x[i]) - (base ? bf(base[i]) : 0.0f);
    num += (v - r) * (v - r); den += r * r;
  }
  return sqrtf(num / den);
}

enum { PH_RMS1, PH_QKV, PH_ROPE, PH_ATTN, PH_OPROJ, PH_RMS2, PH_GU, PH_SWIGLU, PH_DOWN, PH_RES2, PH_N };
static const char *ph_name[PH_N] = {"rmsnorm1 -> Q", "Q/K/V proj", "bias + rope + K/V cache", "attention (E4M3)",
                                    "o_proj", "residual1 + rmsnorm2", "gate/up proj", "swiglu -> Q", "down proj",
                                    "residual2"};
#define IDEAL(m, k, n) ((uint64_t) (m) * (k) * (n) / ((uint64_t) DIM * DIM * RATE))
#define PASS_IDEAL (2ULL * ATTN_SQ * ATTN_SK * ATTN_D / (DIM * DIM))
static const uint64_t ph_ideal[PH_N] = {0, IDEAL(LM, LD, LD + 2 * LKV), 0, LNH * PASS_IDEAL, IDEAL(LM, LD, LD),
                                        0, 2 * IDEAL(LM, LD, LF), 0, IDEAL(LM, LF, LD), 0};

int main() {
  lmx_rw = (uint8_t *) LMX_AT(0, uint8_t);
  __asm__ volatile("" : "+r"(lmx_rw));
  const uint16_t *H_PRE = LMX_AT(LMX_OFF_H_PRE, uint16_t);
  printf("llama_layer_mx: %s layer, %s projections (attention E4M3), DIM %d: %d tokens after a %d-token cache, D=%d "
         "F=%d, %d heads / %d kv heads, head dim %d%s\n", LMX_MODEL_NAME, LMX_FMT_NAME, DIM, LM, LMX_SK, LD, LF, LNH,
         LMX_NKV, LHD, LMX_BIAS ? ", Q/K/V bias" : "");
  printf("plan: RMSNorm chunk %d (%d partial sums), V^T requant %d rows, key blocks %d + %d x %d + %d\n", NC, LD / NC, VR,
         BK_FIRST, BK, NB - 2, BK_LAST);
  attn_print_config();
  gemmini_flush(0);
  uint64_t ph[PH_N], tm;
  int bad = 0;
  gemmini_fence();
  const uint64_t t_start = read_cycles();
  tm = t_start;
#define PHASE_END(p) do { gemmini_fence(); const uint64_t t_ = read_cycles(); ph[p] = t_ - tm; tm = t_; } while (0)

  OUT_TABLE(XN1);
  rms_phase(H_PRE, 0, 0, LMX_AT(LMX_OFF_W_IN_LN, uint16_t), xn1_c, xn1_sr);
  PHASE_END(PH_RMS1);
  scales_t(xn1_s, xn1_sr, LM, LD / 32);
  fmt_cfg();
  bad |= MATMUL(xn1_c, WC(Q), Qb, LD, xn1_s, WS(Q), WLUT(Q), ALUT(XN1), LM, LD, LD);
  bad |= MATMUL(xn1_c, WC(K), Kb, LKV, xn1_s, WS(K), WLUT(K), ALUT(XN1), LM, LD, LKV);
  bad |= MATMUL(xn1_c, WC(V), Vb, LKV, xn1_s, WS(V), WLUT(V), ALUT(XN1), LM, LD, LKV);
  PHASE_END(PH_QKV);

#if LMX_FMT == 1
  gemmini_mx_lut_disable();   // E4M3 from here to o_proj: the operand LUT flags off (the tables stay)
#endif
#if LMX_BIAS
  const uint16_t *bq = LMX_AT(LMX_OFF_BQ, uint16_t), *bk = LMX_AT(LMX_OFF_BK, uint16_t);
  vbias_phase();
#else
  const uint16_t *bq = 0, *bk = 0;
#endif
  bad |= rope_phase(Qb, LD, QH, LMX_AT(LMX_OFF_ROPE_CQ, uint16_t), LMX_AT(LMX_OFF_ROPE_SQ, uint16_t), bq,
                    q1_c, q2_c, q1_sr, q2_sr);
  bad |= rope_phase(Kb, LKV, KH, LMX_AT(LMX_OFF_ROPE_CK, uint16_t), LMX_AT(LMX_OFF_ROPE_SK, uint16_t), bk,
                    k1_c, k2_c, k1_sr, k2_sr);
  gemmini_fence();   // the RoPE banks are reused by the V requant; V + bv must be in DRAM for the transpose
  vt_transpose();
  vt_requant();
  gemmini_fence();
  cache_append();
  PHASE_END(PH_ROPE);

  scale_halves_reset();
  OUT_TABLE(O);
#if LMX_FMT == 1
  gemmini_mx_lut_disable();
#endif
  for (int h = 0; h < LNH; h++) {
    cur_head = h; cur_kv = h / (LNH / LMX_NKV);
    q_scales_of_head(h);
    attn_load_q_scales();
    fmask = 0;
    run_pipelined();
  }
  PHASE_END(PH_ATTN);

  scale_halves_reset();
  fmt_cfg();
  bad |= MATMUL(o_c, WC(O), Ya, LD, o_s, WS(O), WLUT(O), ALUT(O), LM, LD, LD);
  PHASE_END(PH_OPROJ);

  OUT_TABLE(XN2);
  rms_phase(H_PRE, Ya, Hmid, LMX_AT(LMX_OFF_W_POST_LN, uint16_t), xn2_c, xn2_sr);
  PHASE_END(PH_RMS2);
  scales_t(xn2_s, xn2_sr, LM, LD / 32);
  fmt_cfg();
  bad |= MATMUL(xn2_c, WC(G), Gb, LF, xn2_s, WS(G), WLUT(G), ALUT(XN2), LM, LD, LF);
  bad |= MATMUL(xn2_c, WC(U), Ub, LF, xn2_s, WS(U), WLUT(U), ALUT(XN2), LM, LD, LF);
  PHASE_END(PH_GU);

  OUT_TABLE(H);
  bad |= swiglu_phase();
  PHASE_END(PH_SWIGLU);
  scales_t(h_s, h_sr, LM, LF / 32);
  fmt_cfg();
  bad |= MATMUL(h_c, WC(D), Ym, LD, h_s, WS(D), WLUT(D), ALUT(H), LM, LF, LD);
  PHASE_END(PH_DOWN);

  residual_phase(Hmid, Ym, Hout);
  PHASE_END(PH_RES2);
  const uint64_t total = read_cycles() - t_start;
  if (bad) printf("a phase was rejected (shape / alignment / scratchpad fit)\n");

  // ---- hashes vs the rtl_exact golden ----
  const struct { const void *p; size_t sz; } hs[LMX_NHASH] = {
    {xn1_c, sizeof(xn1_c)}, {xn1_s, sizeof(xn1_s)}, {Qb, sizeof(Qb)}, {Kb, sizeof(Kb)}, {Vb, sizeof(Vb)},
    {q1_c, sizeof(q1_c)}, {q2_c, sizeof(q2_c)}, {k1_c, sizeof(k1_c)}, {k2_c, sizeof(k2_c)}, {vt_c, sizeof(vt_c)},
    {o_c, sizeof(o_c)}, {o_s, sizeof(o_s)}, {Ya, sizeof(Ya)}, {Hmid, sizeof(Hmid)}, {xn2_c, sizeof(xn2_c)},
    {xn2_s, sizeof(xn2_s)}, {Gb, sizeof(Gb)}, {Ub, sizeof(Ub)}, {h_c, sizeof(h_c)}, {h_s, sizeof(h_s)},
    {Ym, sizeof(Ym)}, {Hout, sizeof(Hout)}};
  int hmis = 0;
  for (int i = 0; i < LMX_NHASH; i++) {
    const uint64_t hv = fnv(hs[i].p, hs[i].sz);
    hmis += hv != LMX_EXPECT[i];
    printf("hash %-6s %016llx %s\n", LMX_HASH_NAME[i], (unsigned long long) hv,
           hv == LMX_EXPECT[i] ? "match" : "MISMATCH vs rtl_exact golden");
  }
  // ---- accuracy vs the fp64 layer (same BF16 input, unquantized weights) ----
  const int n = LM * LD;
  printf("accuracy  Yattn  rel_fro %6d ppm vs fp64\n", (int) (1e6f * rel_bf16(Ya, LMX_AT(LMX_OFF_REF_YATTN, float), 0, n)));
  printf("accuracy  h_mid  rel_fro %6d ppm vs fp64 (update h_mid - h_pre: %d ppm)\n",
         (int) (1e6f * rel_bf16(Hmid, LMX_AT(LMX_OFF_REF_HMID, float), 0, n)),
         (int) (1e6f * rel_bf16(Hmid, LMX_AT(LMX_OFF_REF_HMID, float), H_PRE, n)));
  printf("accuracy  Ymlp   rel_fro %6d ppm vs fp64\n", (int) (1e6f * rel_bf16(Ym, LMX_AT(LMX_OFF_REF_YMLP, float), 0, n)));
  printf("accuracy  h_out  rel_fro %6d ppm vs fp64 (layer update h_out - h_pre: %d ppm)\n",
         (int) (1e6f * rel_bf16(Hout, LMX_AT(LMX_OFF_REF_HOUT, float), 0, n)),
         (int) (1e6f * rel_bf16(Hout, LMX_AT(LMX_OFF_REF_HOUT, float), H_PRE, n)));

  uint64_t ideal = 0;
  for (int p = 0; p < PH_N; p++) {
    ideal += ph_ideal[p];
    if (ph_ideal[p])
      printf("PERF phase %-24s %11llu cycles (%2llu%%); mesh ideal %11llu -> util %llu%%\n", ph_name[p],
             (unsigned long long) ph[p], (unsigned long long) (ph[p] * 100 / total), (unsigned long long) ph_ideal[p],
             (unsigned long long) (ph_ideal[p] * 100 / ph[p]));
    else
      printf("PERF phase %-24s %11llu cycles (%2llu%%)\n", ph_name[p], (unsigned long long) ph[p],
             (unsigned long long) (ph[p] * 100 / total));
  }
  printf("PERF layer: %llu cycles from launch to h_out; mesh ideal %llu -> util %llu.%llu%%\n",
         (unsigned long long) total, (unsigned long long) ideal, (unsigned long long) (ideal * 100 / total),
         (unsigned long long) (ideal * 1000 / total % 10));
  const int fail = bad || hmis;
  printf("llama_layer_mx %s %s DIM %d %s (%d/%d stages match the rtl_exact golden)\n", LMX_MODEL_NAME, LMX_FMT_NAME, DIM,
         fail ? "FAILED" : "PASSED", LMX_NHASH - hmis, LMX_NHASH);
  return fail;
}
#endif
