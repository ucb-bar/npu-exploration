// The MoE block of one gpt-oss-20b decoder layer on MxGemmini, for T tokens (a decode step, or the verify pass of
// speculative decoding: T - 1 drafted tokens + 1): every stage on the device except the top-4 pick (CPU, 32 compares
// per token). Experts in E4M3 or E2M1 (MOE_FMT, the checkpoint's native MXFP4), the router always E4M3.
// (gen/gen_moe_mx.py: synthetic weights at gpt-oss-20b's shapes, a router solved for a target routing.)
//
//   VPU+SR  xr = E4M3(rmsnorm(h) * w);  xe = Q(same)              16 tokens at a time, column chunks of MOE_NC
//   mesh    logits = xr @ Wr;  VPU  + br                           [MT][64], columns 32.. padding
//   CPU     top-4 per token (descending, ties to the lower expert)
//   VPU     p = softmax over the 4 picked logits                    -inf padded to MOE_SW lanes per token
//   for each expert any token picked, in ascending order:
//     mesh    G, U = xe @ Wg_e / Wu_e                               all MT token rows (weight 0 if not routed here)
//     VPU+SR  H = Q((clamp(u + bu, -7, 7) + 1) * (g' = min(g + bg, 7)) * sigmoid(1.702 g'))
//     mesh    Y = H @ Wd_e
//     VPU     moe += (Y + bd) * p_e                                 p_e = the token's weight for e, else 0
//   VPU     h_out = h + moe
// Q(.) = the experts' SPAD_REQUANT (E4M3, or FP4 E2M1 in row pairs). Tokens are padded to MT rows (MOE_MT); every
// touched expert's matmuls run the first MXR (T rounded up to the mesh's row tile, PQ * DIM) -- for T <= PQ * DIM the
// same mesh work as gathering its tokens into a padded group; the rows past MXR stay 0.
//
// Timed from the first command to h_out in DRAM; phases fenced, expert phases summed over the experts. 13 stage hashes
// against the rtl_exact golden (gen/golden_moe_mx.py, MOE_EXPECT; the expert stages chained over the experts in order).
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"
#include MOE_IN_H
#include MOE_EXPERTS_H
#include MOE_EXPECT_H

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: needs the VPU config (MX_ROCKET) or Spike\n"); return 0; }
#else
// ATTN_PARAMS_DIM: DIM / BANK_ROWS / ACC_ROWS from the mesh's params header (-iquote dim<N>/<cfg>), as attn_flash.c;
// otherwise the 16x16 mesh against the shared header
#ifndef ATTN_PARAMS_DIM
#define DIM 16
#undef BANK_ROWS
#define BANK_ROWS 4096
#endif
#include "mx_native.h"
typedef char moe_expect_dim_ok[(MOE_EXPECT_DIM == DIM) ? 1 : -1];   // the golden was run for this mesh

#define VL  (DIM / 2)                  // BF16 lanes per spad row
#define LD  MOE_D
#define LF  MOE_F
#define PQ  (MOE_FMT ? 2 : 1)          // expert operand rows per byte (FP4: row pairs)
#define RATE (PQ * PQ)                 // expert MACs per mesh cycle relative to E4M3
#define TB  16                         // tokens per VPU block (SPAD_REQUANT: TB * N / 32 % 32 == 0 at N = 960)
#define RQ  (TB > PQ * DIM ? TB : PQ * DIM)
#define MT  ((MOE_T + RQ - 1) / RQ * RQ)   // token rows of the VPU phases
#define MXR ((MOE_T + PQ * DIM - 1) / (PQ * DIM) * (PQ * DIM))   // token rows of the expert matmuls (the mesh's row tile)
#ifdef MOE_MRR   // diagnostic: the router's row count (its extra rows are zero tokens, so the hashes do not change)
#define MRR MOE_MRR
#else
#define MRR ((MOE_T + DIM - 1) / DIM * DIM)                       // ... of the E4M3 router
#endif
#define SW  MOE_SW                     // softmax lanes per token
#define SWR (SW / VL)
#define SPR(b, r) ((uint32_t) ((b) * BANK_ROWS + (r)))
#define BANK_B (BANK_ROWS * DIM)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define A64 __attribute__((aligned(64)))
typedef char moe_mt_ok[(MT % TB == 0 && MXR <= MT && MRR <= MT && MT <= 64) ? 1 : -1];

// ---- DRAM buffers ----
static uint16_t Hp[MT * LD] A64;                                             // h, pad tokens 0
static uint8_t  xr_c[MT * LD] A64, xr_sr[MT * LD / 32] A64, xr_s[LD / 32 * MT] A64;
#if MOE_FMT
static uint8_t  xe_c[MT / PQ * LD] A64, xe_sr[MT * LD / 32] A64, xe_s[LD / 32 * MT] A64;
#else
#define xe_c  xr_c
#define xe_sr xr_sr
#define xe_s  xr_s
#endif
static uint16_t Lg[MT * MOE_RN] A64;                                         // router logits (+ bias)
static uint16_t Sm[MT * SW] A64, Pw[MT * SW] A64;                            // softmax in / out
static uint16_t Gb[MT * LF] A64, Ub[MT * LF] A64, Yb[MT * LD] A64;
static uint8_t  h_c[MT / PQ * LF] A64, h_sr[MT * LF / 32] A64, h_s[LF / 32 * MT] A64;
static uint16_t Acc[MT * LD] A64, Hout[MT * LD] A64;
static uint16_t wrow[MT * VL] A64;                                           // this expert's weight per token row
static uint16_t cneg7[TB * VL] A64;                                          // -7 rows (the clamps)
static uint8_t  pick[MT][MOE_K];                                             // the top-4 experts per token
static uint16_t pick_w[MT][MOE_K];                                           // their softmax weights (BF16)
static uint8_t  touched[MOE_E];
static int      ntouched;
static uint32_t scale_sink[512] A64;

int *__errno(void) { static int e; return &e; }
static inline float bf(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint64_t fnv_cont(uint64_t h, const void *p, size_t n) {
  const uint64_t *w = (const uint64_t *) p;
  for (size_t i = 0; i < n / 8; i++) { h ^= w[i]; h *= 1099511628211ULL; }
  return h;
}
#define FNV0 1469598103934665603ULL
static uint64_t fnv(const void *p, size_t n) { return fnv_cont(FNV0, p, n); }

static void mv_in(const void *src, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvin((const uint8_t *) src + (size_t) r * DIM, sp + r, DIM, MIN(DIM, rows - r));
}
static void mv_out(void *dst, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvout((uint8_t *) dst + (size_t) r * DIM, sp + r, DIM, MIN(DIM, rows - r));
}
// SR scales of TB-row commands ([groups][TB] per command, at src + t * groups + (column / 32) * TB) -> [groups][rows]
static void scales_t(uint8_t *dst, const uint8_t *src, int rows, int groups) {
  for (int t = 0; t < rows; t += TB)
    for (int g = 0; g < groups; g++)
      for (int r = 0; r < TB; r++) dst[g * rows + t + r] = src[t * groups + g * TB + r];
}

// ---- matmuls: mx_native.h's native loops with POWER-OF-TWO tiles dividing gpt-oss's 2880 (so NC <= 64, Kt <= 64):
// the N-chunk NC (a multiple of PQ * DIM) and K-tile Kt (a multiple of 32 and DIM) maximizing the bytes per loop, with
// C in one accumulator half, A + B in one scratchpad half and the loop-managed scale slices in 4 KB. The K-tiling does
// not change the numbers (the accumulator carries across K-tiles). Spike takes any divisor, the RTL does not: on FireSim
// (2026-11-02) NC 720 / Kt 160 hung at every mesh, and divisors <= 512 (NC 240-360, Kt 320-480) hung on 8x8 and gave
// wrong results on 16x16 fp8 / 8x8 fp4; power-of-two tiles passed bit-exact on all five meshes with an E4M3 path. ----
static int moe_tiles(int M, int K, int N, int q, int *pnc, int *pkt) {
  long best = 0;
  for (int nc = q * DIM; nc <= N && nc <= 512; nc += q * DIM) {
    if (N % nc || (nc & (nc - 1)) || M * nc / (DIM * 4) > ACC_ROWS / 2) continue;
    for (int kt = 32; kt <= K && kt <= 512; kt += 32) {
      if (K % kt || (kt & (kt - 1)) || kt % DIM || M * kt / (q * DIM) + kt * nc / (q * DIM) > MXN_HALF_ROWS ||
          (kt / 32) * nc > 4096 || (kt / 32) * M > 4096) continue;
      if ((long) nc * kt > best || ((long) nc * kt == best && nc > *pnc)) { best = (long) nc * kt; *pnc = nc; *pkt = kt; }
    }
  }
  return best ? 0 : 1;
}
// C [M][N] BF16 (pitch ldc) = A @ B: fmt 0 E4M3 (A [M][K], B [K][N]), 2 E2M1 (A [M/2][K] row pairs, B [K][N/2] nibbles);
// A_sc [K/32][lds] (the first M of each row), B_sc [K/32][N]
static int moe_matmul(int fmt, const uint8_t *A, const uint8_t *B, uint16_t *C, const uint8_t *A_sc, int lds,
                      const uint8_t *B_sc, int M, int K, int N) {
  const int q = fmt ? 2 : 1;
  int NC = 0, Kt = 0;
  if (moe_tiles(M, K, N, q, &NC, &Kt) || M % (q * DIM)) {
    printf("moe_matmul: no tiling for M=%d K=%d N=%d fmt %d\n", M, K, N, fmt);
    return 1;
  }
  const int lda = K, ldb = N / q, ldc = N;
  if (fmt) { gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 2, 3, 0); }
  else     { gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0); }
  gemmini_extended3_config_ld(lda, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(ldb, MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(q * ldc * sizeof(uint16_t));
  const int kts = K / Kt;
#ifdef MOE_TRACE
  printf("[trace] @%lu matmul fmt %d M %d K %d N %d: NC %d Kt %d, %d loops\n", (unsigned long) read_cycles(), fmt, M, K, N,
         NC, Kt, N / NC * kts);
#endif
  for (int c = 0; c < N / NC; c++)
    for (int t = 0; t < kts; t++) {
      const int h = 1 + (int) (mxn_loops++ & 1);
      gemmini_loop_ws_mx(M / (q * DIM), NC / (q * DIM), Kt / DIM, A + (size_t) t * Kt,
                         B + (size_t) t * Kt * ldb + (size_t) c * NC / q, t == kts - 1 ? C + (size_t) c * NC : NULL,
                         lda, ldb, ldc, A_sc + (size_t) (t * Kt / 32) * lds, B_sc + (size_t) (t * Kt / 32) * N + (size_t) c * NC,
                         lds, N, /*accumulate*/ t > 0, h, h);
    }
#ifdef MOE_TRACE
  gemmini_fence();
  printf("[trace] @%lu matmul done\n", (unsigned long) read_cycles());
#endif
  return 0;
}
#if MOE_FMT
#define SR_X(dst, src, m, n, sc) gemmini_spad_requant_fp4(dst, src, m, n, 0, (uint64_t) (sc), 0)
#else
#define SR_X(dst, src, m, n, sc) gemmini_spad_requant(dst, src, m, n, 0, (uint64_t) (sc), 0)
#endif

// ---- RMSNorm -> E4M3 (router) and Q(.) (experts): TB tokens, column chunks of NC; pass A sums x^2 per chunk (RSUM),
// partial sums added in BF16; pass B normalizes and requantizes. bank 0 x, bank 1 w + stats, bank 2 y, bank 3 codes ----
#define NC  MOE_NC
#define NCR (NC / VL)
typedef char moe_nc_ok[(LD % NC == 0 && NC % 64 == 0 && TB * NCR <= BANK_ROWS && NCR <= 1023 &&
                        NCR + 2 * TB <= BANK_ROWS && TB * NC / DIM + TB * NC / (PQ * DIM) <= BANK_ROWS &&
                        (TB * NC / 32) % 32 == 0 && TB * NC / 32 <= 2048) ? 1 : -1];
static void rms_phase(void) {
  const uint16_t inv_d = vpu_f_to_bf16(1.0f / LD), eps = vpu_f_to_bf16(MOE_EPS);
  const uint16_t *w = MOE_AT(MOE_OFF_W_LN, uint16_t);
  const int nch = LD / NC, n = TB * NCR;
  const uint32_t X = SPR(0, 0), W = SPR(1, 0), S = SPR(1, NCR), P = SPR(1, NCR + TB), Y = SPR(2, 0);
  const uint32_t C8 = SPR(3, 0), CX = SPR(3, TB * NC / DIM);
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  if (nch == 1) mv_in(w, W, NCR);
  for (int t0 = 0; t0 < MT; t0 += TB) {
    for (int k = 0; k < nch; k++) {   // pass A
      for (int m = 0; m < TB; m++) mv_in(Hp + (size_t) (t0 + m) * LD + k * NC, X + m * NCR, NCR);
      gemmini_vpu_binary(VPU_MUL, Y, X, X, n);
      gemmini_vpu_reduce(VPU_RSUM, k ? P : S, Y, n, NCR);
      if (k) gemmini_vpu_binary(VPU_ADD, S, S, P, TB);
    }
    gemmini_vpu_scalar(VPU_MULS, S, S, inv_d, TB);
    gemmini_vpu_scalar(VPU_ADDS, S, S, eps, TB);
    gemmini_vpu_unary(VPU_RSQRT, S, S, TB);
    for (int k = 0; k < nch; k++) {   // pass B
      if (nch > 1) {
        mv_in(w + k * NC, W, NCR);
        for (int m = 0; m < TB; m++) mv_in(Hp + (size_t) (t0 + m) * LD + k * NC, X + m * NCR, NCR);
      }
      gemmini_vpu_bcast(VPU_MUL, Y, X, S, n, NCR);
      for (int m = 0; m < TB; m++) gemmini_vpu_binary(VPU_MUL, Y + m * NCR, Y + m * NCR, W, NCR);
      gemmini_spad_requant(C8, Y, TB, NC, 0, (uint64_t) (xr_sr + t0 * (LD / 32) + (k * NC / 32) * TB), 0);
      for (int m = 0; m < TB; m++) mv_out(xr_c + (size_t) (t0 + m) * LD + k * NC, C8 + m * (NC / DIM), NC / DIM);
#if MOE_FMT
      SR_X(CX, Y, TB, NC, xe_sr + t0 * (LD / 32) + (k * NC / 32) * TB);
      for (int r = 0; r < TB / PQ; r++) mv_out(xe_c + (size_t) (t0 / PQ + r) * LD + k * NC, CX + r * (NC / DIM), NC / DIM);
#endif
    }
  }
}

// ---- router bias: logits [MT][RN] + br, 8 tokens at a time (bank 0 the tokens, bank 1 the bias row) ----
#define RNR (MOE_RN / VL)
static void router_bias_phase(void) {
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  mv_in(MOE_AT(MOE_OFF_BR, uint16_t), SPR(1, 0), RNR);
  for (int t0 = 0; t0 < MT; t0 += 8) {
    mv_in(Lg + (size_t) t0 * MOE_RN, SPR(0, 0), 8 * RNR);
    for (int m = 0; m < 8; m++) gemmini_vpu_binary(VPU_ADD, SPR(0, m * RNR), SPR(0, m * RNR), SPR(1, 0), RNR);
    mv_out(Lg + (size_t) t0 * MOE_RN, SPR(0, 0), 8 * RNR);
  }
}

// ---- top-4 (CPU) and the softmax over them (VPU) ----
static inline uint16_t ord16(uint16_t x) { return (x & 0x8000) ? (uint16_t) ~x : (uint16_t) (x ^ 0x8000); }
static void topk_phase(void) {
  memset(touched, 0, sizeof(touched));
  for (int t = 0; t < MT; t++) {
    uint16_t *s = Sm + (size_t) t * SW;
    if (t >= MOE_T) { for (int j = 0; j < SW; j++) s[j] = 0; continue; }
    const uint16_t *l = Lg + (size_t) t * MOE_RN;
    int sel[MOE_K];
    for (int k = 0; k < MOE_K; k++) {   // the k-th largest: strictly greater wins, so ties go to the lower expert
      int b = -1;
      for (int e = 0; e < MOE_E; e++) {
        int used = 0;
        for (int j = 0; j < k; j++) used |= sel[j] == e;
        if (!used && (b < 0 || ord16(l[e]) > ord16(l[b]))) b = e;
      }
      sel[k] = b;
      pick[t][k] = (uint8_t) b;
      touched[b] = 1;
      s[k] = l[b];
    }
    for (int j = MOE_K; j < SW; j++) s[j] = 0xFF80;   // -inf
  }
  ntouched = 0;
  for (int e = 0; e < MOE_E; e++) ntouched += touched[e];
  const uint32_t S = SPR(0, 0), Mx = SPR(1, 0), L = SPR(1, MT);
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  mv_in(Sm, S, MT * SWR);
  gemmini_vpu_reduce(VPU_RMAX, Mx, S, MT * SWR, SWR);
  gemmini_vpu_bcast(VPU_SUB, S, S, Mx, MT * SWR, SWR);
  gemmini_vpu_unary(VPU_EXP, S, S, MT * SWR);
  gemmini_vpu_reduce(VPU_RSUM, L, S, MT * SWR, SWR);
  gemmini_vpu_unary(VPU_RCP, L, L, MT);
  gemmini_vpu_bcast(VPU_MUL, S, S, L, MT * SWR, SWR);
  mv_out(Pw, S, MT * SWR);
}

// ---- the gated activation -> Q(.): TB tokens, column chunks of FC. bank 0 G, bank 1 U, bank 2 the bias chunks and
// the -7 rows, bank 3 the codes.  U = -max(-max(U, -7), -7) + 1;  U = U * G'  with G' = -max(-G, -7);
// G = rcp(exp(G' * -1.702) + 1);  U = U * G  ->  Q(U) ----
#define FC  MOE_FC
#define FCR (FC / VL)
typedef char moe_fc_ok[(LF % FC == 0 && FC % 64 == 0 && TB * FCR <= BANK_ROWS && FCR <= 1023 &&
                        2 * FCR + TB <= BANK_ROWS && TB * FC / (PQ * DIM) <= BANK_ROWS &&
                        (TB * FC / 32) % 32 == 0 && TB * FC / 32 <= 2048) ? 1 : -1];
static void act_phase(int e) {
  const uint16_t *bg = MOE_X(e, BG, uint16_t), *bu = MOE_X(e, BU, uint16_t);
  const uint16_t m1 = 0xBF80, p1 = 0x3F80, ma = vpu_f_to_bf16(-MOE_ALPHA);
  const uint32_t G = SPR(0, 0), U = SPR(1, 0), BGs = SPR(2, 0), BUs = SPR(2, FCR), C7 = SPR(2, 2 * FCR), CO = SPR(3, 0);
  const int n = TB * FCR;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  mv_in(cneg7, C7, TB);
  for (int t0 = 0; t0 < MT; t0 += TB)
    for (int c0 = 0; c0 < LF; c0 += FC) {
      mv_in(bg + c0, BGs, FCR); mv_in(bu + c0, BUs, FCR);
      for (int m = 0; m < TB; m++) {
        mv_in(Gb + (size_t) (t0 + m) * LF + c0, G + m * FCR, FCR);
        mv_in(Ub + (size_t) (t0 + m) * LF + c0, U + m * FCR, FCR);
      }
      for (int m = 0; m < TB; m++) {
        gemmini_vpu_binary(VPU_ADD, G + m * FCR, G + m * FCR, BGs, FCR);
        gemmini_vpu_binary(VPU_ADD, U + m * FCR, U + m * FCR, BUs, FCR);
      }
      gemmini_vpu_scalar(VPU_MULS, G, G, m1, n);                 // g' = -max(-g, -7)
      gemmini_vpu_bcast(VPU_MAX, G, G, C7, n, FCR);
      gemmini_vpu_scalar(VPU_MULS, G, G, m1, n);
      gemmini_vpu_bcast(VPU_MAX, U, U, C7, n, FCR);              // u' = -max(-max(u, -7), -7)
      gemmini_vpu_scalar(VPU_MULS, U, U, m1, n);
      gemmini_vpu_bcast(VPU_MAX, U, U, C7, n, FCR);
      gemmini_vpu_scalar(VPU_MULS, U, U, m1, n);
      gemmini_vpu_scalar(VPU_ADDS, U, U, p1, n);                 // (u' + 1) * g'
      gemmini_vpu_binary(VPU_MUL, U, U, G, n);
      gemmini_vpu_scalar(VPU_MULS, G, G, ma, n);                 // sigmoid(1.702 g')
      gemmini_vpu_unary(VPU_EXP, G, G, n);
      gemmini_vpu_scalar(VPU_ADDS, G, G, p1, n);
      gemmini_vpu_unary(VPU_RCP, G, G, n);
      gemmini_vpu_binary(VPU_MUL, U, U, G, n);
      SR_X(CO, U, TB, FC, h_sr + t0 * (LF / 32) + (c0 / 32) * TB);
      for (int r = 0; r < TB / PQ; r++) mv_out(h_c + (size_t) (t0 / PQ + r) * LF + c0, CO + r * (FC / DIM), FC / DIM);
    }
}

// ---- moe (+)= (Y + bd) * p_e per token row: TB tokens, column chunks of NC. bank 0 moe, bank 1 Y, bank 2 bd + the
// weight rows. The first expert writes moe = (Y + bd) * p_e ----
static void combine_phase(int e, int first) {
  const uint16_t *bd = MOE_X(e, BD, uint16_t);
  const uint32_t A = SPR(0, 0), Yr = SPR(1, 0), Bd = SPR(2, 0), Wr = SPR(2, NCR);
  const int n = TB * NCR;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  for (int t0 = 0; t0 < MT; t0 += TB) {
    mv_in(wrow + (size_t) t0 * VL, Wr, TB);
    for (int c0 = 0; c0 < LD; c0 += NC) {
      mv_in(bd + c0, Bd, NCR);
      for (int m = 0; m < TB; m++) {
        mv_in(Yb + (size_t) (t0 + m) * LD + c0, Yr + m * NCR, NCR);
        if (!first) mv_in(Acc + (size_t) (t0 + m) * LD + c0, A + m * NCR, NCR);
      }
      for (int m = 0; m < TB; m++) gemmini_vpu_binary(VPU_ADD, Yr + m * NCR, Yr + m * NCR, Bd, NCR);
      gemmini_vpu_bcast(VPU_MUL, Yr, Yr, Wr, n, NCR);
      if (!first) gemmini_vpu_binary(VPU_ADD, A, A, Yr, n);
      for (int m = 0; m < TB; m++) mv_out(Acc + (size_t) (t0 + m) * LD + c0, (first ? Yr : A) + m * NCR, NCR);
    }
  }
}

// ---- h_out = h + moe ----
static void residual_phase(void) {
  const int n = TB * NCR;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  for (int t0 = 0; t0 < MT; t0 += TB)
    for (int c0 = 0; c0 < LD; c0 += NC) {
      for (int m = 0; m < TB; m++) {
        mv_in(Hp + (size_t) (t0 + m) * LD + c0, SPR(0, m * NCR), NCR);
        mv_in(Acc + (size_t) (t0 + m) * LD + c0, SPR(1, m * NCR), NCR);
      }
      gemmini_fence();   // llama_layer_mx's residual race (LMX_RES_FENCE): mvin / VPU / mvout ordered by fences
      gemmini_vpu_binary(VPU_ADD, SPR(0, 0), SPR(0, 0), SPR(1, 0), n);
      gemmini_fence();
      for (int m = 0; m < TB; m++) mv_out(Hout + (size_t) (t0 + m) * LD + c0, SPR(0, m * NCR), NCR);
      gemmini_fence();
    }
}

static float rel_bf16(const uint16_t *x, const float *ref, const uint16_t *base, int rows, int ld) {
  float num = 0, den = 0;
  for (int t = 0; t < rows; t++)
    for (int i = 0; i < LD; i++) {
      const float b = base ? bf(base[(size_t) t * ld + i]) : 0.0f;
      const float r = ref[(size_t) t * LD + i] - b, v = bf(x[(size_t) t * ld + i]) - b;
      num += (v - r) * (v - r); den += r * r;
    }
  return sqrtf(num / den);
}

enum { PH_RMS, PH_ROUTER, PH_TOPK, PH_GU, PH_ACT, PH_DOWN, PH_COMB, PH_RES, PH_N };
static const char *ph_name[PH_N] = {"rmsnorm -> E4M3 / Q", "router (E4M3) + bias", "top-4 + softmax",
                                    "gate/up proj", "activation -> Q", "down proj", "combine", "residual"};

int main() {
  printf("moe_layer_mx: gpt-oss-20b MoE block (%s), %s experts, DIM %d: %d tokens (%d rows), D=%d F=%d, %d experts, "
         "top-%d\n", MOE_CFG_NAME, MOE_FMT_NAME, DIM, MOE_T, MT, LD, LF, MOE_E, MOE_K);
  {
    int nc, kt;
    moe_tiles(MXR, LD, LF, PQ, &nc, &kt);
    printf("plan: matmul rows %d (router %d); VPU chunks %d (RMSNorm, %d partial sums) / %d (activation); expert tiles NC "
           "%d Kt %d\n", MXR, MRR, NC, LD / NC, FC, nc, kt);
  }
#ifdef MOE_TRACE
  printf("[trace] @%lu start\n", (unsigned long) read_cycles());
#endif
  // inputs into the padded token buffer, the clamp constants (setup, untimed)
  memcpy(Hp, MOE_AT(MOE_OFF_H, uint16_t), (size_t) MOE_T * LD * sizeof(uint16_t));
  memset(Hp + (size_t) MOE_T * LD, 0, (size_t) (MT - MOE_T) * LD * sizeof(uint16_t));
  for (int i = 0; i < TB * VL; i++) cneg7[i] = 0xC0E0;   // -7
  memset(Lg, 0, sizeof(Lg)); memset(Gb, 0, sizeof(Gb)); memset(Ub, 0, sizeof(Ub)); memset(Yb, 0, sizeof(Yb));   // pad rows
  gemmini_flush(0);
  uint64_t ph[PH_N] = {0}, tm;
  int bad = 0;
  gemmini_fence();
  const uint64_t t_start = read_cycles();
  tm = t_start;
#ifdef MOE_TRACE   // diagnostic: where a hang stops (the prints land inside the timed phases)
#define PHASE_END(p) do { gemmini_fence(); printf("[trace] @%lu %s done\n", (unsigned long) read_cycles(), ph_name[p]); const uint64_t t_ = read_cycles(); \
                          ph[p] += t_ - tm; tm = t_; } while (0)
#else
#define PHASE_END(p) do { gemmini_fence(); const uint64_t t_ = read_cycles(); ph[p] += t_ - tm; tm = t_; } while (0)
#endif

  rms_phase();
  PHASE_END(PH_RMS);
  scales_t(xr_s, xr_sr, MT, LD / 32);
#if MOE_FMT
  scales_t(xe_s, xe_sr, MT, LD / 32);
#endif
  bad |= moe_matmul(0, xr_c, MOE_AT(MOE_OFF_WR_C, uint8_t), Lg, xr_s, MT, MOE_AT(MOE_OFF_WR_S, uint8_t), MRR, LD, MOE_RN);
  gemmini_fence();
  router_bias_phase();
  PHASE_END(PH_ROUTER);
  topk_phase();
  PHASE_END(PH_TOPK);
  for (int t = 0; t < MOE_T; t++)
    for (int k = 0; k < MOE_K; k++) pick_w[t][k] = Pw[(size_t) t * SW + k];

  uint64_t hg = FNV0, hu = FNV0, hh = FNV0, hhs = FNV0, hy = FNV0;
  int first = 1;
  for (int e = 0; e < MOE_E; e++) {
    if (!touched[e]) continue;
    bad |= moe_matmul(MOE_FMT, xe_c, MOE_X(e, GC, uint8_t), Gb, xe_s, MT, MOE_X(e, GS, uint8_t), MXR, LD, LF);
    bad |= moe_matmul(MOE_FMT, xe_c, MOE_X(e, UC, uint8_t), Ub, xe_s, MT, MOE_X(e, US, uint8_t), MXR, LD, LF);
    PHASE_END(PH_GU);
    act_phase(e);
    PHASE_END(PH_ACT);
    scales_t(h_s, h_sr, MT, LF / 32);
    bad |= moe_matmul(MOE_FMT, h_c, MOE_X(e, DC, uint8_t), Yb, h_s, MT, MOE_X(e, DS, uint8_t), MXR, LF, LD);
    PHASE_END(PH_DOWN);
    for (int t = 0; t < MT; t++) {   // this expert's weight per token row (0 if the token did not pick it)
      uint16_t w = 0;
      if (t < MOE_T)
        for (int k = 0; k < MOE_K; k++)
          if (pick[t][k] == e) w = pick_w[t][k];
      for (int l = 0; l < VL; l++) wrow[(size_t) t * VL + l] = w;
    }
    combine_phase(e, first);
    PHASE_END(PH_COMB);
    // hashes of this expert's stages, chained over the experts (outside the timed phases' work, but inside the
    // layer time: the CPU reads ~0.5 MB here; excluded below)
    const uint64_t th = read_cycles();
    hg = fnv_cont(hg, Gb, sizeof(Gb)); hu = fnv_cont(hu, Ub, sizeof(Ub)); hh = fnv_cont(hh, h_c, sizeof(h_c));
    hhs = fnv_cont(hhs, h_s, sizeof(h_s)); hy = fnv_cont(hy, Yb, sizeof(Yb));
    tm += read_cycles() - th;
    first = 0;
  }
  residual_phase();
  PHASE_END(PH_RES);
  uint64_t total = 0;
  for (int p = 0; p < PH_N; p++) total += ph[p];
  if (bad) printf("a phase was rejected (shape / alignment / scratchpad fit)\n");

  // ---- routing ----
  int tgt_ok = 1;
  for (int t = 0; t < MOE_T; t++) {
    const uint8_t *tg = MOE_AT(MOE_OFF_TARGET, uint8_t) + t * MOE_K;
    for (int k = 0; k < MOE_K; k++) {
      int in = 0;
      for (int j = 0; j < MOE_K; j++) in |= tg[j] == pick[t][k];
      tgt_ok &= in;
    }
    if (t < 4 || t == MOE_T - 1)
      printf("route token %2d: experts %2d %2d %2d %2d  weights (1e-4) %d %d %d %d\n", t, pick[t][0], pick[t][1],
             pick[t][2], pick[t][3], (int) (1e4f * bf(pick_w[t][0])), (int) (1e4f * bf(pick_w[t][1])),
             (int) (1e4f * bf(pick_w[t][2])), (int) (1e4f * bf(pick_w[t][3])));
  }
  printf("routing: %d distinct experts over %d tokens (target %d; %s)\n", ntouched, MOE_T, MOE_TARGET_DISTINCT,
         tgt_ok ? "every pick as targeted" : "DIFFERS from the target");

  // ---- hashes vs the rtl_exact golden ----
  const uint64_t hv[MOE_NHASH] = {fnv(xr_c, sizeof(xr_c)), fnv(xr_s, sizeof(xr_s)), fnv(xe_c, sizeof(xe_c)),
                                  fnv(xe_s, sizeof(xe_s)), fnv(Lg, sizeof(Lg)), fnv(Pw, sizeof(Pw)), hg, hu, hh, hhs, hy,
                                  fnv(Acc, sizeof(Acc)), fnv(Hout, sizeof(Hout))};
  int hmis = 0;
  for (int i = 0; i < MOE_NHASH; i++) {
    hmis += hv[i] != MOE_EXPECT[i];
    printf("hash %-5s %016llx %s\n", MOE_HASH_NAME[i], (unsigned long long) hv[i],
           hv[i] == MOE_EXPECT[i] ? "match" : "MISMATCH vs rtl_exact golden");
  }
  printf("accuracy  moe    rel_fro %6d ppm vs fp64\n",
         (int) (1e6f * rel_bf16(Acc, MOE_AT(MOE_OFF_REF_MOE, float), 0, MOE_T, LD)));
  printf("accuracy  h_out  rel_fro %6d ppm vs fp64 (update h_out - h: %d ppm)\n",
         (int) (1e6f * rel_bf16(Hout, MOE_AT(MOE_OFF_REF_HOUT, float), 0, MOE_T, LD)),
         (int) (1e6f * rel_bf16(Hout, MOE_AT(MOE_OFF_REF_HOUT, float), Hp, MOE_T, LD)));

  // ---- performance ----
  const uint64_t xbytes = (uint64_t) 3 * LD * LF / PQ + (uint64_t) 3 * LD * LF / 32;   // one expert's codes + scales
  const uint64_t ideal_r = (uint64_t) MRR * LD * MOE_RN / (DIM * DIM);
  const uint64_t ideal_x = (uint64_t) ntouched * 3 * MXR * LD * LF / ((uint64_t) DIM * DIM * RATE);
  for (int p = 0; p < PH_N; p++)
    printf("PERF phase %-22s %11llu cycles (%2llu%%)\n", ph_name[p], (unsigned long long) ph[p],
           (unsigned long long) (ph[p] * 100 / total));
  const uint64_t mm = ph[PH_GU] + ph[PH_DOWN];
  printf("PERF experts: %d touched, %llu weight bytes (%llu per expert); expert matmuls %llu cycles -> %llu.%02llu B/cycle"
         ", mesh ideal %llu -> util %llu%%\n", ntouched, (unsigned long long) (ntouched * xbytes),
         (unsigned long long) xbytes, (unsigned long long) mm, (unsigned long long) (ntouched * xbytes / mm),
         (unsigned long long) (ntouched * xbytes * 100 / mm % 100), (unsigned long long) ideal_x,
         (unsigned long long) (ideal_x * 100 / mm));
  printf("PERF layer: %llu cycles from launch to h_out; mesh ideal %llu -> util %llu.%llu%%; %llu cycles per token\n",
         (unsigned long long) total, (unsigned long long) (ideal_r + ideal_x),
         (unsigned long long) ((ideal_r + ideal_x) * 100 / total), (unsigned long long) ((ideal_r + ideal_x) * 1000 / total % 10),
         (unsigned long long) (total / MOE_T));
  const int fail = bad || hmis;
  printf("moe_layer_mx %s %s DIM %d %s (%d/%d stages match the rtl_exact golden)\n", MOE_CFG_NAME, MOE_FMT_NAME, DIM,
         fail ? "FAILED" : "PASSED", MOE_NHASH - hmis, MOE_NHASH);
  return fail;
}
#endif
