// TinyLlama layer 5's MLP block on MxGemmini, every stage on the device, on the layer's input h_pre (64 tokens):
//
//   VPU+SR  xn = MX(rmsnorm(h_pre) * w_post)        16-token chunks, two 8-token streams (two VPUs)
//   mesh    G, U = xn @ Wg / Wu                      native DRAM loops (2048 -> 5632)
//   VPU+SR  h = MX(silu(G) * U)                      8-token chunks, two column-half streams
//   mesh    y = h @ Wd                               (5632 -> 2048)
//   VPU     h_out = h_pre + y
//
// MLP_FP4 0: E4M3 (llama_layer_e2e.bin's weights). 1: FP4 x FP4 projections on the quad mesh (4x the E4M3 rate), FP4
// SPAD_REQUANT of xn and h (codes two rows per byte), weights from llama_mlp_fp4.bin. Phases fenced; utilization = the
// MLP's mesh-ideal cycles at the format's rate / the measured cycles. Hashes vs the golden (gen/gen_llama_mlp_fp4.py).
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"
#include "llama_layer_e2e.h"
#ifndef MLP_FP4
#define MLP_FP4 0
#endif
#if MLP_FP4
#include "llama_mlp_fp4.h"
#include "llama_mlp_e2e_fp4_expect.h"
#else
#include "llama_mlp_e2e_expect.h"
#endif

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: needs the VPU config (MX_ROCKET) or Spike\n"); return 0; }
#else

#define DIM 16
#undef BANK_ROWS
#define BANK_ROWS 4096
#include "mx_native.h"

#define LM E2E_M
#define LD E2E_D
#define LF E2E_F
#define FH (LF / 2)
#define QT (MLP_FP4 ? 2 : 1)   // activation codes per byte (FP4: two token rows per byte)
#define SPR(b, r) ((uint32_t) ((b) * BANK_ROWS + (r)))
#define A64 __attribute__((aligned(64)))
static uint8_t  xn_c[LM * LD / QT] A64, xn_sr[LM * LD / 32] A64, xn_s[LD / 32 * LM] A64;
static uint16_t Gb[LM * LF] A64, Ub[LM * LF] A64;
static uint8_t  h_c[LM * LF / QT] A64, h_sr[2][LM * FH / 32] A64, h_s[LF / 32 * LM] A64;
static uint16_t Ym[LM * LD] A64, Hout[LM * LD] A64;

#if MLP_FP4
#define H_PRE_   MLP4_AT(MLP4_OFF_H_PRE, uint16_t)
#define W_POST_  MLP4_AT(MLP4_OFF_W_POST_LN, uint16_t)
#define WG_      MLP4_AT(MLP4_OFF_WG4_CODES, uint8_t)
#define WG_S_    MLP4_AT(MLP4_OFF_WG_SCALES, uint8_t)
#define WU_      MLP4_AT(MLP4_OFF_WU4_CODES, uint8_t)
#define WU_S_    MLP4_AT(MLP4_OFF_WU_SCALES, uint8_t)
#define WD_      MLP4_AT(MLP4_OFF_WD4_CODES, uint8_t)
#define WD_S_    MLP4_AT(MLP4_OFF_WD_SCALES, uint8_t)
#define MATMUL(A, lda, B, ldb, C, ldc, As, Bs, sc_ldb, M, K, N) \
  mxn_matmul_fp4(A, lda, B, (ldb) / 2, C, ldc, As, M, Bs, sc_ldb, M, K, N)
#define SR(dst, src, m, n, sc) gemmini_spad_requant_fp4(dst, src, m, n, 0, (uint64_t) (sc), 0)
#else
#define H_PRE_   E2E_AT(E2E_OFF_H_PRE, uint16_t)
#define W_POST_  E2E_AT(E2E_OFF_W_POST_LN, uint16_t)
#define WG_      E2E_AT(E2E_OFF_WG_CODES, uint8_t)
#define WG_S_    E2E_AT(E2E_OFF_WG_SCALES, uint8_t)
#define WU_      E2E_AT(E2E_OFF_WU_CODES, uint8_t)
#define WU_S_    E2E_AT(E2E_OFF_WU_SCALES, uint8_t)
#define WD_      E2E_AT(E2E_OFF_WD_CODES, uint8_t)
#define WD_S_    E2E_AT(E2E_OFF_WD_SCALES, uint8_t)
#define MATMUL(A, lda, B, ldb, C, ldc, As, Bs, sc_ldb, M, K, N) \
  mxn_matmul(A, lda, B, ldb, C, ldc, As, M, Bs, sc_ldb, M, K, N)
#define SR(dst, src, m, n, sc) gemmini_spad_requant(dst, src, m, n, 0, (uint64_t) (sc), 0)
#endif

int *__errno(void) { static int e; return &e; }
static uint64_t fnv(const void *p, size_t n) {
  const uint64_t *w = (const uint64_t *) p; uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n / 8; i++) { h ^= w[i]; h *= 1099511628211ULL; }
  return h;
}
static void mvin16(const void *src, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvin((const uint8_t *) src + r * DIM, sp + r, DIM, DIM);
}
static void mvout16(void *dst, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvout((uint8_t *) dst + r * DIM, sp + r, DIM, DIM);
}
static void scales_t(uint8_t *dst, const uint8_t *src, int rows, int groups) {   // [rows][groups] -> [groups][rows]
  for (int g = 0; g < groups; g++)
    for (int m = 0; m < rows; m++) dst[g * rows + m] = src[m * groups + g];
}

// ---- RMSNorm -> MX: 16 tokens per chunk, stream s in banks 2s (x, weight copy) / 2s+1 (y, stats, codes) ----
#define RX(s) SPR(2 * (s), 0)
#define RW(s) SPR(2 * (s), 2048)
#define RY(s) SPR(2 * (s) + 1, 0)
#define RS(s) SPR(2 * (s) + 1, 2048)
#define RC(s) SPR(2 * (s) + 1, 2064)
static void rms_phase(void) {
  const uint16_t inv_d = vpu_f_to_bf16(1.0f / LD), eps = vpu_f_to_bf16(E2E_EPS);
  const int n = 8 * LD / 8;   // BF16 rows per stream (8 tokens)
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  for (int s = 0; s < 2; s++) mvin16(W_POST_, RW(s), LD / 8);
  for (int c = 0; c < LM / 16; c++) {
    const int t[2] = {c * 16, c * 16 + 8};
    for (int s = 0; s < 2; s++) mvin16(H_PRE_ + t[s] * LD, RX(s), n);
    for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, RY(s), RX(s), RX(s), n);
    for (int s = 0; s < 2; s++) gemmini_vpu_reduce(VPU_RSUM, RS(s), RY(s), n, LD / 8);
    for (int s = 0; s < 2; s++) gemmini_vpu_scalar(VPU_MULS, RS(s), RS(s), inv_d, 8);
    for (int s = 0; s < 2; s++) gemmini_vpu_scalar(VPU_ADDS, RS(s), RS(s), eps, 8);
    for (int s = 0; s < 2; s++) gemmini_vpu_unary(VPU_RSQRT, RS(s), RS(s), 8);
    for (int s = 0; s < 2; s++) gemmini_vpu_bcast(VPU_MUL, RY(s), RX(s), RS(s), n, LD / 8);
    for (int k = 0; k < 8; k++)
      for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, RY(s) + k * (LD / 8), RY(s) + k * (LD / 8), RW(s), LD / 8);
    for (int s = 0; s < 2; s++) SR(RC(s), RY(s), 8, LD, xn_sr + t[s] * (LD / 32));
    for (int s = 0; s < 2; s++) mvout16(xn_c + t[s] / QT * LD, RC(s), n / 2 / QT);
  }
}

// ---- SwiGLU -> MX: 8 tokens per chunk, stream s = column half s (banks 0-1 / 2-3; codes between G and U) ----
#define SG_N (8 * FH / 8)
static const uint32_t SG_G[2] = {0, 2 * BANK_ROWS}, SG_C[2] = {SG_N, 2 * BANK_ROWS + SG_N},
                      SG_U[2] = {SG_N + SG_N / 2, 2 * BANK_ROWS + SG_N + SG_N / 2};
static void swiglu_phase(void) {
  const int n = SG_N;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  for (int c = 0; c < LM / 8; c++) {
    const int t0 = c * 8;
    for (int s = 0; s < 2; s++)
      for (int m = 0; m < 8; m++) {
        mvin16(Gb + (size_t) (t0 + m) * LF + s * FH, SG_G[s] + m * (FH / 8), FH / 8);
        mvin16(Ub + (size_t) (t0 + m) * LF + s * FH, SG_U[s] + m * (FH / 8), FH / 8);
      }
    for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, SG_U[s], SG_U[s], SG_G[s], n);
    for (int s = 0; s < 2; s++) gemmini_vpu_scalar(VPU_MULS, SG_G[s], SG_G[s], 0xBF80, n);   // -1
    for (int s = 0; s < 2; s++) gemmini_vpu_unary(VPU_EXP, SG_G[s], SG_G[s], n);
    for (int s = 0; s < 2; s++) gemmini_vpu_scalar(VPU_ADDS, SG_G[s], SG_G[s], 0x3F80, n);   // +1
    for (int s = 0; s < 2; s++) gemmini_vpu_unary(VPU_RCP, SG_G[s], SG_G[s], n);
    for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_MUL, SG_U[s], SG_U[s], SG_G[s], n);
    for (int s = 0; s < 2; s++) SR(SG_C[s], SG_U[s], 8, FH, h_sr[s] + t0 * (FH / 32));
    for (int s = 0; s < 2; s++)   // code row r (a token, or a token pair for FP4) of this half: FH bytes
      for (int r = 0; r < 8 / QT; r++) mvout16(h_c + (size_t) (t0 / QT + r) * LF + s * FH, SG_C[s] + r * (FH / 16), FH / 16);
  }
}

static void residual_phase(void) {
  const int n = 8 * LD / 8;
  gemmini_config_ld(DIM); gemmini_config_st(DIM);
  for (int c = 0; c < LM / 16; c++) {
    const int t[2] = {c * 16, c * 16 + 8};
    for (int s = 0; s < 2; s++) { mvin16(H_PRE_ + t[s] * LD, RX(s), n); mvin16(Ym + t[s] * LD, RY(s), n); }
    for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_ADD, RX(s), RX(s), RY(s), n);
    for (int s = 0; s < 2; s++) mvout16(Hout + t[s] * LD, RX(s), n);
  }
}

enum { PH_RMS, PH_GU, PH_SWIGLU, PH_DOWN, PH_RES, PH_N };
static const char *ph_name[PH_N] = {"rmsnorm -> MX", "gate/up proj", "swiglu -> MX", "down proj", "residual"};
#define RATE (QT * QT)   // mesh MACs per cycle relative to E4M3
static const uint64_t ph_ideal[PH_N] = {0, 2 * MXN_IDEAL(LM, LD, LF) / RATE, 0, MXN_IDEAL(LM, LF, LD) / RATE, 0};

int main() {
  printf("llama_mlp_e2e: TinyLlama layer 5 MLP, %d tokens, D=%d F=%d, %s projections, all stages on MxGemmini\n",
         LM, LD, LF, MLP_FP4 ? "FP4 x FP4 (quad mesh)" : "E4M3");
  gemmini_flush(0);
  uint64_t ph[PH_N], tm;
  int bad = 0;
  gemmini_fence();
  const uint64_t t_start = read_cycles();
  tm = t_start;
#define PHASE_END(p) do { gemmini_fence(); const uint64_t t_ = read_cycles(); ph[p] = t_ - tm; tm = t_; } while (0)
  rms_phase();
  PHASE_END(PH_RMS);
  scales_t(xn_s, xn_sr, LM, LD / 32);
  bad |= MATMUL(xn_c, LD, WG_, LF, Gb, LF, xn_s, WG_S_, LF, LM, LD, LF);
  bad |= MATMUL(xn_c, LD, WU_, LF, Ub, LF, xn_s, WU_S_, LF, LM, LD, LF);
  PHASE_END(PH_GU);
  swiglu_phase();
  PHASE_END(PH_SWIGLU);
  for (int s = 0; s < 2; s++)
    for (int g = 0; g < FH / 32; g++)
      for (int m = 0; m < LM; m++) h_s[(s * (FH / 32) + g) * LM + m] = h_sr[s][m * (FH / 32) + g];
  bad |= MATMUL(h_c, LF, WD_, LD, Ym, LD, h_s, WD_S_, LD, LM, LF, LD);
  PHASE_END(PH_DOWN);
  residual_phase();
  PHASE_END(PH_RES);
  const uint64_t total = read_cycles() - t_start;
  if (bad) printf("a matmul was rejected (shape/alignment)\n");

  struct { const char *n; const void *p; size_t sz; } hs[] = {
    {"xn", xn_c, sizeof(xn_c)}, {"g", Gb, sizeof(Gb)}, {"u", Ub, sizeof(Ub)}, {"h", h_c, sizeof(h_c)},
    {"ymlp", Ym, sizeof(Ym)}, {"hout", Hout, sizeof(Hout)}};
  int hmis = 0;
  for (int i = 0; i < 6; i++) {
    const uint64_t h = fnv(hs[i].p, hs[i].sz);
    hmis += h != MLP_EXPECT[i];
    printf("hash %-5s %016llx %s\n", hs[i].n, (unsigned long long) h, h == MLP_EXPECT[i] ? "match" : "MISMATCH vs golden");
  }
  uint64_t ideal = 0;
  for (int p = 0; p < PH_N; p++) {
    ideal += ph_ideal[p];
    printf("PHASE %-15s %9llu cycles", ph_name[p], (unsigned long long) ph[p]);
    if (ph_ideal[p]) printf("  mesh ideal %8llu -> util %3llu%%", (unsigned long long) ph_ideal[p],
                            (unsigned long long) (100 * ph_ideal[p] / ph[p]));
    printf("\n");
  }
  printf("PERF mlp %llu cycles; mesh ideal %llu (%s rate) -> util %llu%%\n", (unsigned long long) total,
         (unsigned long long) ideal, MLP_FP4 ? "FP4 quad" : "E4M3", (unsigned long long) (100 * ideal / total));
  printf("llama_mlp_e2e %s\n", (hmis || bad) ? "FAILED" : "PASSED");
  return hmis || bad;
}
#endif
