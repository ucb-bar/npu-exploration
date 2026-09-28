// N STACKED TinyLlama DECODER LAYERS on MxGemmini -- the whole model, one ELF, real data.
//
// `llama_layer_full.c` runs ONE layer on the residual stream the capture handed it. This runs the
// stack: layer n consumes what layer n-1's DEVICE output produced, so the quantization error
// compounds through the depth exactly as it does in hardware, and -- with the head -- ends in
// logits that can be graded against TinyLlama's own and turned into a perplexity.
//
//   host   x = embed_out                                      (the model's token embeddings)
//   for n in 0..NL-1:
//     host   xn1  = rmsnorm(x, w_in_ln[n])                     -> MX
//     mesh   Q,K,V = Xn1 @ Wq/Wk/Wv[n]   (N-chunked, Xn1 resident)
//     host   RoPE per head; K transposed per kv head
//     mesh   per head  S_h = Q_h @ K_kv^T ; host causal softmax ; O_h = P_h @ V_kv -> FP8 requant
//     mesh   Yattn = sum_h O_h @ Wo_h[n] (accumulated in smem across heads)
//     host   x    = x + Yattn                                  <-- residual 1
//     host   xn2  = rmsnorm(x, w_post_ln[n])                   -> MX
//     mesh   G,U  = Xn2 @ Wg/Wu[n]       (F-chunked, Xn2 resident)
//     host   H    = silu(G) * U                                -> MX
//     mesh   Ymlp = H @ Wd[n]            (K-tiled)
//     host   x    = x + Ymlp                                   <-- residual 2
//   host   xf     = rmsnorm(x, w_final_ln)                     -> MX
//   mesh   logits = Xf @ W_lm            [M,D]x[D,V]           (V-chunked)
//   host   softmax + NLL against the true next tokens          -> perplexity
//
// WHAT IS NEW HERE IS DEPTH, and depth is not a formality. One seam cost the MLP 5.3 points
// (11.26% -> 16.60%) on a 0.32%-perturbed input; this has 2*NL of them in series, and whether that
// compounds or is absorbed by the residual stream is the question the kernel answers. Nothing about
// the per-layer numbers predicts it, which is why the goldens go all the way down: every mesh stage
// of every layer is still checked bit-exact, and each layer's output is ALSO compared against the
// model's own so a divergence names the layer it started in rather than only showing up in the
// logits.
//
// ONE UNIFORM STRIDE ADDRESSES EVERY LAYER. The blob holds NL identically-laid-out blocks, so this
// source is the same for 2 layers or 22 and the layer count lives in the data, not the C.
//
// THE SCRATCHPAD IS REPLANNED PER PHASE, not per layer: every layer uses the same schedule, so the
// compile-time plan is computed once and the weights simply move in from a different base.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "mx_host.h"

// The layer count lives in the DATA, not here -- so a 2-layer bring-up build and the full 22-layer
// one are the same source with a different generated header (and its matching blob symbol).
// `-DLLAMA_MODEL_HEADER='"llama_model_l2.h"'` selects the short one; see the Makefile.
#ifndef LLAMA_MODEL_HEADER
#define LLAMA_MODEL_HEADER "llama_model.h"
#endif
#include LLAMA_MODEL_HEADER

#define DIM 16

#ifndef LLAMA_BANK_ROWS
#define LLAMA_BANK_ROWS 4096
#endif
#if BANK_ROWS != LLAMA_BANK_ROWS
#warning "gemmini_params.h BANK_ROWS differs from this kernel's: pinning the kernel's value. \
Correct if that header is set for another bitstream; -DLLAMA_BANK_ROWS=N to retarget."
#endif
#undef BANK_ROWS
#define BANK_ROWS LLAMA_BANK_ROWS

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

#if !defined(SPIKE_SIM) && !defined(MX_ROCKET)
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#endif

#define OUT_BF16 3
#define OUT_FP8  0
#define SPAD_STORE 0x38
#define LOOP_WS_REQUANT_TILED (1u << 10)

#define SPAD_ROWS    (BANK_NUM * BANK_ROWS)
#define ROWS8(m, n)  ((m) * (n) / DIM)
#define ROWS16(m, n) ((m) * (n) * 2 / DIM)

#ifndef LLAMA_SCALE_ROWS_MAX
#define LLAMA_SCALE_ROWS_MAX 256
#endif
#define SCALE_ROWS(k, n) ((n) * (k) / 512)

// ---- attention half (identical plan to llama_layer_full.c) ----
#define A_P1_COST(nc) (ROWS8(LLAMA_M, LLAMA_D) + ROWS8(LLAMA_D, (nc)) + ROWS16(LLAMA_M, (nc)))
#define A_P1FITS(nc)  (A_P1_COST(nc) <= SPAD_ROWS && \
                       SCALE_ROWS(LLAMA_D, (nc)) <= LLAMA_SCALE_ROWS_MAX && \
                       SCALE_ROWS(LLAMA_D, LLAMA_M) <= LLAMA_SCALE_ROWS_MAX)
#define A_PROJ_N   (A_P1FITS(512) ? 512 : A_P1FITS(256) ? 256 : \
                    A_P1FITS(128) ? 128 : A_P1FITS(64) ? 64 : A_P1FITS(32) ? 32 : 16)
#define A_SPAD_XN     0
#define A_PROJ_C      ROWS8(LLAMA_M, LLAMA_D)
#define A_H2_A_Q      0
#define A_H2_B_KT     (A_H2_A_Q + ROWS8(LLAMA_M, LLAMA_H))
#define A_H2_B_KT_ARG (A_H2_B_KT + ROWS8(LLAMA_H, LLAMA_M))
#define A_H2_A_P      A_H2_B_KT_ARG
#define A_H2_B_V      (A_H2_A_P + ROWS8(LLAMA_M, LLAMA_M))
#define A_H2_B_V_ARG  (A_H2_B_V + ROWS8(LLAMA_M, LLAMA_H))
#define A_H2_C_S      A_H2_B_V_ARG
#define A_H2_C_O      (A_H2_C_S + ROWS16(LLAMA_M, LLAMA_M))
#define A_P3_COST(dc) (ROWS16(LLAMA_M, (dc)) + ROWS8(LLAMA_H, (dc)) + ROWS8(LLAMA_M, LLAMA_H))
#define A_YCHUNK   (A_P3_COST(2048) <= SPAD_ROWS ? 2048 : A_P3_COST(1024) <= SPAD_ROWS ? 1024 : \
                    A_P3_COST(512) <= SPAD_ROWS ? 512 : A_P3_COST(256) <= SPAD_ROWS ? 256 : \
                    A_P3_COST(128) <= SPAD_ROWS ? 128 : 64)
#define A_YCHUNKS     (LLAMA_D / A_YCHUNK)
#define A_SPAD_Y      0
#define A_SPAD_WO     (A_SPAD_Y + ROWS16(LLAMA_M, A_YCHUNK))
#define A_SPAD_WO_ARG (A_SPAD_WO + ROWS8(LLAMA_H, A_YCHUNK))
#define A_SPAD_OA     A_SPAD_WO_ARG

// ---- MLP half ----
#define M_P_COST(kt, fc) (ROWS8(LLAMA_M, (kt)) + ROWS8((kt), (fc)) + ROWS16(LLAMA_M, (fc)))
#define M_PFITS(kt, fc)  (M_P_COST((kt), (fc)) <= SPAD_ROWS && \
                          SCALE_ROWS((kt), (fc)) <= LLAMA_SCALE_ROWS_MAX && \
                          SCALE_ROWS((kt), LLAMA_M) <= LLAMA_SCALE_ROWS_MAX)
#define M_PKTILE (M_PFITS(LLAMA_D, 16) ? LLAMA_D : M_PFITS(1024, 16) ? 1024 : \
                  M_PFITS(512, 16) ? 512 : M_PFITS(256, 16) ? 256 : 128)
#define M_PKTILES (LLAMA_D / M_PKTILE)
#define M_PKGRP   (M_PKTILE / 32)
#define M_FCHUNK (M_PFITS(M_PKTILE, 512) ? 512 : M_PFITS(M_PKTILE, 256) ? 256 : \
                  M_PFITS(M_PKTILE, 128) ? 128 : M_PFITS(M_PKTILE, 64) ? 64 : \
                  M_PFITS(M_PKTILE, 32) ? 32 : 16)
#define M_FCHUNKS (LLAMA_F / M_FCHUNK)
#define M_SPAD_XN     0
#define M_SPAD_WB     (M_SPAD_XN + ROWS8(LLAMA_M, M_PKTILE))
#define M_SPAD_WB_ARG (M_SPAD_WB + ROWS8(M_PKTILE, M_FCHUNK))
#define M_SPAD_GU     M_SPAD_WB_ARG
#define M_Y_COST(kt, nc) (ROWS8(LLAMA_M, (kt)) + ROWS8((kt), (nc)) + ROWS16(LLAMA_M, (nc)))
#define M_YFITS(kt, nc)  (M_Y_COST((kt), (nc)) <= SPAD_ROWS && \
                          SCALE_ROWS((kt), (nc)) <= LLAMA_SCALE_ROWS_MAX && \
                          SCALE_ROWS((kt), LLAMA_M) <= LLAMA_SCALE_ROWS_MAX)
#define M_NCHUNK (M_YFITS(128, 1024) ? 1024 : M_YFITS(128, 512) ? 512 : \
                  M_YFITS(128, 256) ? 256 : M_YFITS(128, 128) ? 128 : 64)
#define M_NCHUNKS (LLAMA_D / M_NCHUNK)
#define M_KTILE (M_YFITS(512, M_NCHUNK) && (LLAMA_F % 512) == 0 ? 512 : \
                 M_YFITS(256, M_NCHUNK) && (LLAMA_F % 256) == 0 ? 256 : \
                 M_YFITS(128, M_NCHUNK) && (LLAMA_F % 128) == 0 ? 128 : \
                 M_YFITS(64, M_NCHUNK) && (LLAMA_F % 64) == 0 ? 64 : 32)
#define M_KTILES (LLAMA_F / M_KTILE)
#define M_KGRP   (M_KTILE / 32)
#define M_SPAD_H      0
#define M_SPAD_WD     (M_SPAD_H + ROWS8(LLAMA_M, M_KTILE))
#define M_SPAD_WD_ARG (M_SPAD_WD + ROWS8(M_KTILE, M_NCHUNK))
#define M_SPAD_Y      M_SPAD_WD_ARG

// ---- the head: logits = Xf @ W_lm, [M,D] x [D,V]. Same shape as a projection, V-chunked. ----
#if LLAMA_HAS_HEAD
#define L_CHUNK  A_PROJ_N
#define L_CHUNKS (LLAMA_V / L_CHUNK)
#define L_SPAD_XF 0
#define L_SPAD_C  ROWS8(LLAMA_M, LLAMA_D)
#endif

#define LLAMA_REQUIRE(name, cond) typedef char llama_plan_##name[(cond) ? 1 : -1]
LLAMA_REQUIRE(a_proj_scale_fits, SCALE_ROWS(LLAMA_D, A_PROJ_N) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(a_heads_fit,       A_H2_C_O + ROWS8(LLAMA_M, LLAMA_H) <= SPAD_ROWS);
LLAMA_REQUIRE(a_ychunk_divides,  A_YCHUNK * A_YCHUNKS == LLAMA_D);
LLAMA_REQUIRE(a_oproj_fits,      A_SPAD_OA + ROWS8(LLAMA_M, LLAMA_H) <= SPAD_ROWS);
LLAMA_REQUIRE(m_pktile_divides,  M_PKTILE * M_PKTILES == LLAMA_D);
LLAMA_REQUIRE(m_fchunk_divides,  M_FCHUNK * M_FCHUNKS == LLAMA_F);
LLAMA_REQUIRE(m_proj_fits,       M_SPAD_GU + ROWS16(LLAMA_M, M_FCHUNK) <= SPAD_ROWS);
LLAMA_REQUIRE(m_ktile_divides,   M_KTILE * M_KTILES == LLAMA_F);
LLAMA_REQUIRE(m_nchunk_divides,  M_NCHUNK * M_NCHUNKS == LLAMA_D);
LLAMA_REQUIRE(m_down_fits,       M_SPAD_Y + ROWS16(LLAMA_M, M_NCHUNK) <= SPAD_ROWS);
LLAMA_REQUIRE(m_proj_scale_fits, SCALE_ROWS(M_PKTILE, M_FCHUNK) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(m_down_scale_fits, SCALE_ROWS(M_KTILE, M_NCHUNK) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(m_down_ascale_fits, SCALE_ROWS(M_KTILE, LLAMA_M) <= LLAMA_SCALE_ROWS_MAX);
#if LLAMA_HAS_HEAD
LLAMA_REQUIRE(l_chunk_divides_v, L_CHUNK * L_CHUNKS == LLAMA_V);
LLAMA_REQUIRE(l_scale_fits,      SCALE_ROWS(LLAMA_D, L_CHUNK) <= LLAMA_SCALE_ROWS_MAX);
#endif

// ---- host buffers ----
static float    xn_f[LLAMA_M * LLAMA_D];
static uint8_t  xn_codes[LLAMA_M * LLAMA_D];
static uint8_t  xn_scales[LLAMA_GD * LLAMA_M];
static uint16_t Q_hw[LLAMA_M * LLAMA_QD], K_hw[LLAMA_M * LLAMA_KVD], V_hw[LLAMA_M * LLAMA_KVD];
static float    tmp_f[LLAMA_M * LLAMA_H], tmp_t[LLAMA_H * LLAMA_M];
static uint8_t  q_codes[LLAMA_NH][LLAMA_M * LLAMA_H], q_scales[LLAMA_NH][LLAMA_GH * LLAMA_M];
static uint8_t  kt_codes[LLAMA_NKV][LLAMA_H * LLAMA_M], kt_scales[LLAMA_NKV][LLAMA_GH * LLAMA_M];
static uint8_t  v_codes[LLAMA_NKV][LLAMA_M * LLAMA_H], v_scales[LLAMA_NKV][LLAMA_GM * LLAMA_H];
static uint16_t S_hw[LLAMA_M * LLAMA_M];
static float    p_f[LLAMA_M * LLAMA_M];
static uint8_t  p_codes[LLAMA_M * LLAMA_M], p_scales[LLAMA_GM * LLAMA_M];
static uint8_t  O_hw[LLAMA_NH][LLAMA_M * LLAMA_H];
static uint32_t o_scales_dram[LLAMA_NH][64] __attribute__((aligned(32)));
static uint8_t  wo_scales_chunk[LLAMA_GH * A_YCHUNK];
static uint8_t  o_scales_a[LLAMA_GH * LLAMA_M];
static uint8_t  a_proj_scales_chunk[LLAMA_GD * A_PROJ_N];
static float    h_f[LLAMA_M * LLAMA_F];
static uint8_t  h_codes[LLAMA_M * LLAMA_F];
static uint8_t  h_scales[LLAMA_GF * LLAMA_M];
static uint16_t G_hw[LLAMA_M * LLAMA_F], U_hw[LLAMA_M * LLAMA_F];
static uint8_t  m_proj_scales_chunk[M_PKGRP * M_FCHUNK];
static uint8_t  wd_scales_chunk[M_KGRP * M_NCHUNK];
#define CHUNK_MAX (A_YCHUNK > M_NCHUNK ? A_YCHUNK : M_NCHUNK)
static uint16_t chunk16[LLAMA_M * CHUNK_MAX];
static uint16_t Y_hw[LLAMA_M * LLAMA_D];
static uint16_t X_hw[LLAMA_M * LLAMA_D];       // THE RESIDUAL STREAM, carried across layers
static uint16_t Xmid_hw[LLAMA_M * LLAMA_D];
static uint32_t scale_sink[512] __attribute__((aligned(32)));
#if LLAMA_HAS_HEAD
static uint16_t logits_hw[LLAMA_M * LLAMA_V];
static uint8_t  lmh_scales_chunk[LLAMA_GD * L_CHUNK];
#endif

uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  printf("TRAP cause=%d epc=%lx\n", (int) cause, (unsigned long) epc);
  tohost_exit(1337);
  return 0;
}

static void mvin_A_strided(const uint8_t *A, int M, int K, int stride, uint32_t a_spad) {
  gemmini_config_ld(stride * sizeof(uint8_t));
  int tiles_I = M / DIM, tiles_K = K / DIM;
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void *) (A + (size_t) i * DIM * stride + (size_t) k * DIM),
                            a_spad + (i * tiles_K + k) * DIM, DIM, DIM);
}

static void mvin_A(const uint8_t *A, int M, int K, uint32_t a_spad) {
  mvin_A_strided(A, M, K, K, a_spad);
}

static void mvin_B(const uint8_t *B, int K, int N_full, int n0, int N, uint32_t b_spad) {
  gemmini_config_ld(N_full * sizeof(uint8_t));
  int tiles_K = K / DIM, tiles_J = N / DIM;
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++)
      gemmini_extended_mvin((void *) (B + (size_t) k * DIM * N_full + (size_t) (n0 + j * DIM)),
                            b_spad + (k * tiles_J + j) * DIM, DIM, DIM);
}

static void mvout_bf16(uint16_t *dst, uint32_t spad, int M, int N) {
  gemmini_config_st(DIM * sizeof(uint8_t));
  int total_rows = M * N * 2 / DIM;
  uint8_t *b = (uint8_t *) dst;
  for (int r = 0; r < total_rows; r += DIM)
    gemmini_extended_mvout(b + (size_t) r * DIM, spad + r, DIM, DIM);
  gemmini_fence();
}

static void mvout_detile(uint8_t *dst, uint32_t spad, int M, int N) {
  static uint8_t tiled[LLAMA_M * LLAMA_H];
  int tiles_I = M / DIM, tiles_N = N / DIM, total_rows = M * N / DIM;
  gemmini_config_st(DIM * sizeof(uint8_t));
  for (int r = 0; r < total_rows; r += DIM)
    gemmini_extended_mvout(tiled + (size_t) r * DIM, spad + r, DIM, DIM);
  gemmini_fence();
  for (int i = 0; i < tiles_I; i++)
    for (int nt = 0; nt < tiles_N; nt++)
      for (int r = 0; r < DIM; r++)
        for (int c = 0; c < DIM; c++)
          dst[(size_t) (i * DIM + r) * N + nt * DIM + c] =
              tiled[(size_t) ((i * tiles_N + nt) * DIM + r) * DIM + c];
}

static void mesh_matmul(int M, int K, int N, uint32_t a_spad, uint32_t b_arg, uint32_t c_spad,
                        int out_fmt, uint64_t scale_dram, int resident, int accum) {
  int I = M / DIM, J = N / DIM, Kt = K / DIM;
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              0, 0, out_fmt, 0);
  gemmini_config_st((out_fmt == OUT_BF16 ? N * (int) sizeof(uint16_t) : (int) sizeof(uint16_t)));
  // Braces are load-bearing: these are multi-statement macros, so a brace-less if/else does not
  // compile ("'else' without a previous 'if'").
  if (resident) {
    gemmini_mxquant_config_mvout_resident(scale_dram, I, J, Kt, 0, 0, 1);
  } else {
    gemmini_mxquant_config_mvout(scale_dram, I, J, Kt, 0, 0, 1);
  }
  gemmini_loop_ws_spad(I, J, Kt, 0, 0, 0, a_spad, b_arg, 0, c_spad,
                       false, false, false, false, accum, NO_ACTIVATION, 0, 0, false,
                       SPAD_STORE | (out_fmt == OUT_FP8 ? LOOP_WS_REQUANT_TILED : 0));
  gemmini_fence();
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  const uint16_t *EMBED_OUT = LLAMA_AT(LLAMA_OFF_EMBED_OUT, uint16_t);

  printf("llama MODEL: %d layer(s)  M=%d D=%d F=%d heads=%d kv=%d%s (fp8 e4m3 + E8M0)\n",
         LLAMA_NL, LLAMA_M, LLAMA_D, LLAMA_F, LLAMA_NH, LLAMA_NKV,
         LLAMA_HAS_HEAD ? " + lm_head" : "");
  printf("plan  spad %d rows | attn proj N=%d, o_proj %d x %d | mlp proj %d x %d, "
         "down %d x %d (%d K-tiles of %d)\n",
         SPAD_ROWS, A_PROJ_N, A_YCHUNKS, A_YCHUNK, M_FCHUNKS, M_FCHUNK,
         M_NCHUNKS, M_NCHUNK, M_KTILES, M_KTILE);

  gemmini_flush(0);
  uint64_t t_host = 0, t_mesh = 0, t0;
  int total_mesh_diff = 0, total_host_diff = 0, total_seam_diff = 0;

  memcpy(X_hw, EMBED_OUT, sizeof(X_hw));

  for (int n = 0; n < LLAMA_NL; n++) {
    const uint16_t *W_IN_LN   = LLAMA_LAT(n, LOFF_W_IN_LN, uint16_t);
    const uint16_t *W_POST_LN = LLAMA_LAT(n, LOFF_W_POST_LN, uint16_t);
    const uint32_t *ROPE_COS  = LLAMA_LAT(n, LOFF_A_ROPE_COS, uint32_t);
    const uint32_t *ROPE_SIN  = LLAMA_LAT(n, LOFF_A_ROPE_SIN, uint32_t);
    const uint8_t  *A_XN_C    = LLAMA_LAT(n, LOFF_A_XN_CODES, uint8_t);
    const uint8_t  *A_XN_S    = LLAMA_LAT(n, LOFF_A_XN_SCALES, uint8_t);
    const uint8_t  *WQ_C = LLAMA_LAT(n, LOFF_A_WQ_CODES, uint8_t);
    const uint8_t  *WQ_S = LLAMA_LAT(n, LOFF_A_WQ_SCALES, uint8_t);
    const uint8_t  *WK_C = LLAMA_LAT(n, LOFF_A_WK_CODES, uint8_t);
    const uint8_t  *WK_S = LLAMA_LAT(n, LOFF_A_WK_SCALES, uint8_t);
    const uint8_t  *WV_C = LLAMA_LAT(n, LOFF_A_WV_CODES, uint8_t);
    const uint8_t  *WV_S = LLAMA_LAT(n, LOFF_A_WV_SCALES, uint8_t);
    const uint8_t  *WO_C = LLAMA_LAT(n, LOFF_A_WO_CODES, uint8_t);
    const uint8_t  *WO_S = LLAMA_LAT(n, LOFF_A_WO_SCALES, uint8_t);
    const uint16_t *Q_G  = LLAMA_LAT(n, LOFF_A_Q_OUT, uint16_t);
    const uint16_t *K_G  = LLAMA_LAT(n, LOFF_A_K_OUT, uint16_t);
    const uint16_t *V_G  = LLAMA_LAT(n, LOFF_A_V_OUT, uint16_t);
    const uint16_t *S_G  = LLAMA_LAT(n, LOFF_A_S_OUT, uint16_t);
    const uint8_t  *O_G  = LLAMA_LAT(n, LOFF_A_O_CODES, uint8_t);
    const uint8_t  *OS_G = LLAMA_LAT(n, LOFF_A_O_SCALES, uint8_t);
    const uint16_t *YA_G = LLAMA_LAT(n, LOFF_A_Y_OUT, uint16_t);
    const uint16_t *HMID_G = LLAMA_LAT(n, LOFF_H_MID_OUT, uint16_t);
    const uint8_t  *M_XN_C = LLAMA_LAT(n, LOFF_M_XN_CODES, uint8_t);
    const uint8_t  *M_XN_S = LLAMA_LAT(n, LOFF_M_XN_SCALES, uint8_t);
    const uint8_t  *WG_C = LLAMA_LAT(n, LOFF_M_WG_CODES, uint8_t);
    const uint8_t  *WG_S = LLAMA_LAT(n, LOFF_M_WG_SCALES, uint8_t);
    const uint8_t  *WU_C = LLAMA_LAT(n, LOFF_M_WU_CODES, uint8_t);
    const uint8_t  *WU_S = LLAMA_LAT(n, LOFF_M_WU_SCALES, uint8_t);
    const uint8_t  *WD_C = LLAMA_LAT(n, LOFF_M_WD_CODES, uint8_t);
    const uint8_t  *WD_S = LLAMA_LAT(n, LOFF_M_WD_SCALES, uint8_t);
    const uint16_t *G_G  = LLAMA_LAT(n, LOFF_M_G_OUT, uint16_t);
    const uint16_t *U_G  = LLAMA_LAT(n, LOFF_M_U_OUT, uint16_t);
    const uint8_t  *H_G  = LLAMA_LAT(n, LOFF_M_H_CODES, uint8_t);
    const uint8_t  *HS_G = LLAMA_LAT(n, LOFF_M_H_SCALES, uint8_t);
    const uint16_t *YM_G = LLAMA_LAT(n, LOFF_M_Y_OUT, uint16_t);
    const uint16_t *HOUT_G = LLAMA_LAT(n, LOFF_H_OUT_OUT, uint16_t);
    const uint16_t *HOUT_R = LLAMA_LAT(n, LOFF_H_OUT_REF, uint16_t);

    int mesh_d = 0, host_d = 0, seam_d = 0;

    // ---- attention ----
    t0 = read_cycles();
    mx_rmsnorm(X_hw, W_IN_LN, LLAMA_M, LLAMA_D, LLAMA_RMS_EPS, xn_f);
    mx_quantize_rows(xn_f, LLAMA_M, LLAMA_D, xn_codes, xn_scales);
    t_host += read_cycles() - t0;
    host_d += mx_count_diff_u8(xn_codes, A_XN_C, LLAMA_M * LLAMA_D);
    host_d += mx_count_diff_u8(xn_scales, A_XN_S, LLAMA_GD * LLAMA_M);

    t0 = read_cycles();
    mvin_A(xn_codes, LLAMA_M, LLAMA_D, A_SPAD_XN);
    gemmini_mx_load_scales((uint64_t) xn_scales, sizeof(xn_scales), 0);
    gemmini_fence();
    t_mesh += read_cycles() - t0;

    const uint8_t *wc[3]    = { WQ_C, WK_C, WV_C };
    const uint8_t *ws[3]    = { WQ_S, WK_S, WV_S };
    uint16_t *dstp[3]       = { Q_hw, K_hw, V_hw };
    const uint16_t *goldp[3]= { Q_G, K_G, V_G };
    const int wid[3]        = { LLAMA_QD, LLAMA_KVD, LLAMA_KVD };
    for (int s = 0; s < 3; s++) {
      const int nc = wid[s] < A_PROJ_N ? wid[s] : A_PROJ_N;
      for (int c = 0; c < wid[s] / nc; c++) {
        for (int g = 0; g < LLAMA_GD; g++)
          memcpy(a_proj_scales_chunk + (size_t) g * nc,
                 ws[s] + (size_t) g * wid[s] + (size_t) c * nc, nc);
        t0 = read_cycles();
        gemmini_mx_load_scales((uint64_t) a_proj_scales_chunk, LLAMA_GD * nc, 1);
        gemmini_fence();
        mvin_B(wc[s], LLAMA_D, wid[s], c * nc, nc, SPAD_ROWS - ROWS8(LLAMA_D, nc));
        mesh_matmul(LLAMA_M, LLAMA_D, nc, A_SPAD_XN, SPAD_ROWS, A_PROJ_C,
                    OUT_BF16, (uint64_t) scale_sink, 0, 0);
        t_mesh += read_cycles() - t0;
        mvout_bf16(chunk16, A_PROJ_C, LLAMA_M, nc);
        for (int m = 0; m < LLAMA_M; m++)
          memcpy(&dstp[s][(size_t) m * wid[s] + c * nc], &chunk16[(size_t) m * nc],
                 nc * sizeof(uint16_t));
      }
      mesh_d += mx_count_diff_u16(dstp[s], goldp[s], LLAMA_M * wid[s]);
    }

    t0 = read_cycles();
    for (int h = 0; h < LLAMA_NH; h++) {
      mx_rope_at(Q_hw, LLAMA_QD, h * LLAMA_H, ROPE_COS, ROPE_SIN, LLAMA_M, LLAMA_H, tmp_f);
      mx_quantize_rows(tmp_f, LLAMA_M, LLAMA_H, q_codes[h], q_scales[h]);
    }
    for (int kv = 0; kv < LLAMA_NKV; kv++) {
      mx_rope_at(K_hw, LLAMA_KVD, kv * LLAMA_H, ROPE_COS, ROPE_SIN, LLAMA_M, LLAMA_H, tmp_f);
      mx_transpose_f32(tmp_f, LLAMA_M, LLAMA_H, tmp_t);
      mx_quantize_cols(tmp_t, LLAMA_H, LLAMA_M, kt_codes[kv], kt_scales[kv]);
      for (int m = 0; m < LLAMA_M; m++)
        for (int j = 0; j < LLAMA_H; j++)
          tmp_f[(size_t) m * LLAMA_H + j] = mx_bf16_to_f32(V_hw[(size_t) m * LLAMA_KVD
                                                                + kv * LLAMA_H + j]);
      mx_quantize_cols(tmp_f, LLAMA_M, LLAMA_H, v_codes[kv], v_scales[kv]);
    }
    t_host += read_cycles() - t0;

    for (int h = 0; h < LLAMA_NH; h++) {
      const int kv = h / LLAMA_PER;
      t0 = read_cycles();
      mvin_A(q_codes[h], LLAMA_M, LLAMA_H, A_H2_A_Q);
      mvin_B(kt_codes[kv], LLAMA_H, LLAMA_M, 0, LLAMA_M, A_H2_B_KT);
      gemmini_mx_load_scales((uint64_t) q_scales[h], LLAMA_GH * LLAMA_M, 0);
      gemmini_mx_load_scales((uint64_t) kt_scales[kv], LLAMA_GH * LLAMA_M, 1);
      gemmini_fence();
      mesh_matmul(LLAMA_M, LLAMA_H, LLAMA_M, A_H2_A_Q, A_H2_B_KT_ARG, A_H2_C_S,
                  OUT_BF16, (uint64_t) scale_sink, 0, 0);
      t_mesh += read_cycles() - t0;
      mvout_bf16(S_hw, A_H2_C_S, LLAMA_M, LLAMA_M);
      mesh_d += mx_count_diff_u16(S_hw, S_G + (size_t) h * LLAMA_M * LLAMA_M, LLAMA_M * LLAMA_M);

      t0 = read_cycles();
      mx_softmax_causal(S_hw, LLAMA_M, 1.0f / sqrtf((float) LLAMA_H), p_f);
      mx_quantize_rows(p_f, LLAMA_M, LLAMA_M, p_codes, p_scales);
      t_host += read_cycles() - t0;

      t0 = read_cycles();
      mvin_A(p_codes, LLAMA_M, LLAMA_M, A_H2_A_P);
      mvin_B(v_codes[kv], LLAMA_M, LLAMA_H, 0, LLAMA_H, A_H2_B_V);
      gemmini_mx_load_scales((uint64_t) p_scales, LLAMA_GM * LLAMA_M, 0);
      gemmini_mx_load_scales((uint64_t) v_scales[kv], LLAMA_GM * LLAMA_H, 1);
      gemmini_fence();
      mesh_matmul(LLAMA_M, LLAMA_M, LLAMA_H, A_H2_A_P, A_H2_B_V_ARG, A_H2_C_O,
                  OUT_FP8, (uint64_t) o_scales_dram[h], 1, 0);
      t_mesh += read_cycles() - t0;
      mvout_detile(O_hw[h], A_H2_C_O, LLAMA_M, LLAMA_H);
      mesh_d += mx_count_diff_u8(O_hw[h], O_G + (size_t) h * LLAMA_M * LLAMA_H,
                                 LLAMA_M * LLAMA_H);
      mesh_d += mx_count_diff_u8((const uint8_t *) o_scales_dram[h],
                                 OS_G + (size_t) h * LLAMA_M * LLAMA_GH, LLAMA_M * LLAMA_GH);
    }

    for (int c = 0; c < A_YCHUNKS; c++) {
      for (int h = 0; h < LLAMA_NH; h++) {
        for (int g = 0; g < LLAMA_GH; g++)
          memcpy(wo_scales_chunk + (size_t) g * A_YCHUNK,
                 WO_S + (size_t) (h * LLAMA_GH + g) * LLAMA_D + c * A_YCHUNK, A_YCHUNK);
        const uint8_t *osrc = (const uint8_t *) o_scales_dram[h];
        for (int g = 0; g < LLAMA_GH; g++)
          for (int m = 0; m < LLAMA_M; m++)
            o_scales_a[(size_t) g * LLAMA_M + m] = osrc[(size_t) m * LLAMA_GH + g];
        t0 = read_cycles();
        mvin_A(O_hw[h], LLAMA_M, LLAMA_H, A_SPAD_OA);
        gemmini_mx_load_scales((uint64_t) o_scales_a, sizeof(o_scales_a), 0);
        gemmini_mx_load_scales((uint64_t) wo_scales_chunk, sizeof(wo_scales_chunk), 1);
        gemmini_fence();
        mvin_B(WO_C + (size_t) h * LLAMA_H * LLAMA_D, LLAMA_H, LLAMA_D, c * A_YCHUNK,
               A_YCHUNK, A_SPAD_WO);
        mesh_matmul(LLAMA_M, LLAMA_H, A_YCHUNK, A_SPAD_OA, A_SPAD_WO_ARG, A_SPAD_Y,
                    OUT_BF16, (uint64_t) scale_sink, 0, h > 0);
        t_mesh += read_cycles() - t0;
      }
      mvout_bf16(chunk16, A_SPAD_Y, LLAMA_M, A_YCHUNK);
      for (int m = 0; m < LLAMA_M; m++)
        memcpy(&Y_hw[(size_t) m * LLAMA_D + c * A_YCHUNK], &chunk16[(size_t) m * A_YCHUNK],
               A_YCHUNK * sizeof(uint16_t));
    }
    mesh_d += mx_count_diff_u16(Y_hw, YA_G, LLAMA_M * LLAMA_D);

    // ---- residual 1 ----
    t0 = read_cycles();
    for (int i = 0; i < LLAMA_M * LLAMA_D; i++)
      Xmid_hw[i] = mx_f32_to_bf16_rne(mx_bf16_to_f32(X_hw[i]) + mx_bf16_to_f32(Y_hw[i]));
    t_host += read_cycles() - t0;
    seam_d += mx_count_diff_u16(Xmid_hw, HMID_G, LLAMA_M * LLAMA_D);

    // ---- MLP ----
    t0 = read_cycles();
    mx_rmsnorm(Xmid_hw, W_POST_LN, LLAMA_M, LLAMA_D, LLAMA_RMS_EPS, xn_f);
    mx_quantize_rows(xn_f, LLAMA_M, LLAMA_D, xn_codes, xn_scales);
    t_host += read_cycles() - t0;
    host_d += mx_count_diff_u8(xn_codes, M_XN_C, LLAMA_M * LLAMA_D);
    host_d += mx_count_diff_u8(xn_scales, M_XN_S, LLAMA_GD * LLAMA_M);

    t0 = read_cycles();
#if M_PKTILES == 1
    mvin_A_strided(xn_codes, LLAMA_M, LLAMA_D, LLAMA_D, M_SPAD_XN);
    gemmini_fence();
#endif
    {
      const uint8_t *mwc[2] = { WG_C, WU_C };
      const uint8_t *mws[2] = { WG_S, WU_S };
      uint16_t *mdst[2]     = { G_hw, U_hw };
      for (int s = 0; s < 2; s++)
        for (int c = 0; c < M_FCHUNKS; c++) {
          for (int t = 0; t < M_PKTILES; t++) {
            for (int g = 0; g < M_PKGRP; g++)
              memcpy(m_proj_scales_chunk + (size_t) g * M_FCHUNK,
                     mws[s] + (size_t) (t * M_PKGRP + g) * LLAMA_F + (size_t) c * M_FCHUNK,
                     M_FCHUNK);
#if M_PKTILES > 1
            mvin_A_strided(xn_codes + (size_t) t * M_PKTILE, LLAMA_M, M_PKTILE, LLAMA_D,
                           M_SPAD_XN);
#endif
            gemmini_mx_load_scales((uint64_t) (xn_scales + (size_t) t * M_PKGRP * LLAMA_M),
                                   M_PKGRP * LLAMA_M, 0);
            gemmini_mx_load_scales((uint64_t) m_proj_scales_chunk, M_PKGRP * M_FCHUNK, 1);
            gemmini_fence();
            mvin_B(mwc[s] + (size_t) t * M_PKTILE * LLAMA_F, M_PKTILE, LLAMA_F, c * M_FCHUNK,
                   M_FCHUNK, M_SPAD_WB);
            mesh_matmul(LLAMA_M, M_PKTILE, M_FCHUNK, M_SPAD_XN, M_SPAD_WB_ARG, M_SPAD_GU,
                        OUT_BF16, (uint64_t) scale_sink, 0, t > 0);
          }
          mvout_bf16(chunk16, M_SPAD_GU, LLAMA_M, M_FCHUNK);
          for (int m = 0; m < LLAMA_M; m++)
            memcpy(&mdst[s][(size_t) m * LLAMA_F + (size_t) c * M_FCHUNK],
                   &chunk16[(size_t) m * M_FCHUNK], M_FCHUNK * sizeof(uint16_t));
        }
    }
    t_mesh += read_cycles() - t0;
    mesh_d += mx_count_diff_u16(G_hw, G_G, LLAMA_M * LLAMA_F);
    mesh_d += mx_count_diff_u16(U_hw, U_G, LLAMA_M * LLAMA_F);

    t0 = read_cycles();
    mx_swiglu(G_hw, U_hw, LLAMA_M * LLAMA_F, h_f);
    mx_quantize_rows(h_f, LLAMA_M, LLAMA_F, h_codes, h_scales);
    t_host += read_cycles() - t0;
    host_d += mx_count_diff_u8(h_codes, H_G, LLAMA_M * LLAMA_F);
    host_d += mx_count_diff_u8(h_scales, HS_G, LLAMA_GF * LLAMA_M);

    for (int c = 0; c < M_NCHUNKS; c++) {
      for (int t = 0; t < M_KTILES; t++) {
        for (int g = 0; g < M_KGRP; g++)
          memcpy(wd_scales_chunk + (size_t) g * M_NCHUNK,
                 WD_S + (size_t) (t * M_KGRP + g) * LLAMA_D + (size_t) c * M_NCHUNK, M_NCHUNK);
        t0 = read_cycles();
        mvin_A_strided(h_codes + (size_t) t * M_KTILE, LLAMA_M, M_KTILE, LLAMA_F, M_SPAD_H);
        gemmini_mx_load_scales((uint64_t) (h_scales + (size_t) t * M_KGRP * LLAMA_M),
                               M_KGRP * LLAMA_M, 0);
        gemmini_mx_load_scales((uint64_t) wd_scales_chunk, M_KGRP * M_NCHUNK, 1);
        gemmini_fence();
        mvin_B(WD_C + (size_t) t * M_KTILE * LLAMA_D, M_KTILE, LLAMA_D, c * M_NCHUNK,
               M_NCHUNK, M_SPAD_WD);
        mesh_matmul(LLAMA_M, M_KTILE, M_NCHUNK, M_SPAD_H, M_SPAD_WD_ARG, M_SPAD_Y,
                    OUT_BF16, (uint64_t) scale_sink, 0, t > 0);
        t_mesh += read_cycles() - t0;
      }
      mvout_bf16(chunk16, M_SPAD_Y, LLAMA_M, M_NCHUNK);
      for (int m = 0; m < LLAMA_M; m++)
        memcpy(&Y_hw[(size_t) m * LLAMA_D + (size_t) c * M_NCHUNK],
               &chunk16[(size_t) m * M_NCHUNK], M_NCHUNK * sizeof(uint16_t));
    }
    mesh_d += mx_count_diff_u16(Y_hw, YM_G, LLAMA_M * LLAMA_D);

    // ---- residual 2: this layer's output becomes the next layer's input ----
    t0 = read_cycles();
    for (int i = 0; i < LLAMA_M * LLAMA_D; i++)
      X_hw[i] = mx_f32_to_bf16_rne(mx_bf16_to_f32(Xmid_hw[i]) + mx_bf16_to_f32(Y_hw[i]));
    t_host += read_cycles() - t0;
    seam_d += mx_count_diff_u16(X_hw, HOUT_G, LLAMA_M * LLAMA_D);

    // Drift against the MODEL's own h_out for this layer -- what localizes a divergence.
    printf("layer %2d  mesh %d  host %d  seam %d  |  h_out vs model: rel_fro %d ppm\n",
           n, mesh_d, host_d, seam_d,
           MX_PPM(mx_rel_fro_bf16(X_hw, HOUT_R, LLAMA_M * LLAMA_D)));
    total_mesh_diff += mesh_d;
    total_host_diff += host_d;
    total_seam_diff += seam_d;
  }

#if LLAMA_HAS_HEAD
  // ---- the head: final RMSNorm, then logits = Xf @ W_lm ----
  {
    const uint16_t *W_FINAL_LN = LLAMA_AT(LLAMA_OFF_W_FINAL_LN, uint16_t);
    const uint8_t  *XF_C   = LLAMA_AT(LLAMA_OFF_XF_CODES, uint8_t);
    const uint8_t  *XF_S   = LLAMA_AT(LLAMA_OFF_XF_SCALES, uint8_t);
    const uint8_t  *LMH_C  = LLAMA_AT(LLAMA_OFF_LMH_CODES, uint8_t);
    const uint8_t  *LMH_S  = LLAMA_AT(LLAMA_OFF_LMH_SCALES, uint8_t);
    const uint16_t *LG_G   = LLAMA_AT(LLAMA_OFF_LOGITS_OUT, uint16_t);
    const float    *LG_T   = LLAMA_AT(LLAMA_OFF_LOGITS_TORCH, float);
    const int32_t  *LABELS = LLAMA_AT(LLAMA_OFF_LABELS, int32_t);

    t0 = read_cycles();
    mx_rmsnorm(X_hw, W_FINAL_LN, LLAMA_M, LLAMA_D, LLAMA_RMS_EPS, xn_f);
    mx_quantize_rows(xn_f, LLAMA_M, LLAMA_D, xn_codes, xn_scales);
    t_host += read_cycles() - t0;
    int hd = mx_count_diff_u8(xn_codes, XF_C, LLAMA_M * LLAMA_D)
           + mx_count_diff_u8(xn_scales, XF_S, LLAMA_GD * LLAMA_M);
    total_host_diff += hd;
    printf("host  final rmsnorm+quant: %d byte(s) differ\n", hd);

    t0 = read_cycles();
    mvin_A(xn_codes, LLAMA_M, LLAMA_D, L_SPAD_XF);
    gemmini_mx_load_scales((uint64_t) xn_scales, sizeof(xn_scales), 0);
    gemmini_fence();
    t_mesh += read_cycles() - t0;
    for (int c = 0; c < L_CHUNKS; c++) {
      for (int g = 0; g < LLAMA_GD; g++)
        memcpy(lmh_scales_chunk + (size_t) g * L_CHUNK,
               LMH_S + (size_t) g * LLAMA_V + (size_t) c * L_CHUNK, L_CHUNK);
      t0 = read_cycles();
      gemmini_mx_load_scales((uint64_t) lmh_scales_chunk, LLAMA_GD * L_CHUNK, 1);
      gemmini_fence();
      mvin_B(LMH_C, LLAMA_D, LLAMA_V, c * L_CHUNK, L_CHUNK,
             SPAD_ROWS - ROWS8(LLAMA_D, L_CHUNK));
      mesh_matmul(LLAMA_M, LLAMA_D, L_CHUNK, L_SPAD_XF, SPAD_ROWS, L_SPAD_C,
                  OUT_BF16, (uint64_t) scale_sink, 0, 0);
      t_mesh += read_cycles() - t0;
      mvout_bf16(chunk16, L_SPAD_C, LLAMA_M, L_CHUNK);
      for (int m = 0; m < LLAMA_M; m++)
        memcpy(&logits_hw[(size_t) m * LLAMA_V + (size_t) c * L_CHUNK],
               &chunk16[(size_t) m * L_CHUNK], L_CHUNK * sizeof(uint16_t));
    }
    int ld = mx_count_diff_u16(logits_hw, LG_G, LLAMA_M * LLAMA_V);
    total_mesh_diff += ld;
    printf("mesh  logits = Xf @ W_lm : %d/%d differ (%d chunks of %d)\n",
           ld, LLAMA_M * LLAMA_V, L_CHUNKS, L_CHUNK);

    // ---- what the model is FOR: argmax agreement and perplexity ----
    t0 = read_cycles();
    int agree = 0;
    float nll_mx = 0.0f, nll_t = 0.0f;
    for (int m = 0; m < LLAMA_M; m++) {
      const uint16_t *row = &logits_hw[(size_t) m * LLAMA_V];
      const float *rowt = &LG_T[(size_t) m * LLAMA_V];
      int am = 0, at = 0;
      float best = mx_bf16_to_f32(row[0]), bestt = rowt[0];
      for (int v = 1; v < LLAMA_V; v++) {
        float x = mx_bf16_to_f32(row[v]);
        if (x > best) { best = x; am = v; }
        if (rowt[v] > bestt) { bestt = rowt[v]; at = v; }
      }
      if (am == at) agree++;
      // log-sum-exp in fp32, shifted by the max for stability
      float se = 0.0f, set = 0.0f;
      for (int v = 0; v < LLAMA_V; v++) {
        se += expf(mx_bf16_to_f32(row[v]) - best);
        set += expf(rowt[v] - bestt);
      }
      int lab = (int) LABELS[m];
      nll_mx += logf(se) + best - mx_bf16_to_f32(row[lab]);
      nll_t  += logf(set) + bestt - rowt[lab];
    }
    nll_mx /= (float) LLAMA_M;
    nll_t  /= (float) LLAMA_M;
    t_host += read_cycles() - t0;

    printf("grade logits vs torch      : rel_fro %d ppm\n",
           MX_PPM(mx_rel_fro_bf16_f32(logits_hw, LG_T, LLAMA_M * LLAMA_V)));
    printf("grade argmax agrees on %d/%d tokens\n", agree, LLAMA_M);
    printf("grade nll  MX %d.%03d  vs torch %d.%03d  (x1000)\n",
           (int) nll_mx, (int) ((nll_mx - (int) nll_mx) * 1000),
           (int) nll_t, (int) ((nll_t - (int) nll_t) * 1000));
    printf("grade ppl  MX %d.%03d  vs torch %d.%03d\n",
           (int) expf(nll_mx), (int) ((expf(nll_mx) - (int) expf(nll_mx)) * 1000),
           (int) expf(nll_t), (int) ((expf(nll_t) - (int) expf(nll_t)) * 1000));
  }
#else
  printf("grade final hidden vs model: rel_fro %d ppm\n",
         MX_PPM(mx_rel_fro_bf16(X_hw, LLAMA_AT(LLAMA_OFF_H_FINAL_REF, uint16_t),
                                LLAMA_M * LLAMA_D)));
#endif

  printf("cycles mesh %d, host %d\n", (int) t_mesh, (int) t_host);
  if (total_mesh_diff == 0 && total_seam_diff == 0 && total_host_diff == 0)
    printf("llama MODEL test PASSED (%d layer(s)%s; every mesh stage, every seam and every host "
           "stage bit-exact).\n", LLAMA_NL, LLAMA_HAS_HEAD ? " + lm_head" : "");
  else if (total_mesh_diff == 0 && total_seam_diff == 0)
    printf("llama MODEL test PASSED WITH DRIFT: mesh and seams bit-exact, %d host byte(s) "
           "differ.\n", total_host_diff);
  else
    printf("llama MODEL test FAILED: %d mesh, %d seam element(s) differ (%d host byte(s) too).\n",
           total_mesh_diff, total_seam_diff, total_host_diff);

#ifndef BAREMETAL
  exit((total_mesh_diff + total_seam_diff) != 0);
#else
  return (total_mesh_diff + total_seam_diff) != 0;
#endif
}
