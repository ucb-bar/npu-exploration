// ONE COMPLETE TinyLlama DECODER LAYER on MxGemmini -- attention and MLP, both whole, one ELF.
//
// `llama_attention_full.c` and `llama_mlp_full.c` each compute one sub-layer against a residual
// stream the capture handed them. This computes the LAYER: the two halves joined by both RMSNorms
// and both residuals, exactly as LlamaDecoderLayer.forward does it.
//
//   host   xn1   = rmsnorm(h_pre, w_in_ln)                    fp32 -> MX
//   mesh   Q,K,V = Xn1 @ Wq/Wk/Wv        (N-chunked, Xn1 resident across all three)
//   host   RoPE per head; K transposed per kv head
//   mesh   per head  S_h = Q_h @ K_kv^T
//   host   per head  causal softmax                           fp32 -> MX
//   mesh   per head  O_h = P_h @ V_kv    -> FP8 requant into the scratchpad
//   mesh   Yattn = sum_h O_h @ Wo_h      (ACCUMULATED IN SMEM ACROSS HEADS)
//   host   h_mid = h_pre + Yattn                              <-- RESIDUAL 1, THE SEAM
//   host   xn2   = rmsnorm(h_mid, w_post_ln)                  fp32 -> MX
//   mesh   G,U   = Xn2 @ Wg/Wu           (F-chunked, Xn2 resident across both)
//   host   H     = silu(G) * U                                fp32 -> MX
//   mesh   Ymlp  = H @ Wd                (K-TILED: K=5632 exceeds the scale window)
//   host   h_out = h_mid + Ymlp                               <-- RESIDUAL 2
//
// THE SEAM IS WHAT IS NEW. Everything else here already ran, bit-exact, in the two sub-layer
// kernels. What has never run is the MLP over an `h_mid` the DEVICE produced: `h_pre` plus an
// attention output carrying the accumulated MX error of 32 heads and four matmul stages. Its
// RMSNorm is a different vector, its fp8 codes are different, and the two halves' errors COMPOUND
// rather than add. That is precisely why a stacked model cannot be predicted from per-sub-layer
// numbers, and it is the quantity this kernel exists to measure.
//
// THE GOLDENS FOLLOW THE DEVICE, NOT THE CAPTURE. Every M_* golden in the blob was re-derived from
// the residual the model of this chain computed, so the bit-exact gate still holds end to end:
// a mismatch means the hardware diverged, not that the reference was drawn from somewhere else.
// The residual itself is checked too (H_MID_OUT), because it is the one host stage whose output is
// re-quantized -- a single ulp there flips an E4M3 code and cascades through the entire MLP.
//
// H_OUT_TORCH is the layer's TRUE output from TinyLlama's own forward pass. Neither sub-layer
// kernel could be graded against it.
//
// The two halves run SEQUENTIALLY and each replans the whole scratchpad from row 0, so their
// region maps are independent; they are namespaced A_* and M_* because the names mean different
// things in each. Every width is chosen at COMPILE TIME from BANK_NUM * BANK_ROWS.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "mx_host.h"
#include "llama_layer_full.h"

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

// -DLAYER_NATIVE: every weight matmul (Q/K/V, per-head S, o_proj over all heads, G/U, down) is one
// mxn_matmul (include/mx_native.h): native DRAM loops, K-tiles x N-chunks, each loop in its own spad half,
// loop-managed scales, C straight to DRAM. P@V keeps the resident scratchpad-requant loop per head.
// -DLLAMA_GOLDEN_HOST: the host stages (RMSNorm, RoPE, softmax, SwiGLU, residual 1) take their GOLDEN
// outputs from the blob instead of computing them, so an RTL run is mostly mesh time; every mesh stage is
// still checked bit-exact.
#ifdef LAYER_NATIVE
#include "mx_native.h"
#endif

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
// Rows one matmul needs in each scale window: B side (N/16)*(K/32), A side the same with M for N.
#define SCALE_ROWS(k, n) ((n) * (k) / 512)

// =============================================================================================
// ATTENTION HALF -- the plan of llama_attention_full.c, namespaced.
// =============================================================================================
// -- phase A1: the projections, [M,D] x [D,Nc]. Xn1 resident as A across all three.
#define A_P1_COST(nc) (ROWS8(LLAMA_M, LLAMA_D) + ROWS8(LLAMA_D, (nc)) + ROWS16(LLAMA_M, (nc)))
#define A_P1FITS(nc)  (A_P1_COST(nc) <= SPAD_ROWS && \
                       SCALE_ROWS(LLAMA_D, (nc)) <= LLAMA_SCALE_ROWS_MAX && \
                       SCALE_ROWS(LLAMA_D, LLAMA_M) <= LLAMA_SCALE_ROWS_MAX)
#define A_PROJ_N   (A_P1FITS(512) ? 512 : \
                    A_P1FITS(256) ? 256 : \
                    A_P1FITS(128) ? 128 : \
                    A_P1FITS(64)  ? 64  : \
                    A_P1FITS(32)  ? 32  : 16)
#define A_SPAD_XN     0
#define A_PROJ_C      ROWS8(LLAMA_M, LLAMA_D)
#define A_PROJ_B      (SPAD_ROWS - ROWS8(LLAMA_D, A_PROJ_N))
#define A_PROJ_B_ARG  SPAD_ROWS

// -- phase A2: per head, S = Q_h @ K_kv^T then O = P_h @ V_kv. Regions reused across heads.
#define A_H2_A_Q      0
#define A_H2_B_KT     (A_H2_A_Q + ROWS8(LLAMA_M, LLAMA_H))
#define A_H2_B_KT_ARG (A_H2_B_KT + ROWS8(LLAMA_H, LLAMA_M))
#define A_H2_A_P      A_H2_B_KT_ARG
#define A_H2_B_V      (A_H2_A_P + ROWS8(LLAMA_M, LLAMA_M))
#define A_H2_B_V_ARG  (A_H2_B_V + ROWS8(LLAMA_M, LLAMA_H))
#define A_H2_C_S      A_H2_B_V_ARG
#define A_H2_C_O      (A_H2_C_S + ROWS16(LLAMA_M, LLAMA_M))

// -- phase A3: o_proj. One Y chunk stays resident while all NH heads accumulate into it.
#define A_P3_COST(dc) (ROWS16(LLAMA_M, (dc)) + ROWS8(LLAMA_H, (dc)) + ROWS8(LLAMA_M, LLAMA_H))
#define A_YCHUNK   (A_P3_COST(2048) <= SPAD_ROWS ? 2048 : \
                    A_P3_COST(1024) <= SPAD_ROWS ? 1024 : \
                    A_P3_COST(512)  <= SPAD_ROWS ? 512  : \
                    A_P3_COST(256)  <= SPAD_ROWS ? 256  : \
                    A_P3_COST(128)  <= SPAD_ROWS ? 128  : 64)
#define A_YCHUNKS     (LLAMA_D / A_YCHUNK)
#define A_SPAD_Y      0
#define A_SPAD_WO     (A_SPAD_Y + ROWS16(LLAMA_M, A_YCHUNK))
#define A_SPAD_WO_ARG (A_SPAD_WO + ROWS8(LLAMA_H, A_YCHUNK))
#define A_SPAD_OA     A_SPAD_WO_ARG

// =============================================================================================
// MLP HALF -- the plan of llama_mlp_full.c, namespaced.
// =============================================================================================
// -- phase M1: gate/up, [M,D] x [D,Fc]. Fewest K-tiles first, so Xn2 is moved in ONCE.
#define M_P_COST(kt, fc) (ROWS8(LLAMA_M, (kt)) + ROWS8((kt), (fc)) + ROWS16(LLAMA_M, (fc)))
#define M_PFITS(kt, fc)  (M_P_COST((kt), (fc)) <= SPAD_ROWS && \
                          SCALE_ROWS((kt), (fc)) <= LLAMA_SCALE_ROWS_MAX && \
                          SCALE_ROWS((kt), LLAMA_M) <= LLAMA_SCALE_ROWS_MAX)
#define M_PKTILE (M_PFITS(LLAMA_D, 16) ? LLAMA_D : \
                  M_PFITS(1024, 16)    ? 1024    : \
                  M_PFITS(512, 16)     ? 512     : \
                  M_PFITS(256, 16)     ? 256     : 128)
#define M_PKTILES (LLAMA_D / M_PKTILE)
#define M_PKGRP   (M_PKTILE / 32)
#define M_FCHUNK (M_PFITS(M_PKTILE, 512) ? 512 : \
                  M_PFITS(M_PKTILE, 256) ? 256 : \
                  M_PFITS(M_PKTILE, 128) ? 128 : \
                  M_PFITS(M_PKTILE, 64)  ? 64  : \
                  M_PFITS(M_PKTILE, 32)  ? 32  : 16)
#define M_FCHUNKS (LLAMA_F / M_FCHUNK)
#define M_SPAD_XN     0
#define M_SPAD_WB     (M_SPAD_XN + ROWS8(LLAMA_M, M_PKTILE))
#define M_SPAD_WB_ARG (M_SPAD_WB + ROWS8(M_PKTILE, M_FCHUNK))
#define M_SPAD_GU     M_SPAD_WB_ARG

// -- phase M2: down_proj, [M,F] x [F,Dc], K-tiled. F = 5632 = 2^9 * 11, so the K-tile candidates
//    that DIVIDE it are the powers of two up to 512; 1024 is deliberately absent.
#define M_Y_COST(kt, nc) (ROWS8(LLAMA_M, (kt)) + ROWS8((kt), (nc)) + ROWS16(LLAMA_M, (nc)))
#define M_YFITS(kt, nc)  (M_Y_COST((kt), (nc)) <= SPAD_ROWS && \
                          SCALE_ROWS((kt), (nc)) <= LLAMA_SCALE_ROWS_MAX && \
                          SCALE_ROWS((kt), LLAMA_M) <= LLAMA_SCALE_ROWS_MAX)
#define M_NCHUNK (M_YFITS(128, 1024) ? 1024 : \
                  M_YFITS(128, 512)  ? 512  : \
                  M_YFITS(128, 256)  ? 256  : \
                  M_YFITS(128, 128)  ? 128  : 64)
#define M_NCHUNKS (LLAMA_D / M_NCHUNK)
#define M_KTILE (M_YFITS(512, M_NCHUNK) && (LLAMA_F % 512) == 0 ? 512 : \
                 M_YFITS(256, M_NCHUNK) && (LLAMA_F % 256) == 0 ? 256 : \
                 M_YFITS(128, M_NCHUNK) && (LLAMA_F % 128) == 0 ? 128 : \
                 M_YFITS(64,  M_NCHUNK) && (LLAMA_F % 64)  == 0 ? 64  : 32)
#define M_KTILES (LLAMA_F / M_KTILE)
#define M_KGRP   (M_KTILE / 32)
#define M_SPAD_H      0
#define M_SPAD_WD     (M_SPAD_H + ROWS8(LLAMA_M, M_KTILE))
#define M_SPAD_WD_ARG (M_SPAD_WD + ROWS8(M_KTILE, M_NCHUNK))
#define M_SPAD_Y      M_SPAD_WD_ARG

// Checked at COMPILE time: an overflowing region aliases silently and yields plausible numbers.
#define LLAMA_REQUIRE(name, cond) typedef char llama_plan_##name[(cond) ? 1 : -1]
LLAMA_REQUIRE(a_proj_fits,       A_PROJ_C + ROWS16(LLAMA_M, A_PROJ_N) <= A_PROJ_B);
LLAMA_REQUIRE(a_heads_fit,       A_H2_C_O + ROWS8(LLAMA_M, LLAMA_H) <= SPAD_ROWS);
LLAMA_REQUIRE(a_ychunk_divides,  A_YCHUNK * A_YCHUNKS == LLAMA_D);
LLAMA_REQUIRE(a_oproj_fits,      A_SPAD_OA + ROWS8(LLAMA_M, LLAMA_H) <= SPAD_ROWS);
LLAMA_REQUIRE(a_proj_n_divides,  (LLAMA_QD % A_PROJ_N) == 0);
LLAMA_REQUIRE(a_proj_scale_fits, SCALE_ROWS(LLAMA_D, A_PROJ_N) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(m_pktile_divides,  M_PKTILE * M_PKTILES == LLAMA_D);
LLAMA_REQUIRE(m_pktile_blocked,  (M_PKTILE % 32) == 0);
LLAMA_REQUIRE(m_fchunk_divides,  M_FCHUNK * M_FCHUNKS == LLAMA_F);
LLAMA_REQUIRE(m_proj_fits,       M_SPAD_GU + ROWS16(LLAMA_M, M_FCHUNK) <= SPAD_ROWS);
LLAMA_REQUIRE(m_ktile_divides,   M_KTILE * M_KTILES == LLAMA_F);
LLAMA_REQUIRE(m_ktile_blocked,   (M_KTILE % 32) == 0);
LLAMA_REQUIRE(m_nchunk_divides,  M_NCHUNK * M_NCHUNKS == LLAMA_D);
LLAMA_REQUIRE(m_down_fits,       M_SPAD_Y + ROWS16(LLAMA_M, M_NCHUNK) <= SPAD_ROWS);
LLAMA_REQUIRE(m_proj_scale_fits, SCALE_ROWS(M_PKTILE, M_FCHUNK) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(m_proj_ascale_fits, SCALE_ROWS(M_PKTILE, LLAMA_M) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(m_down_scale_fits, SCALE_ROWS(M_KTILE, M_NCHUNK) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(m_down_ascale_fits, SCALE_ROWS(M_KTILE, LLAMA_M) <= LLAMA_SCALE_ROWS_MAX);

// ---- host buffers -----------------------------------------------------------------------------
// xn_* serve BOTH halves: attention finishes with its Xn1 before the MLP needs Xn2, and the
// operand lives in the scratchpad meanwhile.
static float    xn_f[LLAMA_M * LLAMA_D];
static uint8_t  xn_codes[LLAMA_M * LLAMA_D] __attribute__((aligned(64)));
static uint8_t  xn_scales[LLAMA_GD * LLAMA_M] __attribute__((aligned(64)));
// attention
static uint16_t Q_hw[LLAMA_M * LLAMA_QD] __attribute__((aligned(64))), K_hw[LLAMA_M * LLAMA_KVD] __attribute__((aligned(64))), V_hw[LLAMA_M * LLAMA_KVD] __attribute__((aligned(64)));
static float    tmp_f[LLAMA_M * LLAMA_H], tmp_t[LLAMA_H * LLAMA_M];
static uint8_t  q_codes[LLAMA_NH][LLAMA_M * LLAMA_H] __attribute__((aligned(64))), q_scales[LLAMA_NH][LLAMA_GH * LLAMA_M] __attribute__((aligned(64)));
static uint8_t  kt_codes[LLAMA_NKV][LLAMA_H * LLAMA_M] __attribute__((aligned(64))), kt_scales[LLAMA_NKV][LLAMA_GH * LLAMA_M] __attribute__((aligned(64)));
static uint8_t  v_codes[LLAMA_NKV][LLAMA_M * LLAMA_H] __attribute__((aligned(64))), v_scales[LLAMA_NKV][LLAMA_GM * LLAMA_H] __attribute__((aligned(64)));
static uint16_t S_hw[LLAMA_NH][LLAMA_M * LLAMA_M] __attribute__((aligned(64)));
static float    p_f[LLAMA_M * LLAMA_M];
static uint8_t  p_codes[LLAMA_NH][LLAMA_M * LLAMA_M] __attribute__((aligned(64))), p_scales[LLAMA_NH][LLAMA_GM * LLAMA_M] __attribute__((aligned(64)));
static uint8_t  O_hw[LLAMA_NH][LLAMA_M * LLAMA_H] __attribute__((aligned(64)));
static uint32_t o_scales_dram[LLAMA_NH][64] __attribute__((aligned(32)));
static uint8_t  wo_scales_chunk[LLAMA_GH * A_YCHUNK] __attribute__((aligned(64)));
static uint8_t  o_scales_a[LLAMA_GH * LLAMA_M] __attribute__((aligned(64)));
static uint8_t  a_proj_scales_chunk[LLAMA_GD * A_PROJ_N] __attribute__((aligned(64)));
// mlp
static float    h_f[LLAMA_M * LLAMA_F];
static uint8_t  h_codes[LLAMA_M * LLAMA_F] __attribute__((aligned(64)));
static uint8_t  h_scales[LLAMA_GF * LLAMA_M] __attribute__((aligned(64)));
static uint16_t G_hw[LLAMA_M * LLAMA_F] __attribute__((aligned(64))), U_hw[LLAMA_M * LLAMA_F] __attribute__((aligned(64)));
static uint8_t  m_proj_scales_chunk[M_PKGRP * M_FCHUNK] __attribute__((aligned(64)));
static uint8_t  wd_scales_chunk[M_KGRP * M_NCHUNK] __attribute__((aligned(64)));
// shared scratch + the residual stream
#define CHUNK_MAX (A_PROJ_N > A_YCHUNK ? A_PROJ_N : \
                   (A_YCHUNK > M_FCHUNK ? A_YCHUNK : \
                    (M_FCHUNK > M_NCHUNK ? M_FCHUNK : M_NCHUNK)))
static uint16_t chunk16[LLAMA_M * CHUNK_MAX] __attribute__((aligned(64)));
static uint16_t Yattn_hw[LLAMA_M * LLAMA_D] __attribute__((aligned(64)));
static uint16_t Ymlp_hw[LLAMA_M * LLAMA_D] __attribute__((aligned(64)));
static uint16_t H_MID_hw[LLAMA_M * LLAMA_D] __attribute__((aligned(64)));
static uint16_t H_OUT_hw[LLAMA_M * LLAMA_D] __attribute__((aligned(64)));
static uint32_t scale_sink[512] __attribute__((aligned(32)));
#ifdef LAYER_NATIVE
static uint8_t  O_cat[LLAMA_M * LLAMA_QD] __attribute__((aligned(64)));      // [M][QD]: heads along K
static uint8_t  oa_scales[(LLAMA_QD / 32) * LLAMA_M] __attribute__((aligned(64)));   // [QD/32][M]
enum { NP_QKV, NP_S, NP_O, NP_OPROJ, NP_GU, NP_DOWN, NP_N };
static uint64_t nat_ph[NP_N];
#endif

uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  printf("TRAP cause=%d epc=%lx\n", (int) cause, (unsigned long) epc);
  tohost_exit(1337);
  return 0;
}

// ---- movers -----------------------------------------------------------------------------------
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

// Read a BLOCK-TILED fp8 tile back to a flat [M][N] buffer -- contiguous mvout then a software
// de-tile, because a strided de-tiling mvout makes the writer DMA emit whole cache lines and
// zero-fill the gaps on RTL (matmul_tiled_fp8_64x64_chain.c:79-83).
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

// `accum` is loop_ws's ex_accumulate (rs1 bit 0): 0 OVERWRITES the output region, 1 adds into it.
static void mesh_matmul(int M, int K, int N, uint32_t a_spad, uint32_t b_arg, uint32_t c_spad,
                        int out_fmt, uint64_t scale_dram, int resident, int accum) {
  int I = M / DIM, J = N / DIM, Kt = K / DIM;
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              0, 0, out_fmt, 0);
  gemmini_config_st((out_fmt == OUT_BF16 ? N * (int) sizeof(uint16_t) : (int) sizeof(uint16_t)));
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
  const uint16_t *H_PRE       = LLAMA_AT(LLAMA_OFF_H_PRE, uint16_t);
  const uint16_t *W_IN_LN     = LLAMA_AT(LLAMA_OFF_W_IN_LN, uint16_t);
  const uint16_t *W_POST_LN   = LLAMA_AT(LLAMA_OFF_W_POST_LN, uint16_t);
  const uint32_t *ROPE_COS    = LLAMA_AT(LLAMA_OFF_A_ROPE_COS, uint32_t);
  const uint32_t *ROPE_SIN    = LLAMA_AT(LLAMA_OFF_A_ROPE_SIN, uint32_t);
  const uint8_t  *A_XN_CODES  = LLAMA_AT(LLAMA_OFF_A_XN_CODES, uint8_t);
  const uint8_t  *A_XN_SCALES = LLAMA_AT(LLAMA_OFF_A_XN_SCALES, uint8_t);
  const uint8_t  *WQ_CODES    = LLAMA_AT(LLAMA_OFF_A_WQ_CODES, uint8_t);
  const uint8_t  *WQ_SCALES   = LLAMA_AT(LLAMA_OFF_A_WQ_SCALES, uint8_t);
  const uint8_t  *WK_CODES    = LLAMA_AT(LLAMA_OFF_A_WK_CODES, uint8_t);
  const uint8_t  *WK_SCALES   = LLAMA_AT(LLAMA_OFF_A_WK_SCALES, uint8_t);
  const uint8_t  *WV_CODES    = LLAMA_AT(LLAMA_OFF_A_WV_CODES, uint8_t);
  const uint8_t  *WV_SCALES   = LLAMA_AT(LLAMA_OFF_A_WV_SCALES, uint8_t);
  const uint8_t  *WO_CODES    = LLAMA_AT(LLAMA_OFF_A_WO_CODES, uint8_t);
  const uint8_t  *WO_SCALES   = LLAMA_AT(LLAMA_OFF_A_WO_SCALES, uint8_t);
  const uint16_t *Q_OUT       = LLAMA_AT(LLAMA_OFF_A_Q_OUT, uint16_t);
  const uint16_t *K_OUT       = LLAMA_AT(LLAMA_OFF_A_K_OUT, uint16_t);
  const uint16_t *V_OUT       = LLAMA_AT(LLAMA_OFF_A_V_OUT, uint16_t);
  const uint8_t  *G_Q_CODES   = LLAMA_AT(LLAMA_OFF_A_Q_CODES, uint8_t);
  const uint8_t  *G_KT_CODES  = LLAMA_AT(LLAMA_OFF_A_KT_CODES, uint8_t);
  const uint8_t  *G_V_CODES   = LLAMA_AT(LLAMA_OFF_A_V_CODES, uint8_t);
  const uint16_t *S_OUT       = LLAMA_AT(LLAMA_OFF_A_S_OUT, uint16_t);
  const uint8_t  *G_P_CODES   = LLAMA_AT(LLAMA_OFF_A_P_CODES, uint8_t);
  const uint8_t  *O_OUT       = LLAMA_AT(LLAMA_OFF_A_O_CODES, uint8_t);
  const uint8_t  *O_SCALES    = LLAMA_AT(LLAMA_OFF_A_O_SCALES, uint8_t);
  const uint16_t *YATTN_OUT   = LLAMA_AT(LLAMA_OFF_A_Y_OUT, uint16_t);
  const uint16_t *H_MID_OUT   = LLAMA_AT(LLAMA_OFF_H_MID_OUT, uint16_t);
  const uint8_t  *M_XN_CODES  = LLAMA_AT(LLAMA_OFF_M_XN_CODES, uint8_t);
  const uint8_t  *M_XN_SCALES = LLAMA_AT(LLAMA_OFF_M_XN_SCALES, uint8_t);
  const uint8_t  *WG_CODES    = LLAMA_AT(LLAMA_OFF_M_WG_CODES, uint8_t);
  const uint8_t  *WG_SCALES   = LLAMA_AT(LLAMA_OFF_M_WG_SCALES, uint8_t);
  const uint8_t  *WU_CODES    = LLAMA_AT(LLAMA_OFF_M_WU_CODES, uint8_t);
  const uint8_t  *WU_SCALES   = LLAMA_AT(LLAMA_OFF_M_WU_SCALES, uint8_t);
  const uint8_t  *WD_CODES    = LLAMA_AT(LLAMA_OFF_M_WD_CODES, uint8_t);
  const uint8_t  *WD_SCALES   = LLAMA_AT(LLAMA_OFF_M_WD_SCALES, uint8_t);
  const uint16_t *G_OUT       = LLAMA_AT(LLAMA_OFF_M_G_OUT, uint16_t);
  const uint16_t *U_OUT       = LLAMA_AT(LLAMA_OFF_M_U_OUT, uint16_t);
  const uint8_t  *H_CODES_G   = LLAMA_AT(LLAMA_OFF_M_H_CODES, uint8_t);
  const uint8_t  *H_SCALES_G  = LLAMA_AT(LLAMA_OFF_M_H_SCALES, uint8_t);
  const uint16_t *YMLP_OUT    = LLAMA_AT(LLAMA_OFF_M_Y_OUT, uint16_t);
  const uint16_t *H_OUT_OUT   = LLAMA_AT(LLAMA_OFF_H_OUT_OUT, uint16_t);
  const uint16_t *H_OUT_TORCH = LLAMA_AT(LLAMA_OFF_H_OUT_TORCH, uint16_t);
  const uint16_t *ATTN_TORCH  = LLAMA_AT(LLAMA_OFF_ATTN_TORCH, uint16_t);
  const uint16_t *MLP_TORCH   = LLAMA_AT(LLAMA_OFF_MLP_TORCH, uint16_t);

  printf("llama DECODER LAYER: M=%d D=%d F=%d head_dim=%d heads=%d kv_heads=%d (fp8 e4m3 + E8M0)\n",
         LLAMA_M, LLAMA_D, LLAMA_F, LLAMA_H, LLAMA_NH, LLAMA_NKV);
  printf("plan  spad %d rows | attn: proj N=%d, %d heads, o_proj %d x %d\n",
         SPAD_ROWS, A_PROJ_N, LLAMA_NH, A_YCHUNKS, A_YCHUNK);
  printf("plan  mlp: proj %d x %d (%d K-tile(s) of %d), down %d x %d (%d K-tiles of %d)\n",
         M_FCHUNKS, M_FCHUNK, M_PKTILES, M_PKTILE, M_NCHUNKS, M_NCHUNK, M_KTILES, M_KTILE);

  gemmini_flush(0);
  uint64_t t_host = 0, t_mesh = 0, t0;

  // ==================== ATTENTION HALF ====================
  t0 = read_cycles();
#ifdef LLAMA_GOLDEN_HOST
  memcpy(xn_codes, A_XN_CODES, sizeof(xn_codes)); memcpy(xn_scales, A_XN_SCALES, sizeof(xn_scales));
  (void) W_IN_LN;
#else
  mx_rmsnorm(H_PRE, W_IN_LN, LLAMA_M, LLAMA_D, LLAMA_RMS_EPS, xn_f);
  mx_quantize_rows(xn_f, LLAMA_M, LLAMA_D, xn_codes, xn_scales);
#endif
  t_host += read_cycles() - t0;
  int a_xn_cd = mx_count_diff_u8(xn_codes, A_XN_CODES, LLAMA_M * LLAMA_D);
  int a_xn_sd = mx_count_diff_u8(xn_scales, A_XN_SCALES, LLAMA_GD * LLAMA_M);
  printf("host  rmsnorm1+quant: codes differ %d/%d, scales differ %d/%d\n",
         a_xn_cd, LLAMA_M * LLAMA_D, a_xn_sd, LLAMA_GD * LLAMA_M);

  const uint8_t *wc[3]    = { WQ_CODES, WK_CODES, WV_CODES };
  const uint8_t *ws[3]    = { WQ_SCALES, WK_SCALES, WV_SCALES };
  uint16_t *dst[3]        = { Q_hw, K_hw, V_hw };
  const uint16_t *gold[3] = { Q_OUT, K_OUT, V_OUT };
  const int wid[3]        = { LLAMA_QD, LLAMA_KVD, LLAMA_KVD };
  const char *pn[3]       = { "Q", "K", "V" };

#ifdef LAYER_NATIVE
  t0 = read_cycles();
  for (int s = 0; s < 3; s++)
    if (mxn_matmul(xn_codes, LLAMA_D, wc[s], wid[s], dst[s], wid[s], xn_scales, LLAMA_M, ws[s], wid[s],
                   LLAMA_M, LLAMA_D, wid[s])) return 1;
  gemmini_fence();
  nat_ph[NP_QKV] = read_cycles() - t0;
  t_mesh += nat_ph[NP_QKV];
  int proj_diff = 0;
  for (int s = 0; s < 3; s++) {
    int d = mx_count_diff_u16(dst[s], gold[s], LLAMA_M * wid[s]);
    proj_diff += d;
    printf("mesh  %s = Xn1 @ W%s : %d/%d differ (native)\n", pn[s], pn[s], d, LLAMA_M * wid[s]);
  }
  for (int s = 0; s < 0; s++) {
#else
  t0 = read_cycles();
  mvin_A(xn_codes, LLAMA_M, LLAMA_D, A_SPAD_XN);
  gemmini_mx_load_scales((uint64_t) xn_scales, sizeof(xn_scales), 0);
  gemmini_fence();
  t_mesh += read_cycles() - t0;

  int proj_diff = 0;
  for (int s = 0; s < 3; s++) {
#endif
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
        memcpy(&dst[s][(size_t) m * wid[s] + c * nc], &chunk16[(size_t) m * nc],
               nc * sizeof(uint16_t));
    }
    int d = mx_count_diff_u16(dst[s], gold[s], LLAMA_M * wid[s]);
    proj_diff += d;
    printf("mesh  %s = Xn1 @ W%s : %d/%d differ (%d chunks of %d)\n",
           pn[s], pn[s], d, LLAMA_M * wid[s], wid[s] / nc, nc);
  }

  t0 = read_cycles();
#ifdef LLAMA_GOLDEN_HOST
  memcpy(q_codes, G_Q_CODES, sizeof(q_codes));
  memcpy(q_scales, LLAMA_AT(LLAMA_OFF_A_Q_SCALES, uint8_t), sizeof(q_scales));
  memcpy(kt_codes, G_KT_CODES, sizeof(kt_codes));
  memcpy(kt_scales, LLAMA_AT(LLAMA_OFF_A_KT_SCALES, uint8_t), sizeof(kt_scales));
  memcpy(v_codes, G_V_CODES, sizeof(v_codes));
  memcpy(v_scales, LLAMA_AT(LLAMA_OFF_A_V_SCALES, uint8_t), sizeof(v_scales));
  (void) ROPE_COS; (void) ROPE_SIN;
  for (int h = 0; h < 0; h++) {
#else
  for (int h = 0; h < LLAMA_NH; h++) {
#endif
    mx_rope_at(Q_hw, LLAMA_QD, h * LLAMA_H, ROPE_COS, ROPE_SIN, LLAMA_M, LLAMA_H, tmp_f);
    mx_quantize_rows(tmp_f, LLAMA_M, LLAMA_H, q_codes[h], q_scales[h]);
  }
#ifdef LLAMA_GOLDEN_HOST
  for (int kv = 0; kv < 0; kv++) {
#else
  for (int kv = 0; kv < LLAMA_NKV; kv++) {
#endif
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
  {
    int qd = 0, kd = 0, vd = 0;
    for (int h = 0; h < LLAMA_NH; h++)
      qd += mx_count_diff_u8(q_codes[h], G_Q_CODES + (size_t) h * LLAMA_M * LLAMA_H,
                             LLAMA_M * LLAMA_H);
    for (int kv = 0; kv < LLAMA_NKV; kv++) {
      kd += mx_count_diff_u8(kt_codes[kv], G_KT_CODES + (size_t) kv * LLAMA_H * LLAMA_M,
                             LLAMA_H * LLAMA_M);
      vd += mx_count_diff_u8(v_codes[kv], G_V_CODES + (size_t) kv * LLAMA_M * LLAMA_H,
                             LLAMA_M * LLAMA_H);
    }
    printf("host  RoPE(%d heads) + K^T: Q %d, K^T %d, V %d codes differ\n",
           LLAMA_NH, qd, kd, vd);
  }

  int s_diff = 0, o_diff = 0, os_diff = 0;
#ifdef LAYER_NATIVE
  // S for all heads as chained native matmuls, one fence; then softmax; then P@V per head (resident
  // requant loop, unchanged) with O read back and concatenated along K for one o_proj.
  t0 = read_cycles();
  for (int h = 0; h < LLAMA_NH; h++) {
    const int kv = h / LLAMA_PER;
    if (mxn_matmul(q_codes[h], LLAMA_H, kt_codes[kv], LLAMA_M, S_hw[h], LLAMA_M,
                   q_scales[h], LLAMA_M, kt_scales[kv], LLAMA_M, LLAMA_M, LLAMA_H, LLAMA_M)) return 1;
  }
  gemmini_fence();
  nat_ph[NP_S] = read_cycles() - t0;
  t_mesh += nat_ph[NP_S];
  for (int h = 0; h < LLAMA_NH; h++)
    s_diff += mx_count_diff_u16(S_hw[h], S_OUT + (size_t) h * LLAMA_M * LLAMA_M, LLAMA_M * LLAMA_M);
  t0 = read_cycles();
#ifdef LLAMA_GOLDEN_HOST
  memcpy(p_codes, G_P_CODES, sizeof(p_codes));
  memcpy(p_scales, LLAMA_AT(LLAMA_OFF_A_P_SCALES, uint8_t), sizeof(p_scales));
#else
  for (int h = 0; h < LLAMA_NH; h++) {
    mx_softmax_causal(S_hw[h], LLAMA_M, 1.0f / sqrtf((float) LLAMA_H), p_f);
    mx_quantize_rows(p_f, LLAMA_M, LLAMA_M, p_codes[h], p_scales[h]);
  }
#endif
  t_host += read_cycles() - t0;
  for (int h = 0; h < LLAMA_NH; h++) {
    const int kv = h / LLAMA_PER;
    t0 = read_cycles();
    mvin_A(p_codes[h], LLAMA_M, LLAMA_M, A_H2_A_P);
    mvin_B(v_codes[kv], LLAMA_M, LLAMA_H, 0, LLAMA_H, A_H2_B_V);
    gemmini_mx_load_scales((uint64_t) p_scales[h], LLAMA_GM * LLAMA_M, 0);
    gemmini_mx_load_scales((uint64_t) v_scales[kv], LLAMA_GM * LLAMA_H, 1);
    gemmini_fence();
    mesh_matmul(LLAMA_M, LLAMA_M, LLAMA_H, A_H2_A_P, A_H2_B_V_ARG, A_H2_C_O,
                OUT_FP8, (uint64_t) o_scales_dram[h], 1, 0);
    uint64_t dt = read_cycles() - t0;
    nat_ph[NP_O] += dt;
    t_mesh += dt;
    mvout_detile(O_hw[h], A_H2_C_O, LLAMA_M, LLAMA_H);
    o_diff += mx_count_diff_u8(O_hw[h], O_OUT + (size_t) h * LLAMA_M * LLAMA_H, LLAMA_M * LLAMA_H);
    os_diff += mx_count_diff_u8((const uint8_t *) o_scales_dram[h],
                                O_SCALES + (size_t) h * LLAMA_M * LLAMA_GH, LLAMA_M * LLAMA_GH);
  }
  for (int h = 0; h < 0; h++) {
    const int kv = h / LLAMA_PER;
#else
  for (int h = 0; h < LLAMA_NH; h++) {
    const int kv = h / LLAMA_PER;
#endif
    t0 = read_cycles();
    mvin_A(q_codes[h], LLAMA_M, LLAMA_H, A_H2_A_Q);
    mvin_B(kt_codes[kv], LLAMA_H, LLAMA_M, 0, LLAMA_M, A_H2_B_KT);
    gemmini_mx_load_scales((uint64_t) q_scales[h], LLAMA_GH * LLAMA_M, 0);
    gemmini_mx_load_scales((uint64_t) kt_scales[kv], LLAMA_GH * LLAMA_M, 1);
    gemmini_fence();
    mesh_matmul(LLAMA_M, LLAMA_H, LLAMA_M, A_H2_A_Q, A_H2_B_KT_ARG, A_H2_C_S,
                OUT_BF16, (uint64_t) scale_sink, 0, 0);
    t_mesh += read_cycles() - t0;
    mvout_bf16(S_hw[h], A_H2_C_S, LLAMA_M, LLAMA_M);
    s_diff += mx_count_diff_u16(S_hw[h], S_OUT + (size_t) h * LLAMA_M * LLAMA_M,
                                LLAMA_M * LLAMA_M);

    t0 = read_cycles();
    mx_softmax_causal(S_hw[h], LLAMA_M, 1.0f / sqrtf((float) LLAMA_H), p_f);
    mx_quantize_rows(p_f, LLAMA_M, LLAMA_M, p_codes[h], p_scales[h]);
    t_host += read_cycles() - t0;

    t0 = read_cycles();
    mvin_A(p_codes[h], LLAMA_M, LLAMA_M, A_H2_A_P);
    mvin_B(v_codes[kv], LLAMA_M, LLAMA_H, 0, LLAMA_H, A_H2_B_V);
    gemmini_mx_load_scales((uint64_t) p_scales[h], LLAMA_GM * LLAMA_M, 0);
    gemmini_mx_load_scales((uint64_t) v_scales[kv], LLAMA_GM * LLAMA_H, 1);
    gemmini_fence();
    mesh_matmul(LLAMA_M, LLAMA_M, LLAMA_H, A_H2_A_P, A_H2_B_V_ARG, A_H2_C_O,
                OUT_FP8, (uint64_t) o_scales_dram[h], 1, 0);
    t_mesh += read_cycles() - t0;
    mvout_detile(O_hw[h], A_H2_C_O, LLAMA_M, LLAMA_H);
    o_diff += mx_count_diff_u8(O_hw[h], O_OUT + (size_t) h * LLAMA_M * LLAMA_H,
                               LLAMA_M * LLAMA_H);
    os_diff += mx_count_diff_u8((const uint8_t *) o_scales_dram[h],
                                O_SCALES + (size_t) h * LLAMA_M * LLAMA_GH,
                                LLAMA_M * LLAMA_GH);
  }
  {
    int pd = 0;
    for (int h = 0; h < LLAMA_NH; h++)
      pd += mx_count_diff_u8(p_codes[h], G_P_CODES + (size_t) h * LLAMA_M * LLAMA_M,
                             LLAMA_M * LLAMA_M);
    printf("mesh  S = Q @ K^T : %d/%d   host softmax: P %d/%d differ\n",
           s_diff, LLAMA_NH * LLAMA_M * LLAMA_M, pd, LLAMA_NH * LLAMA_M * LLAMA_M);
  }
  printf("mesh  O = P @ V   : %d/%d codes, %d/%d scales differ (requant -> spad)\n",
         o_diff, LLAMA_NH * LLAMA_M * LLAMA_H, os_diff, LLAMA_NH * LLAMA_M * LLAMA_GH);

#ifdef LAYER_NATIVE
  // o_proj = O_cat[M][QD] @ Wo[QD][D]: all heads along K, one native matmul (16 K-tiles x 4 chunks).
  t0 = read_cycles();
  for (int h = 0; h < LLAMA_NH; h++) {
    const uint8_t *osrc = (const uint8_t *) o_scales_dram[h];   // [M][GH] as the requantizer wrote it
    for (int m = 0; m < LLAMA_M; m++) {
      memcpy(&O_cat[(size_t) m * LLAMA_QD + h * LLAMA_H], &O_hw[h][(size_t) m * LLAMA_H], LLAMA_H);
      for (int g = 0; g < LLAMA_GH; g++)
        oa_scales[(size_t) (h * LLAMA_GH + g) * LLAMA_M + m] = osrc[(size_t) m * LLAMA_GH + g];
    }
  }
  t_host += read_cycles() - t0;
  t0 = read_cycles();
  if (mxn_matmul(O_cat, LLAMA_QD, WO_CODES, LLAMA_D, Yattn_hw, LLAMA_D, oa_scales, LLAMA_M,
                 WO_SCALES, LLAMA_D, LLAMA_M, LLAMA_QD, LLAMA_D)) return 1;
  gemmini_fence();
  nat_ph[NP_OPROJ] = read_cycles() - t0;
  t_mesh += nat_ph[NP_OPROJ];
  for (int c = 0; c < 0; c++) {
#else
  for (int c = 0; c < A_YCHUNKS; c++) {
#endif
    for (int h = 0; h < LLAMA_NH; h++) {
      for (int g = 0; g < LLAMA_GH; g++)
        memcpy(wo_scales_chunk + (size_t) g * A_YCHUNK,
               WO_SCALES + (size_t) (h * LLAMA_GH + g) * LLAMA_D + c * A_YCHUNK, A_YCHUNK);
      const uint8_t *osrc = (const uint8_t *) o_scales_dram[h];      // [M][GH] as written
      for (int g = 0; g < LLAMA_GH; g++)
        for (int m = 0; m < LLAMA_M; m++)
          o_scales_a[(size_t) g * LLAMA_M + m] = osrc[(size_t) m * LLAMA_GH + g];
      t0 = read_cycles();
      mvin_A(O_hw[h], LLAMA_M, LLAMA_H, A_SPAD_OA);
      gemmini_mx_load_scales((uint64_t) o_scales_a, sizeof(o_scales_a), 0);
      gemmini_mx_load_scales((uint64_t) wo_scales_chunk, sizeof(wo_scales_chunk), 1);
      gemmini_fence();
      mvin_B(WO_CODES + (size_t) h * LLAMA_H * LLAMA_D, LLAMA_H, LLAMA_D, c * A_YCHUNK,
             A_YCHUNK, A_SPAD_WO);
      mesh_matmul(LLAMA_M, LLAMA_H, A_YCHUNK, A_SPAD_OA, A_SPAD_WO_ARG, A_SPAD_Y,
                  OUT_BF16, (uint64_t) scale_sink, 0, h > 0);
      t_mesh += read_cycles() - t0;
    }
    mvout_bf16(chunk16, A_SPAD_Y, LLAMA_M, A_YCHUNK);
    for (int m = 0; m < LLAMA_M; m++)
      memcpy(&Yattn_hw[(size_t) m * LLAMA_D + c * A_YCHUNK], &chunk16[(size_t) m * A_YCHUNK],
             A_YCHUNK * sizeof(uint16_t));
  }
  int ya_d = mx_count_diff_u16(Yattn_hw, YATTN_OUT, LLAMA_M * LLAMA_D);
  printf("mesh  Yattn = sum_h O_h @ Wo_h : %d/%d differ (%d heads accumulated)\n",
         ya_d, LLAMA_M * LLAMA_D, LLAMA_NH);

  // ==================== THE SEAM: residual 1 ====================
  // h_mid feeds the MLP's RMSNorm and is re-quantized straight afterwards, so this is the one
  // host stage whose result must match the golden to the BIT -- an ulp here flips an E4M3 code.
  t0 = read_cycles();
#ifdef LLAMA_GOLDEN_HOST
  memcpy(H_MID_hw, H_MID_OUT, sizeof(H_MID_hw));
#else
  for (int i = 0; i < LLAMA_M * LLAMA_D; i++)
    H_MID_hw[i] = mx_f32_to_bf16_rne(mx_bf16_to_f32(H_PRE[i]) + mx_bf16_to_f32(Yattn_hw[i]));
#endif
  t_host += read_cycles() - t0;
  int hm_d = mx_count_diff_u16(H_MID_hw, H_MID_OUT, LLAMA_M * LLAMA_D);
  printf("host  h_mid = h_pre + Yattn : %d/%d differ from golden  <-- THE SEAM\n",
         hm_d, LLAMA_M * LLAMA_D);

  // ==================== MLP HALF, on the device's own h_mid ====================
  t0 = read_cycles();
#ifdef LLAMA_GOLDEN_HOST
  memcpy(xn_codes, M_XN_CODES, sizeof(xn_codes)); memcpy(xn_scales, M_XN_SCALES, sizeof(xn_scales));
  (void) W_POST_LN;
#else
  mx_rmsnorm(H_MID_hw, W_POST_LN, LLAMA_M, LLAMA_D, LLAMA_RMS_EPS, xn_f);
  mx_quantize_rows(xn_f, LLAMA_M, LLAMA_D, xn_codes, xn_scales);
#endif
  t_host += read_cycles() - t0;
  int m_xn_cd = mx_count_diff_u8(xn_codes, M_XN_CODES, LLAMA_M * LLAMA_D);
  int m_xn_sd = mx_count_diff_u8(xn_scales, M_XN_SCALES, LLAMA_GD * LLAMA_M);
  printf("host  rmsnorm2+quant: codes differ %d/%d, scales differ %d/%d\n",
         m_xn_cd, LLAMA_M * LLAMA_D, m_xn_sd, LLAMA_GD * LLAMA_M);

  t0 = read_cycles();
#ifdef LAYER_NATIVE
  if (mxn_matmul(xn_codes, LLAMA_D, WG_CODES, LLAMA_F, G_hw, LLAMA_F, xn_scales, LLAMA_M, WG_SCALES, LLAMA_F,
                 LLAMA_M, LLAMA_D, LLAMA_F) ||
      mxn_matmul(xn_codes, LLAMA_D, WU_CODES, LLAMA_F, U_hw, LLAMA_F, xn_scales, LLAMA_M, WU_SCALES, LLAMA_F,
                 LLAMA_M, LLAMA_D, LLAMA_F)) return 1;
  gemmini_fence();
  nat_ph[NP_GU] = read_cycles() - t0;
  if (0)
#elif M_PKTILES == 1
  mvin_A_strided(xn_codes, LLAMA_M, LLAMA_D, LLAMA_D, M_SPAD_XN);
  gemmini_fence();
#endif
  {
    const uint8_t *mwc[2] = { WG_CODES, WU_CODES };
    const uint8_t *mws[2] = { WG_SCALES, WU_SCALES };
    uint16_t *mdst[2]     = { G_hw, U_hw };
    for (int s = 0; s < 2; s++) {
      for (int c = 0; c < M_FCHUNKS; c++) {
        for (int t = 0; t < M_PKTILES; t++) {
          for (int g = 0; g < M_PKGRP; g++)
            memcpy(m_proj_scales_chunk + (size_t) g * M_FCHUNK,
                   mws[s] + (size_t) (t * M_PKGRP + g) * LLAMA_F + (size_t) c * M_FCHUNK,
                   M_FCHUNK);
#if M_PKTILES > 1
          mvin_A_strided(xn_codes + (size_t) t * M_PKTILE, LLAMA_M, M_PKTILE, LLAMA_D, M_SPAD_XN);
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
  }
  t_mesh += read_cycles() - t0;
  int g_d = mx_count_diff_u16(G_hw, G_OUT, LLAMA_M * LLAMA_F);
  int u_d = mx_count_diff_u16(U_hw, U_OUT, LLAMA_M * LLAMA_F);
  printf("mesh  G = Xn2 @ Wg : %d/%d   U = Xn2 @ Wu : %d/%d differ (%d chunks of %d)\n",
         g_d, LLAMA_M * LLAMA_F, u_d, LLAMA_M * LLAMA_F, M_FCHUNKS, M_FCHUNK);

  t0 = read_cycles();
#ifdef LLAMA_GOLDEN_HOST
  memcpy(h_codes, H_CODES_G, sizeof(h_codes)); memcpy(h_scales, H_SCALES_G, sizeof(h_scales));
#else
  mx_swiglu(G_hw, U_hw, LLAMA_M * LLAMA_F, h_f);
  mx_quantize_rows(h_f, LLAMA_M, LLAMA_F, h_codes, h_scales);
#endif
  t_host += read_cycles() - t0;
  int h_cd = mx_count_diff_u8(h_codes, H_CODES_G, LLAMA_M * LLAMA_F);
  int h_sd = mx_count_diff_u8(h_scales, H_SCALES_G, LLAMA_GF * LLAMA_M);
  printf("host  silu(G)*U + quant: codes differ %d/%d, scales differ %d/%d\n",
         h_cd, LLAMA_M * LLAMA_F, h_sd, LLAMA_GF * LLAMA_M);

#ifdef LAYER_NATIVE
  t0 = read_cycles();
  if (mxn_matmul(h_codes, LLAMA_F, WD_CODES, LLAMA_D, Ymlp_hw, LLAMA_D, h_scales, LLAMA_M, WD_SCALES, LLAMA_D,
                 LLAMA_M, LLAMA_F, LLAMA_D)) return 1;
  gemmini_fence();
  nat_ph[NP_DOWN] = read_cycles() - t0;
  t_mesh += nat_ph[NP_DOWN];
  for (int c = 0; c < 0; c++) {
#else
  for (int c = 0; c < M_NCHUNKS; c++) {
#endif
    for (int t = 0; t < M_KTILES; t++) {
      for (int g = 0; g < M_KGRP; g++)
        memcpy(wd_scales_chunk + (size_t) g * M_NCHUNK,
               WD_SCALES + (size_t) (t * M_KGRP + g) * LLAMA_D + (size_t) c * M_NCHUNK, M_NCHUNK);
      t0 = read_cycles();
      mvin_A_strided(h_codes + (size_t) t * M_KTILE, LLAMA_M, M_KTILE, LLAMA_F, M_SPAD_H);
      gemmini_mx_load_scales((uint64_t) (h_scales + (size_t) t * M_KGRP * LLAMA_M),
                             M_KGRP * LLAMA_M, 0);
      gemmini_mx_load_scales((uint64_t) wd_scales_chunk, M_KGRP * M_NCHUNK, 1);
      gemmini_fence();
      mvin_B(WD_CODES + (size_t) t * M_KTILE * LLAMA_D, M_KTILE, LLAMA_D, c * M_NCHUNK,
             M_NCHUNK, M_SPAD_WD);
      mesh_matmul(LLAMA_M, M_KTILE, M_NCHUNK, M_SPAD_H, M_SPAD_WD_ARG, M_SPAD_Y,
                  OUT_BF16, (uint64_t) scale_sink, 0, t > 0);
      t_mesh += read_cycles() - t0;
    }
    mvout_bf16(chunk16, M_SPAD_Y, LLAMA_M, M_NCHUNK);
    for (int m = 0; m < LLAMA_M; m++)
      memcpy(&Ymlp_hw[(size_t) m * LLAMA_D + (size_t) c * M_NCHUNK],
             &chunk16[(size_t) m * M_NCHUNK], M_NCHUNK * sizeof(uint16_t));
  }
  int ym_d = mx_count_diff_u16(Ymlp_hw, YMLP_OUT, LLAMA_M * LLAMA_D);
  printf("mesh  Ymlp = H @ Wd : %d/%d differ (%d chunks of %d, %d K-tiles of %d)\n",
         ym_d, LLAMA_M * LLAMA_D, M_NCHUNKS, M_NCHUNK, M_KTILES, M_KTILE);

  // ==================== residual 2: the layer's output ====================
  t0 = read_cycles();
  for (int i = 0; i < LLAMA_M * LLAMA_D; i++)
    H_OUT_hw[i] = mx_f32_to_bf16_rne(mx_bf16_to_f32(H_MID_hw[i]) + mx_bf16_to_f32(Ymlp_hw[i]));
  t_host += read_cycles() - t0;
  int ho_d = mx_count_diff_u16(H_OUT_hw, H_OUT_OUT, LLAMA_M * LLAMA_D);
  printf("host  h_out = h_mid + Ymlp : %d/%d differ from golden\n", ho_d, LLAMA_M * LLAMA_D);

  // ---- grading ----
  printf("grade attention out vs the model's : rel_fro %d ppm\n",
         MX_PPM(mx_rel_fro_bf16(Yattn_hw, ATTN_TORCH, LLAMA_M * LLAMA_D)));
  printf("grade MLP out       vs the model's : rel_fro %d ppm  (on the DEVICE's h_mid)\n",
         MX_PPM(mx_rel_fro_bf16(Ymlp_hw, MLP_TORCH, LLAMA_M * LLAMA_D)));
  printf("grade LAYER OUT h_out vs THE MODEL's h_out : rel_fro %d ppm\n",
         MX_PPM(mx_rel_fro_bf16(H_OUT_hw, H_OUT_TORCH, LLAMA_M * LLAMA_D)));
  printf("cycles mesh %d, host %d\n", (int) t_mesh, (int) t_host);
#ifdef LAYER_NATIVE
  {
    const uint64_t ideal[NP_N] = {
      MXN_IDEAL(LLAMA_M, LLAMA_D, LLAMA_QD + 2 * LLAMA_KVD), LLAMA_NH * MXN_IDEAL(LLAMA_M, LLAMA_H, LLAMA_M),
      LLAMA_NH * MXN_IDEAL(LLAMA_M, LLAMA_M, LLAMA_H), MXN_IDEAL(LLAMA_M, LLAMA_QD, LLAMA_D),
      2 * MXN_IDEAL(LLAMA_M, LLAMA_D, LLAMA_F), MXN_IDEAL(LLAMA_M, LLAMA_F, LLAMA_D) };
    const char *nm[NP_N] = { "QKV", "S x32", "O x32 (spad requant)", "o_proj", "G,U", "down" };
    uint64_t tot = 0, itot = 0;
    for (int p = 0; p < NP_N; p++) {
      printf("phase %-22s %9d cyc  ideal %9d  util %3d%%\n", nm[p], (int) nat_ph[p], (int) ideal[p],
             nat_ph[p] ? (int) (100 * ideal[p] / nat_ph[p]) : 0);
      tot += nat_ph[p]; itot += ideal[p];
    }
    printf("phase %-22s %9d cyc  ideal %9d  util %3d%%\n", "LAYER MESH", (int) tot, (int) itot,
           (int) (100 * itot / tot));
  }
#endif

  int exact = proj_diff + s_diff + o_diff + os_diff + ya_d + g_d + u_d + ym_d;
  int seam = hm_d + ho_d;
  int host_drift = a_xn_cd + a_xn_sd + m_xn_cd + m_xn_sd + h_cd + h_sd;
  if (exact == 0 && seam == 0 && host_drift == 0)
    printf("llama DECODER LAYER test PASSED (all %d heads, all %d neurons, both residuals; "
           "every mesh stage and both seams bit-exact).\n", LLAMA_NH, LLAMA_F);
  else if (exact == 0 && seam == 0)
    printf("llama DECODER LAYER test PASSED WITH DRIFT: mesh and seams bit-exact, but %d host "
           "byte(s) differ from the golden.\n", host_drift);
  else
    printf("llama DECODER LAYER test FAILED: %d mesh element(s), %d seam element(s) differ "
           "(%d host byte(s) differed too).\n", exact, seam, host_drift);

#ifndef BAREMETAL
  exit((exact + seam) != 0);
#else
  return (exact + seam) != 0;
#endif
}
