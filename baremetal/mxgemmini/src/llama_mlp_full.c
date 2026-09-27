// A COMPLETE TinyLlama MLP sub-layer on MxGemmini -- ALL 5632 FFN neurons, one ELF, real data.
//
// Unlike llama_mlp.c, which runs 64 of 5632 neurons and so grades a partial sum over ~1% of
// down_proj's reduction, nothing here is truncated: the full 2048x5632 gate/up and 5632x2048 down
// projections, the hidden size whole. The result is therefore this layer's REAL MLP output, and the
// blob carries MLP_TORCH -- what TinyLlama's own mlp module produced on the same tokens -- so this
// is gradeable against the model instead of against a reimplementation of a slice. It is the MLP's
// counterpart to what llama_attention_full.c did for attention.
//
//   host   xn = rmsnorm(h_mid, w_post_ln)                    fp32 -> MX
//   mesh   G = Xn @ Wg   [M,D]x[D,F]                         F-chunked, Xn resident
//   mesh   U = Xn @ Wu   [M,D]x[D,F]                         same A, same pass
//   host   H = silu(G) * U                                   fp32 -> MX
//   mesh   Y = H @ Wd    [M,F]x[F,D]                         K-TILED and N-chunked
//   host   out = h_mid + Y
//
// TWO BUDGETS SHAPE THIS, and at F = 5632 they bind in opposite directions.
//
// 1. THE SCRATCHPAD caps the OUTPUT. [M][F] as BF16 is M*F*2/DIM = 22528 rows against 16384
//    available, so gate/up cannot land whole however they are placed. They are chunked on F.
//
// 2. THE SCALE WINDOW caps the CONTRACTION. `ScaleFactorMem` holds 256 rows per double-buffer half
//    and one matmul needs N*K/512 of them on the B side, M*K/512 on the A side
//    (planning/rtl_fault_b_kdepth.md). down_proj contracts over K = 5632, which needs 704 B-side
//    rows at N = 64 and 352 A-side rows at M = 32 -- BOTH over. Past the ceiling the row address
//    truncates silently and the upper K-groups reuse lower rows: right sign, right magnitude,
//    5-60% wrong. So down_proj MUST be split into accumulating K-tiles. This is the first kernel
//    here forced into that at full scale; mxl5 proved the split bit-exact on RTL, and the A-side
//    term is asserted below because llama_mlp.c's bound only ever checked the B side (at M = 32 the
//    A side is the smaller of the two whenever N >= M, which held for every earlier shape).
//
// THE MATMUL COUNT IS INVARIANT. Both budgets reduce to a cap on the PRODUCT K*N per call
// (131072 here), so the schedule always issues ceil(total work / that) = 88 matmuls per projection
// however the tiles are shaped. What the shape actually decides is DMA, and that is why the plan
// below prefers the FEWEST K-tiles: one K-tile over the full D lets Xn be moved in ONCE and stay
// resident across all 176 projection matmuls, where a deeper split would re-mvin it per tile.
//
// TILING ADAPTS TO THE SCRATCHPAD, as in llama_attention_full.c: every width is chosen at COMPILE
// TIME from BANK_NUM * BANK_ROWS and the banner prints the plan it picked. The blob is independent
// of all of it -- an output column depends only on its own column of B, and mxl5 makes the K-split
// bit-exact -- so any schedule reproduces the same goldens.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "mx_host.h"
#include "llama_mlp_full.h"

#define DIM 16

// PIN THE GEOMETRY, and warn instead of silently following the shared header -- gemmini_params.h is
// hand-flipped per bitstream (DIM 16/BANK_ROWS 4096 <-> DIM 32/BANK_ROWS 2048) and a flip would
// quietly replan this kernel for a machine that is not the one being targeted.
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

#define OUT_BF16 3      // gemmini_extended3_config_ex out_mx_fmt: 3 = BF16, no requant
#define SPAD_STORE 0x38 // loop_ws skips: keep the full-width store into the internal scratchpad

#define SPAD_TOP (BANK_NUM * BANK_ROWS)
#define ROWS8(m, n)  ((m) * (n) / DIM)
#define ROWS16(m, n) ((m) * (n) * 2 / DIM)

#ifndef LLAMA_SCALE_ROWS_MAX
#define LLAMA_SCALE_ROWS_MAX 256
#endif
// Rows one matmul needs in each scale window. The B side is (N/16)*(K/32); the A side is the same
// with M in place of N. Both are checked -- see the header comment on why the A side matters here.
#define SCALE_ROWS(k, n) ((n) * (k) / 512)

// ---- phase 1: gate_proj and up_proj, [M,D] x [D,Fc] -> BF16 ---------------------------------
// Xn (A), the weight tile (B) and the output chunk must be resident together. PKTILE first and as
// large as possible: at PKTILE == D the contraction is one call and Xn is moved in ONCE for all
// 176 matmuls. FCHUNK is then the widest output the remaining space and the scale window allow.
#define P_COST(kt, fc) (ROWS8(LLAMA_M, (kt)) + ROWS8((kt), (fc)) + ROWS16(LLAMA_M, (fc)))
#define PFITS(kt, fc)  (P_COST((kt), (fc)) <= SPAD_TOP && \
                        SCALE_ROWS((kt), (fc)) <= LLAMA_SCALE_ROWS_MAX && \
                        SCALE_ROWS((kt), LLAMA_M) <= LLAMA_SCALE_ROWS_MAX)
#define PKTILE (PFITS(LLAMA_D, 16) ? LLAMA_D : \
                PFITS(1024, 16)    ? 1024    : \
                PFITS(512, 16)     ? 512     : \
                PFITS(256, 16)     ? 256     : 128)
#define PKTILES (LLAMA_D / PKTILE)
#define PKGRP   (PKTILE / 32)
#define FCHUNK (PFITS(PKTILE, 512) ? 512 : \
                PFITS(PKTILE, 256) ? 256 : \
                PFITS(PKTILE, 128) ? 128 : \
                PFITS(PKTILE, 64)  ? 64  : \
                PFITS(PKTILE, 32)  ? 32  : 16)
#define FCHUNKS (LLAMA_F / FCHUNK)

#define SPAD_XN     0
#define SPAD_WB     (SPAD_XN + ROWS8(LLAMA_M, PKTILE))
#define SPAD_WB_ARG (SPAD_WB + ROWS8(PKTILE, FCHUNK))
#define SPAD_GU     SPAD_WB_ARG

// ---- phase 2: down_proj, [M,F] x [F,Dc] -> BF16, K-tiled -------------------------------------
// A K-tile slice of H (A), the weight tile (B) and the output chunk coexist; the chunk stays put
// while every K-tile accumulates into it, the first overwriting and the rest adding. NCHUNK is
// picked against the SHALLOWEST K-tile so the widest output is reachable, then KTILE is the
// deepest contraction that width's scale window still permits.
#define Y_COST(kt, nc) (ROWS8(LLAMA_M, (kt)) + ROWS8((kt), (nc)) + ROWS16(LLAMA_M, (nc)))
#define YFITS(kt, nc)  (Y_COST((kt), (nc)) <= SPAD_TOP && \
                        SCALE_ROWS((kt), (nc)) <= LLAMA_SCALE_ROWS_MAX && \
                        SCALE_ROWS((kt), LLAMA_M) <= LLAMA_SCALE_ROWS_MAX)
#define NCHUNK (YFITS(128, 1024) ? 1024 : \
                YFITS(128, 512)  ? 512  : \
                YFITS(128, 256)  ? 256  : \
                YFITS(128, 128)  ? 128  : 64)
#define NCHUNKS (LLAMA_D / NCHUNK)
// F = 5632 = 2^9 * 11, so the K-tile candidates that DIVIDE it are the powers of two up to 512
// (1024 does not: 5632/1024 = 5.5). Offering 1024 here would elaborate a tiling that drops the
// last partial tile and silently truncate the reduction, so it is deliberately absent.
#define KTILE (YFITS(512, NCHUNK) && (LLAMA_F % 512) == 0 ? 512 : \
               YFITS(256, NCHUNK) && (LLAMA_F % 256) == 0 ? 256 : \
               YFITS(128, NCHUNK) && (LLAMA_F % 128) == 0 ? 128 : \
               YFITS(64,  NCHUNK) && (LLAMA_F % 64)  == 0 ? 64  : 32)
#define KTILES (LLAMA_F / KTILE)
#define KGRP   (KTILE / 32)

#define SPAD_H      0
#define SPAD_WD     (SPAD_H + ROWS8(LLAMA_M, KTILE))
#define SPAD_WD_ARG (SPAD_WD + ROWS8(KTILE, NCHUNK))
#define SPAD_Y      SPAD_WD_ARG

// Checked at COMPILE time: a region that runs off the end, or a scale window that overflows,
// aliases silently and yields plausible-but-wrong numbers rather than an error. Fault B was
// invisible for exactly as long as nothing asserted the scale bound.
#define LLAMA_REQUIRE(name, cond) typedef char llama_plan_##name[(cond) ? 1 : -1]
LLAMA_REQUIRE(pktile_divides_d,  PKTILE * PKTILES == LLAMA_D);
LLAMA_REQUIRE(pktile_is_blocked, (PKTILE % 32) == 0);
LLAMA_REQUIRE(fchunk_divides_f,  FCHUNK * FCHUNKS == LLAMA_F);
LLAMA_REQUIRE(proj_fits,         SPAD_GU + ROWS16(LLAMA_M, FCHUNK) <= SPAD_TOP);
LLAMA_REQUIRE(ktile_divides_f,   KTILE * KTILES == LLAMA_F);
LLAMA_REQUIRE(ktile_is_blocked,  (KTILE % 32) == 0);
LLAMA_REQUIRE(nchunk_divides_d,  NCHUNK * NCHUNKS == LLAMA_D);
LLAMA_REQUIRE(down_fits,         SPAD_Y + ROWS16(LLAMA_M, NCHUNK) <= SPAD_TOP);
LLAMA_REQUIRE(proj_scale_fits,   SCALE_ROWS(PKTILE, FCHUNK) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(proj_ascale_fits,  SCALE_ROWS(PKTILE, LLAMA_M) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(down_scale_fits,   SCALE_ROWS(KTILE, NCHUNK) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_REQUIRE(down_ascale_fits,  SCALE_ROWS(KTILE, LLAMA_M) <= LLAMA_SCALE_ROWS_MAX);

// ---- host buffers ----
static float    xn_f[LLAMA_M * LLAMA_D];
static uint8_t  xn_codes[LLAMA_M * LLAMA_D];
static uint8_t  xn_scales[LLAMA_GD * LLAMA_M];
static float    h_f[LLAMA_M * LLAMA_F];
static uint8_t  h_codes[LLAMA_M * LLAMA_F];
static uint8_t  h_scales[LLAMA_GF * LLAMA_M];
static uint16_t G_hw[LLAMA_M * LLAMA_F];
static uint16_t U_hw[LLAMA_M * LLAMA_F];
static uint16_t chunk16[LLAMA_M * (FCHUNK > NCHUNK ? FCHUNK : NCHUNK)];
// A B-side scale window is [K/32][N] with b_off = group * N + col, so slicing N columns out of the
// [G][N_full] table is a GATHER -- the rows are N_full apart, not contiguous.
static uint8_t  proj_scales_chunk[PKGRP * FCHUNK];
static uint8_t  wd_scales_chunk[KGRP * NCHUNK];
static uint16_t Y_hw[LLAMA_M * LLAMA_D];
static uint16_t OUT_hw[LLAMA_M * LLAMA_D];
static uint32_t scale_sink[512] __attribute__((aligned(32)));

uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  printf("TRAP cause=%d epc=%lx\n", (int) cause, (unsigned long) epc);
  tohost_exit(1337);
  return 0;
}

// ---- movers ----------------------------------------------------------------------------------
// mvin A[M][K] as tiles: tile (i,k) -> a_spad + (i*tiles_K + k)*DIM. `stride` is the SOURCE row
// pitch, which differs from K when the tile is a column slice of a wider array -- that is how a
// K-tile of H[M][F] is moved in without copying it out first.
static void mvin_A_strided(const uint8_t *A, int M, int K, int stride, uint32_t a_spad) {
  gemmini_config_ld(stride * sizeof(uint8_t));
  int tiles_I = M / DIM, tiles_K = K / DIM;
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void *) (A + (size_t) i * DIM * stride + (size_t) k * DIM),
                            a_spad + (i * tiles_K + k) * DIM, DIM, DIM);
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

// `accum` is loop_ws's ex_accumulate (rs1 bit 0): 0 OVERWRITES the output region, 1 adds into it.
// Only a K-tile after the first wants 1.
static void mesh_matmul(int M, int K, int N, uint32_t a_spad, uint32_t b_arg, uint32_t c_spad,
                        int accum) {
  int I = M / DIM, J = N / DIM, Kt = K / DIM;
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              0, 0, OUT_BF16, 0);
  gemmini_config_st(N * sizeof(uint16_t));
  gemmini_mxquant_config_mvout((uint64_t) scale_sink, I, J, Kt, 0, 0, 1);
  gemmini_loop_ws_spad(I, J, Kt, 0, 0, 0, a_spad, b_arg, 0, c_spad,
                       false, false, false, false, accum, NO_ACTIVATION, 0, 0, false,
                       SPAD_STORE);
  gemmini_fence();
}

// One [M,D] x [D,F] projection, chunked on F and K-tiled over D. `dst` receives the whole [M][F].
static void projection(const uint8_t *w_codes, const uint8_t *w_scales, uint16_t *dst) {
  for (int c = 0; c < FCHUNKS; c++) {
    for (int t = 0; t < PKTILES; t++) {
      // The B window wants this chunk's columns packed contiguously per K-group.
      for (int g = 0; g < PKGRP; g++)
        memcpy(proj_scales_chunk + (size_t) g * FCHUNK,
               w_scales + (size_t) (t * PKGRP + g) * LLAMA_F + (size_t) c * FCHUNK, FCHUNK);
#if PKTILES > 1
      // Only a split contraction has to re-establish A; at one K-tile Xn is already resident and
      // stays so across every chunk and both projections.
      mvin_A_strided(xn_codes + (size_t) t * PKTILE, LLAMA_M, PKTILE, LLAMA_D, SPAD_XN);
#endif
      gemmini_mx_load_scales((uint64_t) (xn_scales + (size_t) t * PKGRP * LLAMA_M),
                             PKGRP * LLAMA_M, 0);
      gemmini_mx_load_scales((uint64_t) proj_scales_chunk, PKGRP * FCHUNK, 1);
      gemmini_fence();
      mvin_B(w_codes + (size_t) t * PKTILE * LLAMA_F, PKTILE, LLAMA_F, c * FCHUNK, FCHUNK,
             SPAD_WB);
      mesh_matmul(LLAMA_M, PKTILE, FCHUNK, SPAD_XN, SPAD_WB_ARG, SPAD_GU, t > 0);
    }
    mvout_bf16(chunk16, SPAD_GU, LLAMA_M, FCHUNK);
    for (int m = 0; m < LLAMA_M; m++)
      memcpy(&dst[(size_t) m * LLAMA_F + (size_t) c * FCHUNK], &chunk16[(size_t) m * FCHUNK],
             FCHUNK * sizeof(uint16_t));
  }
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  const uint16_t *H_MID     = LLAMA_AT(LLAMA_OFF_H_MID, uint16_t);
  const uint16_t *W_POST_LN = LLAMA_AT(LLAMA_OFF_W_POST_LN, uint16_t);
  const uint8_t  *XN_CODES  = LLAMA_AT(LLAMA_OFF_XN_CODES, uint8_t);
  const uint8_t  *XN_SCALES = LLAMA_AT(LLAMA_OFF_XN_SCALES, uint8_t);
  const uint8_t  *WG_CODES  = LLAMA_AT(LLAMA_OFF_WG_CODES, uint8_t);
  const uint8_t  *WG_SCALES = LLAMA_AT(LLAMA_OFF_WG_SCALES, uint8_t);
  const uint8_t  *WU_CODES  = LLAMA_AT(LLAMA_OFF_WU_CODES, uint8_t);
  const uint8_t  *WU_SCALES = LLAMA_AT(LLAMA_OFF_WU_SCALES, uint8_t);
  const uint8_t  *WD_CODES  = LLAMA_AT(LLAMA_OFF_WD_CODES, uint8_t);
  const uint8_t  *WD_SCALES = LLAMA_AT(LLAMA_OFF_WD_SCALES, uint8_t);
  const uint16_t *G_OUT     = LLAMA_AT(LLAMA_OFF_G_OUT, uint16_t);
  const uint16_t *U_OUT     = LLAMA_AT(LLAMA_OFF_U_OUT, uint16_t);
  const uint8_t  *H_CODES_G = LLAMA_AT(LLAMA_OFF_H_CODES, uint8_t);
  const uint8_t  *H_SCALES_G= LLAMA_AT(LLAMA_OFF_H_SCALES, uint8_t);
  const uint16_t *Y_OUT     = LLAMA_AT(LLAMA_OFF_Y_OUT, uint16_t);
  const uint16_t *REF_MLP   = LLAMA_AT(LLAMA_OFF_REF_MLP, uint16_t);
  const uint16_t *MLP_TORCH = LLAMA_AT(LLAMA_OFF_MLP_TORCH, uint16_t);

  printf("llama MLP FULL: M=%d D=%d F=%d (ALL neurons, fp8 e4m3 + E8M0)\n",
         LLAMA_M, LLAMA_D, LLAMA_F);
  printf("plan  spad %d rows: proj %d chunk(s) of %d x %d K-tile(s) of %d | "
         "down %d chunk(s) of %d x %d K-tile(s) of %d\n",
         SPAD_TOP, FCHUNKS, FCHUNK, PKTILES, PKTILE, NCHUNKS, NCHUNK, KTILES, KTILE);

  gemmini_flush(0);

  uint64_t t_host = 0, t_mesh = 0, t0;

  // ============ host: RMSNorm over the full D ============
  t0 = read_cycles();
  mx_rmsnorm(H_MID, W_POST_LN, LLAMA_M, LLAMA_D, LLAMA_RMS_EPS, xn_f);
  mx_quantize_rows(xn_f, LLAMA_M, LLAMA_D, xn_codes, xn_scales);
  t_host += read_cycles() - t0;

  int xn_cd = mx_count_diff_u8(xn_codes, XN_CODES, LLAMA_M * LLAMA_D);
  int xn_sd = mx_count_diff_u8(xn_scales, XN_SCALES, LLAMA_GD * LLAMA_M);
  printf("host  rmsnorm+quant: codes differ %d/%d, scales differ %d/%d vs golden\n",
         xn_cd, LLAMA_M * LLAMA_D, xn_sd, LLAMA_GD * LLAMA_M);

  // ============ mesh: gate_proj and up_proj, Xn resident across both ============
  t0 = read_cycles();
#if PKTILES == 1
  mvin_A_strided(xn_codes, LLAMA_M, LLAMA_D, LLAMA_D, SPAD_XN);
  gemmini_fence();
#endif
  projection(WG_CODES, WG_SCALES, G_hw);
  projection(WU_CODES, WU_SCALES, U_hw);
  t_mesh += read_cycles() - t0;

  int g_d = mx_count_diff_u16(G_hw, G_OUT, LLAMA_M * LLAMA_F);
  int u_d = mx_count_diff_u16(U_hw, U_OUT, LLAMA_M * LLAMA_F);
  printf("mesh  G = Xn @ Wg : %d/%d differ from golden (%d chunks of %d)\n",
         g_d, LLAMA_M * LLAMA_F, FCHUNKS, FCHUNK);
  printf("mesh  U = Xn @ Wu : %d/%d differ from golden\n", u_d, LLAMA_M * LLAMA_F);

  // ============ host: SwiGLU ============
  t0 = read_cycles();
  mx_swiglu(G_hw, U_hw, LLAMA_M * LLAMA_F, h_f);
  mx_quantize_rows(h_f, LLAMA_M, LLAMA_F, h_codes, h_scales);
  t_host += read_cycles() - t0;

  int h_cd = mx_count_diff_u8(h_codes, H_CODES_G, LLAMA_M * LLAMA_F);
  int h_sd = mx_count_diff_u8(h_scales, H_SCALES_G, LLAMA_GF * LLAMA_M);
  printf("host  silu(G)*U + quant: codes differ %d/%d, scales differ %d/%d vs golden\n",
         h_cd, LLAMA_M * LLAMA_F, h_sd, LLAMA_GF * LLAMA_M);

  // ============ mesh: down_proj, K-TILED and N-chunked ============
  // Y_c = sum_t H[:, t-slice] @ Wd[t-slice, c-slice]. Every K-tile targets the same C region and
  // mx_smem adds, so the 5632-deep reduction is reached in KTILES accumulating calls -- the only
  // way to stay under the scale window's 256 rows.
  for (int c = 0; c < NCHUNKS; c++) {
    for (int t = 0; t < KTILES; t++) {
      for (int g = 0; g < KGRP; g++)
        memcpy(wd_scales_chunk + (size_t) g * NCHUNK,
               WD_SCALES + (size_t) (t * KGRP + g) * LLAMA_D + (size_t) c * NCHUNK, NCHUNK);
      t0 = read_cycles();
      // The A window is [GF][M], so a K-tile is a CONTIGUOUS row range of it -- no gather, unlike
      // the B side above.
      mvin_A_strided(h_codes + (size_t) t * KTILE, LLAMA_M, KTILE, LLAMA_F, SPAD_H);
      gemmini_mx_load_scales((uint64_t) (h_scales + (size_t) t * KGRP * LLAMA_M),
                             KGRP * LLAMA_M, 0);
      gemmini_mx_load_scales((uint64_t) wd_scales_chunk, KGRP * NCHUNK, 1);
      gemmini_fence();
      mvin_B(WD_CODES + (size_t) t * KTILE * LLAMA_D, KTILE, LLAMA_D, c * NCHUNK, NCHUNK, SPAD_WD);
      mesh_matmul(LLAMA_M, KTILE, NCHUNK, SPAD_H, SPAD_WD_ARG, SPAD_Y, t > 0);
      t_mesh += read_cycles() - t0;
    }
    mvout_bf16(chunk16, SPAD_Y, LLAMA_M, NCHUNK);
    for (int m = 0; m < LLAMA_M; m++)
      memcpy(&Y_hw[(size_t) m * LLAMA_D + (size_t) c * NCHUNK], &chunk16[(size_t) m * NCHUNK],
             NCHUNK * sizeof(uint16_t));
  }

  int y_d = mx_count_diff_u16(Y_hw, Y_OUT, LLAMA_M * LLAMA_D);
  printf("mesh  Y = H @ Wd  : %d/%d differ from golden (%d chunks of %d, %d K-tiles of %d)\n",
         y_d, LLAMA_M * LLAMA_D, NCHUNKS, NCHUNK, KTILES, KTILE);

  // ============ host: the residual ============
  t0 = read_cycles();
  for (int i = 0; i < LLAMA_M * LLAMA_D; i++)
    OUT_hw[i] = mx_f32_to_bf16_rne(mx_bf16_to_f32(H_MID[i]) + mx_bf16_to_f32(Y_hw[i]));
  t_host += read_cycles() - t0;

  printf("grade MLP out vs fp32 reference      : rel_fro %d ppm\n",
         MX_PPM(mx_rel_fro_bf16(Y_hw, REF_MLP, LLAMA_M * LLAMA_D)));
  printf("grade MLP out vs THE MODEL's own out : rel_fro %d ppm\n",
         MX_PPM(mx_rel_fro_bf16(Y_hw, MLP_TORCH, LLAMA_M * LLAMA_D)));
  printf("cycles mesh %d, host %d\n", (int) t_mesh, (int) t_host);

  int exact = g_d + u_d + y_d;
  int host_drift = xn_cd + xn_sd + h_cd + h_sd;
  if (exact == 0 && host_drift == 0)
    printf("llama FULL MLP test PASSED (all %d neurons, every mesh stage bit-exact; "
           "down_proj K-tiled %d ways).\n", LLAMA_F, KTILES);
  else if (exact == 0)
    printf("llama FULL MLP test PASSED WITH DRIFT: mesh stages bit-exact, but %d host byte(s) "
           "differ from the golden.\n", host_drift);
  else
    printf("llama FULL MLP test FAILED: %d mesh element(s) differ from golden (%d host byte(s) "
           "differed too).\n", exact, host_drift);

#ifndef BAREMETAL
  exit(exact != 0);
#else
  return exact != 0;
#endif
}
