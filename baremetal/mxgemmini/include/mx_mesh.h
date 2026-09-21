// The MX mesh calling convention, in one place: operand mvin, output drain, and one matmul.
//
// These bodies are lifted VERBATIM from `src/llama_attention.c`, which is the point. The MX
// bisection ladder (`src/mx_ladder.c`) exists to answer "does the llama kernel fail on RTL because
// of its SHAPES or because of its instruction stream?", and that question is only meaningful if
// the ladder issues the same stream. A second copy of these helpers would let the two drift and
// make rung mxl0 a tautology.
//
// `llama_attention.c` and `llama_mlp.c` still carry their own copies; switching them over to this
// header is a separate change, deliberately not bundled with a debugging artifact.
//
// The caller must define DIM before including this (the ladder pins it to 16; see mx_ladder.c on
// why it does not simply trust gemmini_params.h).
#ifndef INCLUDE_MX_MESH_H
#define INCLUDE_MX_MESH_H

#include <stdint.h>
#include "include/gemmini_testutils.h"

#ifndef DIM
#error "mx_mesh.h needs DIM defined by the caller"
#endif

// `out_mx_fmt`, arg 12 of gemmini_extended3_config_ex (gemmini.h:301): which drain the mesh uses.
#define OUT_BF16 3   //: non-requant BF16, flat row-major in the internal spad
#define OUT_FP8  0   //: FP8 requant, plus the requantizer's E8M0 codes to the scale address

//: loop_ws rs2 low bits: store the mesh output to the internal scratchpad.
#define SPAD_STORE 0x38
//: LOOP_WS rs2 bit10: deposit the requant output in the block-tiled operand-A layout instead of
//: flat row-major, so it can be re-read in place as the next matmul's A.
#define LOOP_WS_REQUANT_TILED (1u << 10)

#define SPAD_TOP (BANK_NUM * BANK_ROWS)

// Scratchpad rows an [m][n] operand occupies: one DIM-byte row per DIM elements. fp8 codes are one
// byte per element, a BF16 output two.
#define ROWS8(m, n)  ((m) * (n) / DIM)
#define ROWS16(m, n) ((m) * (n) * 2 / DIM)

// `stride` is the SOURCE row pitch, which differs from K when the tile is a column slice of a
// wider array -- that is how a K-tile of a [M][D] activation moves in without being copied out.
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

// B tiles go in K-MAJOR order, `(k * tiles_J + j) * DIM`, because that is where loop_ws looks for
// them (`B_sp_addr_start + (k*J + j)*DIM`, gemmini.cc:808 and :1291). `n0` selects a column slice
// of a wider `N_full` array; slicing N is a GATHER, the rows being N_full apart.
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

// Read a BLOCK-TILED fp8 tile back to a flat [M][N] buffer, via a caller-supplied `scratch` of at
// least M*N bytes. Contiguous mvout into the scratch, then a SOFTWARE de-tile: a strided de-tiling
// mvout makes the writer DMA emit whole cache lines and zero-fill the gaps on RTL
// (matmul_tiled_fp8_64x64_chain.c:79-83). Read-only -- the tile stays resident.
static void mvout_detile(uint8_t *dst, uint8_t *scratch, uint32_t spad, int M, int N) {
  int tiles_I = M / DIM, tiles_N = N / DIM, total_rows = M * N / DIM;
  gemmini_config_st(DIM * sizeof(uint8_t));
  for (int r = 0; r < total_rows; r += DIM)
    gemmini_extended_mvout(scratch + (size_t) r * DIM, spad + r, DIM, DIM);
  gemmini_fence();
  for (int i = 0; i < tiles_I; i++)
    for (int nt = 0; nt < tiles_N; nt++)
      for (int r = 0; r < DIM; r++)
        for (int c = 0; c < DIM; c++)
          dst[(size_t) (i * DIM + r) * N + nt * DIM + c] =
              scratch[(size_t) ((i * tiles_N + nt) * DIM + r) * DIM + c];
}

// One mesh matmul. `out_fmt` picks BF16 (drained by the host) or FP8 requant; `resident` routes the
// requantizer's block scales into the act-scale window as well as to DRAM, and FP8 output is
// deposited in the operand-A layout -- together, the next matmul's A operand, in place.
// `accum` is loop_ws's ex_accumulate (rs1 bit 0): 0 OVERWRITES the output region, 1 adds into it.
static void mesh_matmul(int M, int K, int N, uint32_t a_spad, uint32_t b_arg, uint32_t c_spad,
                        int out_fmt, uint64_t scale_dram, int resident, int accum) {
  int I = M / DIM, J = N / DIM, Kt = K / DIM;
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              0, 0, out_fmt, 0);
  gemmini_config_st((out_fmt == OUT_BF16 ? N * (int) sizeof(uint16_t) : (int) sizeof(uint16_t)));
  // Braced: the gemmini_* macros expand to a `{ ... }` block, not a do-while, so an unbraced
  // if/else arm swallows the `else`.
  if (resident) {
    gemmini_mxquant_config_mvout_resident(scale_dram, I, J, Kt, 0, 0, 1);
  } else {
    gemmini_mxquant_config_mvout(scale_dram, I, J, Kt, 0, 0, 1);
  }
  gemmini_loop_ws_spad(I, J, Kt,
                       0, 0, 0,
                       a_spad,
                       b_arg,
                       0,
                       c_spad,
                       false, false,
                       false, false, accum,
                       NO_ACTIVATION,
                       0, 0,
                       false,
                       SPAD_STORE | (out_fmt == OUT_FP8 ? LOOP_WS_REQUANT_TILED : 0));
  gemmini_fence();
}

#endif  // INCLUDE_MX_MESH_H
