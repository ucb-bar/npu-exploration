// A REAL TinyLlama attention head, back to back, on MxGemmini -- one ELF, real data, llama dims.
//
// Six matmuls on the mesh with the host's fp32 glue in the seams the mesh cannot cross. Operands
// are a real decoder layer captured from a real forward pass (kernels/captures/llama_layer.py ->
// gen_llama_layer.py -> include/llama_attn.h): hidden size D = 2048 kept FULL, one query head and
// its GQA kv head, M real tokens of wikitext2.
//
//   host   xn = rmsnorm(h_pre, w_in_ln)                     fp32 -> MX
//   mesh   Q,K,V = Xn @ Wq/Wk/Wv   [M,D]x[D,H] -> bf16      Xn resident for all three
//   host   RoPE on Q and K; K transposed                    fp32 -> MX
//   mesh   S = Q @ K^T             [M,H]x[H,M] -> bf16
//   host   S/sqrt(H), causal mask, softmax                  fp32 -> MX
//   mesh   O = P @ V               [M,M]x[M,H] -> FP8 REQUANT, tiled, INTO the scratchpad
//   mesh   Y = O @ Wo              [M,H]x[H,D] -> bf16      O + its scales read IN PLACE
//   host   out = h_pre + Y
//
// THE RESIDENT SEAM. `P@V` and `O@Wo` are the only two matmuls in a llama layer with no host op
// between them, so this is where the chain path earns its place: the requantizer writes O into the
// scratchpad in the operand-A TILED layout (LOOP_WS_REQUANT_TILED) and its E8M0 codes straight into
// the act-scale window (gemmini_mxquant_config_mvout_resident), and o_proj then reads both where
// they already are -- no mvout, no host transpose, no scale reload. Every other seam here carries a
// host op (RoPE, softmax) and so cannot chain: the values have to reach the scalar core.
//
// WHY K IS TRANSPOSED ON THE HOST. gemmini_loop_ws_spad's signature carries A_transpose/B_transpose,
// but the MX path ignores them -- mx_loop_ws_spad (gemmini.cc:1144) does `(void)rs1` and never reads
// the bits; only the stock int8 loop_ws (:711) honours them. The scale half is free: K's per-row
// scales [M][H/32] transpose into exactly the [H/32][M] the B side indexes.
//
// SCRATCHPAD BUDGET. 16384 rows x 16 B hold A, B and the output together, and `mx_smem` accumulates
// and is never cleared (gemmini.cc:1190), so every matmul needs a DISJOINT output region. The map is
// DERIVED from LLAMA_M/D/H below rather than written out, because the same source builds against
// headers of different D (see LLAMA_ATTN_HEADER); baked addresses would silently overlap. At the
// full D = 2048 it lays out as:
//
//   Xn A     0..4095       Wq/Wk/Wv B  8192..16383  (B arg 16384)
//   Q smem   4096..4351    K smem      4352..4607    V smem  4608..4863
//   Q A      0..127        K^T B       4864..4991   (B arg 4992)    S smem  4992..5119
//   P A      0..63         V B         5120..5247   (B arg 5248)
//   O        5248..5375    <- stays LIVE through o_proj; it is that matmul's A operand
//   Wo B     0..4095      (B arg 4096, on the dead Xn rows)   Y0 5376..9471   Y1 9472..13567
//
// Unified real-RoCC stream for Spike (-DSPIKE_SIM) and the standalone RTL (-DMX_ROCKET).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "mx_host.h"

// WHICH CAPTURE THIS ELF CARRIES. The default is the full D = 2048 layer. A sliced-D header
// (gen_llama_layer.py --d 256) is what makes an RTL run affordable: the host's fp32 glue is 99.9% of
// the cycles (planning/llama_layer_hw_plan.md 8.3) and scales linearly in D, so D = 256 is ~8x less
// simulation for a chain with every structural feature of the full one intact. Nothing below reads D
// as a constant, so the only difference between the two ELFs is this line.
#ifndef LLAMA_ATTN_HEADER
#define LLAMA_ATTN_HEADER "llama_attn.h"
#endif
#include LLAMA_ATTN_HEADER

#define DIM 16

// PIN THE GEOMETRY, and warn instead of silently following the shared header.
//
// `DIM` is overridden above, which is why a dim32 `gemmini_params.h` never broke these kernels --
// but `BANK_ROWS` was not, and it flips with whatever bitstream is being built (4096 for dim16,
// 2048 for dim32). Section 9.3 makes this kernel deliberately ADAPTIVE to the scratchpad size, so
// a flip does not produce wrong numbers -- it quietly retunes the K-tiling and the output chunking
// for a machine that is not the one being targeted, and the ELF looks fine while planning against
// half the memory it has. Pin it, say so at compile time, and keep the adaptivity for anyone who
// overrides it: `-DLLAMA_BANK_ROWS=2048` reproduces the small-config schedule exactly.
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
// LOOP_WS rs2 bit10: deposit the requant output in the block-tiled operand-A layout instead of flat
// row-major, so it can be re-read in place as the next matmul's A. Same flag the chain tests use.
#define LOOP_WS_REQUANT_TILED (1u << 10)

#define SPAD_TOP  (BANK_NUM * BANK_ROWS)

// Scratchpad rows an [m][n] operand occupies: one 16-byte row per DIM elements. fp8 codes are one
// byte per element, a BF16 output two.
#define ROWS8(m, n)  ((m) * (n) / DIM)
#define ROWS16(m, n) ((m) * (n) * 2 / DIM)

// ---- phase 1: Q, K, V = Xn @ Wq/Wk/Wv, [M,D] x [D,H] -> BF16 --------------------------------
// Xn (A) and the weight tile (B) must be resident together; at D = 2048 that is 4096 + 8192 rows,
// more than a 128 KB scratchpad holds at all, so no placement fixes it. The D-deep contraction is
// SPLIT into K-tiles that accumulate into one output region (ex_accumulate = 0 on the first, 1
// after) -- which only became usable once the MX path stopped discarding rs1 (see
// planning/llama_layer_hw_plan.md 10.3). At 16384 rows the largest tile is all of D and the loop
// runs once, so the big-config schedule is unchanged.
//
// THE SCALE WINDOW IS A SECOND BUDGET, and on a deep contraction it binds before the scratchpad
// does. `ScaleFactorMem` holds 256 rows x 16 B per double-buffer half, and one matmul needs
// (N/16) * (K/32) = N*K/512 of them. Past that the row address TRUNCATES and the upper K-groups
// silently reuse lower rows: right sign, right magnitude, 5-60% wrong. That is Fault B
// (planning/rtl_fault_b_kdepth.md), which is what made this kernel fail on hardware while passing
// on spike. The RTL does not signal the overflow, so the kernel must not emit one.
//
// This REPLACES the `LLAMA_KTILE_MAX = 1024` workaround of 2026-09-20. That cap was expressed in
// K, which is only correct while N <= 64 -- deriving the bound from N*K instead means it LIFTED
// ITSELF when the RTL went from 128 usable rows to 256 (D = 2048 at H = 64 needs exactly 256, so
// the projection is one K-tile again), and it still protects a future kernel that widens N.
#ifndef LLAMA_SCALE_ROWS_MAX
#define LLAMA_SCALE_ROWS_MAX 256
#endif
#define SCALE_ROWS(k, n) ((n) * (k) / 512)
#define P_COST(kt) (ROWS8(LLAMA_M, (kt)) + ROWS8((kt), LLAMA_H) + ROWS16(LLAMA_M, LLAMA_H))
#define KFITS(kt)  (P_COST(kt) <= SPAD_TOP && \
                    SCALE_ROWS((kt), LLAMA_H) <= LLAMA_SCALE_ROWS_MAX)
#define KTILE  (KFITS(LLAMA_D) ? LLAMA_D : \
                KFITS(1024)    ? 1024    : \
                KFITS(512)     ? 512     : \
                KFITS(256)     ? 256     : 128)
#define KTILES (LLAMA_D / KTILE)
#define KGRP   (KTILE / 32)                  // E8M0 scale groups spanned by one K-tile

// A `_ARG` address is the row just PAST its B tile: that is what gemmini_loop_ws_spad takes as its
// B argument, since the loop walks the B tiles backwards from there.
#define SPAD_XN      0
#define SPAD_PB      (SPAD_XN + ROWS8(LLAMA_M, KTILE))
#define SPAD_PB_ARG  (SPAD_PB + ROWS8(KTILE, LLAMA_H))
#define SPAD_QKV     SPAD_PB_ARG             // one region: Q, then K, then V, each drained

// ---- phases 2-4: scores, softmax product, o_proj --------------------------------------------
// O is the one value that stays LIVE across a later matmul -- o_proj reads it in place as its A
// operand, with the E8M0 bytes the requantizer wrote into the act-scale window. So O sits at row 0
// and everything else is placed above it; the S/P/V working regions are dead by the time o_proj
// starts, which is why SPAD_WO may reuse their rows.
#define SPAD_O       0                                       // LIVE through o_proj
#define SPAD_QA      (SPAD_O  + ROWS8(LLAMA_M, LLAMA_H))
#define SPAD_KT      (SPAD_QA + ROWS8(LLAMA_M, LLAMA_H))
#define SPAD_KT_ARG  (SPAD_KT + ROWS8(LLAMA_H, LLAMA_M))
#define SPAD_S       SPAD_KT_ARG
#define SPAD_PA      (SPAD_S  + ROWS16(LLAMA_M, LLAMA_M))
#define SPAD_VB      (SPAD_PA + ROWS8(LLAMA_M, LLAMA_M))
#define SPAD_VB_ARG  (SPAD_VB + ROWS8(LLAMA_M, LLAMA_H))

// o_proj is N-chunked, one output region reused: each chunk overwrites and is drained immediately.
#define Y_COST(dc) (ROWS8(LLAMA_M, LLAMA_H) + ROWS8(LLAMA_H, (dc)) + ROWS16(LLAMA_M, (dc)))
#define YFITS(dc) (Y_COST(dc) <= SPAD_TOP && \
                   SCALE_ROWS(LLAMA_H, (dc)) <= LLAMA_SCALE_ROWS_MAX)
#define NCHUNK (YFITS(LLAMA_D) ? LLAMA_D : \
                YFITS(1024)    ? 1024    : \
                YFITS(512)     ? 512     : \
                YFITS(256)     ? 256     : 128)
#define NCHUNKS (LLAMA_D / NCHUNK)
#define SPAD_WO      (SPAD_O + ROWS8(LLAMA_M, LLAMA_H))      // on the dead S/P/V rows
#define SPAD_WO_ARG  (SPAD_WO + ROWS8(LLAMA_H, NCHUNK))
#define SPAD_Y       SPAD_WO_ARG

// Checked at COMPILE time, because an overflowing region aliases silently and produces plausible
// numbers. Each names a live range, not just a bound.
#define LLAMA_SPAD_REQUIRE(name, cond) typedef char llama_spad_##name[(cond) ? 1 : -1]
LLAMA_SPAD_REQUIRE(ktile_divides_d,  KTILE * KTILES == LLAMA_D);
LLAMA_SPAD_REQUIRE(ktile_is_blocked, (KTILE % 32) == 0);
LLAMA_SPAD_REQUIRE(proj_fits,        SPAD_QKV + ROWS16(LLAMA_M, LLAMA_H) <= SPAD_TOP);
LLAMA_SPAD_REQUIRE(attn_fits,        SPAD_VB_ARG <= SPAD_TOP);
LLAMA_SPAD_REQUIRE(nchunk_divides_d, NCHUNK * NCHUNKS == LLAMA_D);
LLAMA_SPAD_REQUIRE(oproj_fits,       SPAD_Y + ROWS16(LLAMA_M, NCHUNK) <= SPAD_TOP);
// The scale-window bound, for every matmul in the kernel. The two adaptive ones pick a tiling that
// satisfies it; these two have fixed shapes, so a check is the only thing that would catch a future
// M or H that breaks them. Fault B was invisible precisely because nothing asserted this.
LLAMA_SPAD_REQUIRE(scores_scale_fits, SCALE_ROWS(LLAMA_H, LLAMA_M) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_SPAD_REQUIRE(pv_scale_fits,     SCALE_ROWS(LLAMA_M, LLAMA_H) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_SPAD_REQUIRE(proj_scale_fits,   SCALE_ROWS(KTILE, LLAMA_H) <= LLAMA_SCALE_ROWS_MAX);
LLAMA_SPAD_REQUIRE(oproj_scale_fits,  SCALE_ROWS(LLAMA_H, NCHUNK) <= LLAMA_SCALE_ROWS_MAX);

#ifdef ATTN_NATIVE
// ---- native schedule (-DATTN_NATIVE) ----
// Q/K/V and S are native DRAM loops (gemmini_loop_ws_mx: operands streamed, BF16 C straight to DRAM,
// loop-managed scales); Q/K/V chain with no fence (K, V pass A = NULL: Xn resident; V's A scales are
// reused in place by the loop unit). P@V is unchanged -- the RESIDENT producer (FP8 requant -> spad,
// tiled; E8M0 -> act-scale window). o_proj is a native loop CONSUMING that residency: A = NULL reads
// O at spad row 0 (= a_spad_id 1's A region) and the act-scale window as written; only Wo's scales
// are loaded (2-D, weight half 0), under a waiting CONFIG_SCALE_MEM (act half 0) -- no fence.
// v1 loop limits: full K per loop, each operand in one spad half, scales in one 4 KB half, C in one
// acc half. Q/K/V: if A + B overflow a half (D = 2048), A takes half 0 and every B half 1 (the RS orders
// each B's mvins behind the previous loop's reads). o_proj is N-chunked (NAT_NC): chunk c loads its Wo
// scale slice into weight half c&1 under a waiting config. Chunk c+2 reuses chunk c's half with no HW
// ordering; it is safe because a chunk issues 2*I*J*K ex commands >> the 16 RS ex slots, so its loads
// can only enter once chunk c's computes have retired (checked below).
#define HALF_ROWS (SPAD_TOP / 2)
#define NAT_ACC_ROWS(m, n) ((m) * (n) / (DIM * 4))   // E4M3-single: 4 acc rows per 16x16 C tile
#define NAT_QKV_SPLIT (ROWS8(LLAMA_M, LLAMA_D) + ROWS8(LLAMA_D, LLAMA_H) > HALF_ROWS)
#define NAT_YFITS(nc) (ROWS8(LLAMA_M, LLAMA_H) + ROWS8(LLAMA_H, (nc)) <= HALF_ROWS && \
                       NAT_ACC_ROWS(LLAMA_M, (nc)) <= ACC_ROWS / 2 && LLAMA_GH * (nc) <= 4096)
#define NAT_NC (NAT_YFITS(LLAMA_D) ? LLAMA_D : NAT_YFITS(1024) ? 1024 : NAT_YFITS(512) ? 512 : \
                NAT_YFITS(256) ? 256 : 128)
#define NAT_NCH (LLAMA_D / NAT_NC)
LLAMA_SPAD_REQUIRE(nat_qkv_spad,   ROWS8(LLAMA_M, LLAMA_D) <= HALF_ROWS && ROWS8(LLAMA_D, LLAMA_H) <= HALF_ROWS);
LLAMA_SPAD_REQUIRE(nat_qkv_acc,    NAT_ACC_ROWS(LLAMA_M, LLAMA_H) <= ACC_ROWS / 2);
LLAMA_SPAD_REQUIRE(nat_qkv_scale,  LLAMA_GD * LLAMA_M <= 4096 && LLAMA_GD * LLAMA_H <= 4096);
LLAMA_SPAD_REQUIRE(nat_s_spad,     ROWS8(LLAMA_M, LLAMA_H) + ROWS8(LLAMA_H, LLAMA_M) <= HALF_ROWS);
LLAMA_SPAD_REQUIRE(nat_oproj_fits, NAT_YFITS(NAT_NC) && NAT_NC * NAT_NCH == LLAMA_D);
LLAMA_SPAD_REQUIRE(nat_oproj_war,  NAT_NCH <= 2 || 2 * (LLAMA_M / DIM) * (NAT_NC / DIM) * (LLAMA_H / DIM) > 16);
LLAMA_SPAD_REQUIRE(nat_o_at_a_region, SPAD_O == 0);
static uint32_t nat_sink[512] __attribute__((aligned(32)));
#define NAT_CONFIG_BF16() gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, \
                                                      0, 0, false, 0, 0, OUT_BF16, 0)
#endif

static float    xn_f[LLAMA_M * LLAMA_D];
static uint8_t  xn_codes[LLAMA_M * LLAMA_D] __attribute__((aligned(64)));
static uint8_t  xn_scales[LLAMA_GD * LLAMA_M] __attribute__((aligned(64)));
static uint16_t Q_hw[LLAMA_M * LLAMA_H] __attribute__((aligned(64))), K_hw[LLAMA_M * LLAMA_H] __attribute__((aligned(64))), V_hw[LLAMA_M * LLAMA_H] __attribute__((aligned(64)));
static float    q_f[LLAMA_M * LLAMA_H], k_f[LLAMA_M * LLAMA_H], kt_f[LLAMA_H * LLAMA_M];
static float    v_f[LLAMA_M * LLAMA_H];
static uint8_t  q_codes[LLAMA_M * LLAMA_H] __attribute__((aligned(64))), q_scales[LLAMA_GH * LLAMA_M] __attribute__((aligned(64)));
static uint8_t  kt_codes[LLAMA_H * LLAMA_M] __attribute__((aligned(64))), kt_scales[LLAMA_GH * LLAMA_M] __attribute__((aligned(64)));
static uint8_t  v_codes[LLAMA_M * LLAMA_H] __attribute__((aligned(64))), v_scales[LLAMA_GM * LLAMA_H] __attribute__((aligned(64)));
static uint16_t S_hw[LLAMA_M * LLAMA_M] __attribute__((aligned(64)));
static float    p_f[LLAMA_M * LLAMA_M];
static uint8_t  p_codes[LLAMA_M * LLAMA_M] __attribute__((aligned(64))), p_scales[LLAMA_GM * LLAMA_M] __attribute__((aligned(64)));
static uint8_t  O_hw[LLAMA_M * LLAMA_H] __attribute__((aligned(64)));
static uint32_t o_scales_dram[512] __attribute__((aligned(32)));
static uint8_t  wo_scales_chunk[LLAMA_GH * NCHUNK];
static uint16_t Ychunk[LLAMA_M * NCHUNK];
static uint16_t Y_hw[LLAMA_M * LLAMA_D] __attribute__((aligned(64)));
static uint16_t OUT_hw[LLAMA_M * LLAMA_D];

// riscv-tests' handle_trap is weak and exits 1337 with no cause, which is indistinguishable from a
// stale libgemmini.so. This turns that into one diagnosable line.
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  printf("TRAP cause=%d epc=%lx\n", (int) cause, (unsigned long) epc);
  tohost_exit(1337);
  return 0;
}

// k-/j-tiles per mvin: rows of DIM*w bytes (64B at w=4 = one full DMA read), block stride DIM keeps the
// per-tile slots. w = largest of MVIN_TILES_MAX/2/1 dividing the tile count. -DMVIN_TILES_MAX=1: 1-tile.
#ifndef MVIN_TILES_MAX
#define MVIN_TILES_MAX 4
#endif
static int mvin_width(int tiles) {
  int w = MVIN_TILES_MAX;
  while (w > 1 && tiles % w) w >>= 1;
  return w;
}

// `stride` is the SOURCE row pitch, which differs from K when the tile is a column slice of a
// wider array -- that is how a K-tile of Xn[M][D] moves in without being copied out first.
static void mvin_A_strided(const uint8_t *A, int M, int K, int stride, uint32_t a_spad) {
  gemmini_config_ld(stride * sizeof(uint8_t));
  int tiles_I = M / DIM, tiles_K = K / DIM, w = mvin_width(tiles_K);
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k += w)
      gemmini_extended_mvin((void *) (A + (size_t) i * DIM * stride + (size_t) k * DIM),
                            a_spad + (i * tiles_K + k) * DIM, DIM * w, DIM);
}

static void mvin_A(const uint8_t *A, int M, int K, uint32_t a_spad) {
  mvin_A_strided(A, M, K, K, a_spad);
}

static void mvin_B(const uint8_t *B, int K, int N_full, int n0, int N, uint32_t b_spad) {
  gemmini_config_ld(N_full * sizeof(uint8_t));
  int tiles_K = K / DIM, tiles_J = N / DIM, w = mvin_width(tiles_J);
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j += w)
      gemmini_extended_mvin((void *) (B + (size_t) k * DIM * N_full + (size_t) (n0 + j * DIM)),
                            b_spad + (k * tiles_J + j) * DIM, DIM * w, DIM);
}

static void mvout_bf16(uint16_t *dst, uint32_t spad, int M, int N) {
  gemmini_config_st(DIM * sizeof(uint8_t));
  int total_rows = M * N * 2 / DIM;
  uint8_t *b = (uint8_t *) dst;
  for (int r = 0; r < total_rows; r += DIM)
    gemmini_extended_mvout(b + (size_t) r * DIM, spad + r, DIM, DIM);
  gemmini_fence();
}

// Read a BLOCK-TILED fp8 tile back to a flat [M][N] buffer. Contiguous mvout into a temp, then a
// software de-tile: a strided de-tiling mvout makes the writer DMA emit whole cache lines and
// zero-fill the gaps on RTL (matmul_tiled_fp8_64x64_chain.c:79-83). Read-only -- O stays resident.
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

// One mesh matmul. `out_fmt` picks BF16 (drained by the host) or FP8 requant; `resident` routes the
// requantizer's block scales into the act-scale window as well as to DRAM, and `tiled` deposits the
// codes in the operand-A layout -- together, the next matmul's A operand, in place.
// `accum` is loop_ws's ex_accumulate (rs1 bit 0): 0 OVERWRITES the output region, 1 adds into it.
// Only a K-tile after the first wants 1.
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

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  printf("llama attention: M=%d D=%d head_dim=%d  (real TinyLlama head, fp8 e4m3 + E8M0)\n",
         LLAMA_M, LLAMA_D, LLAMA_H);
  printf("plan  spad %d rows: proj %d K-tile(s) of %d | o_proj %d chunk(s) of %d\n",
         SPAD_TOP, KTILES, KTILE, NCHUNKS, NCHUNK);

  static uint32_t scale_sink[512] __attribute__((aligned(32)));
  gemmini_flush(0);

  uint64_t t_host = 0, t_mesh = 0, t0;

  // ================= host: RMSNorm =================
  t0 = read_cycles();
#ifdef LLAMA_SKIP_RMSNORM
  // Skip the ~524k-cycle host RMSNorm: use the golden Xn codes/scales directly (they are what the
  // host would produce, verified 0/2048 in the full test). Makes the mesh-only repro fast in VCS.
  for (size_t i = 0; i < (size_t) LLAMA_M * LLAMA_D; i++) xn_codes[i] = ((const uint8_t *) XN_CODES)[i];
  for (size_t i = 0; i < (size_t) LLAMA_GD * LLAMA_M; i++) xn_scales[i] = ((const uint8_t *) XN_SCALES_ROW)[i];
#else
  mx_rmsnorm((const uint16_t *) H_PRE_BF16, W_IN_LN_BF16, LLAMA_M, LLAMA_D, LLAMA_RMS_EPS, xn_f);
  mx_quantize_rows(xn_f, LLAMA_M, LLAMA_D, xn_codes, xn_scales);
#endif
  t_host += read_cycles() - t0;
  printf("host  rmsnorm+quant: codes differ %d/%d, scales differ %d/%d vs golden\n",
         mx_count_diff_u8(xn_codes, (const uint8_t *) XN_CODES, LLAMA_M * LLAMA_D),
         LLAMA_M * LLAMA_D,
         mx_count_diff_u8(xn_scales, (const uint8_t *) XN_SCALES_ROW, LLAMA_GD * LLAMA_M),
         LLAMA_GD * LLAMA_M);

  // ================= mesh: Q, K, V, each K-tiled over the D-deep contraction =================
  const uint8_t *wcodes[3] = { (const uint8_t *) WQ_IN, (const uint8_t *) WK_IN,
                               (const uint8_t *) WV_IN };
  const uint8_t *wscales[3] = { (const uint8_t *) WQ_SCALES_COL, (const uint8_t *) WK_SCALES_COL,
                                (const uint8_t *) WV_SCALES_COL };
  uint16_t *qkv_hw[3] = { Q_hw, K_hw, V_hw };
  const uint16_t *qkv_gold[3] = { (const uint16_t *) Q_OUT_BF16, (const uint16_t *) K_OUT_BF16,
                                  (const uint16_t *) V_OUT_BF16 };
  const char *qkv_name[3] = { "Q", "K", "V" };
  int qkv_diff = 0;
  static uint16_t cap_row0[3][8];   // each matmul's output row 0, captured BEFORE the next iter can clobber it
  static uint16_t cap_qlast[8];     // Q's last row, for leak comparison
  // LLAMA_ONLY_PROJ isolates one projection (0=Q,1=K,2=V) as a standalone FIRST matmul; LLAMA_NPROJ
  // runs the first N (default 3); LLAMA_SEPARATE_QKV drains each to its own spad region instead of
  // reusing SPAD_QKV -- if that makes K/V pass, the bug is the acc/region reuse not being reset.
#ifndef LLAMA_NPROJ
#define LLAMA_NPROJ 3
#endif
  // CPU-side view of Xn in DRAM: changes between projections => something wrote it.
#define XN_PROBE(stage, name) do { gemmini_fence(); \
    printf("xnprobe %s-%s @%p diff=%d/%d row0:", stage, name, (void *) xn_codes, \
           mx_count_diff_u8(xn_codes, (const uint8_t *) XN_CODES, LLAMA_M * LLAMA_D), LLAMA_M * LLAMA_D); \
    for (int c_ = 0; c_ < 16; c_++) printf(" %02x", xn_codes[c_]); printf("\n"); } while (0)
#ifdef ATTN_NATIVE
  if (((uintptr_t) WQ_SCALES_COL | (uintptr_t) WK_SCALES_COL | (uintptr_t) WV_SCALES_COL |
       (uintptr_t) WO_SCALES_COL | (uintptr_t) xn_scales | (uintptr_t) q_scales | (uintptr_t) kt_scales) & 7) {
    printf("native: a scale array is not 8B-aligned\n");
    return 1;
  }
  t0 = read_cycles();
  NAT_CONFIG_BF16();
  gemmini_extended3_config_ld(LLAMA_D * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(LLAMA_H * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(LLAMA_H * sizeof(uint16_t));
  for (int s = 0; s < 3; s++) {
    gemmini_loop_ws_mx(LLAMA_M / DIM, LLAMA_H / DIM, LLAMA_D / DIM,
                       s == 0 ? xn_codes : NULL, wcodes[s], qkv_hw[s], LLAMA_D, LLAMA_H, LLAMA_H,
                       xn_scales, wscales[s], LLAMA_M, LLAMA_H, false, 1, NAT_QKV_SPLIT ? 2 : 1 + (s & 1));
  }
  gemmini_fence();
  uint64_t t_qkv = read_cycles() - t0;
  t_mesh += t_qkv;
  for (int s = 0; s < 3; s++) {
    int d = mx_count_diff_u16(qkv_hw[s], qkv_gold[s], LLAMA_M * LLAMA_H);
    qkv_diff += d;
    printf("mesh  %s = Xn @ W%s : %d/%d differ from golden\n", qkv_name[s], qkv_name[s], d, LLAMA_M * LLAMA_H);
  }
  (void) cap_row0; (void) cap_qlast;
  for (int s = 0; s < 0; s++) {
#elif defined(LLAMA_ONLY_PROJ)
  for (int s = LLAMA_ONLY_PROJ; s < LLAMA_ONLY_PROJ + 1; s++) {
#else
  for (int s = 0; s < LLAMA_NPROJ; s++) {
#endif
#ifdef LLAMA_SEPARATE_QKV
    uint32_t qkv_dst = SPAD_QKV + (uint32_t) s * ROWS16(LLAMA_M, LLAMA_H);
#else
    uint32_t qkv_dst = SPAD_QKV;
#endif
    XN_PROBE("before", qkv_name[s]);
    t0 = read_cycles();
    // Tile t contributes Xn[:, t*KTILE ..] @ W[t*KTILE .., :] into SPAD_QKV: the first overwrites,
    // the rest accumulate, so after the loop the region holds the full D-deep reduction. Both
    // scale windows take a CONTIGUOUS slice -- the A window is [GD][M] and the B window [GD][H],
    // so a K-tile is whole rows of each and no gather is needed.
    for (int kt = 0; kt < KTILES; kt++) {
      mvin_A_strided(xn_codes + (size_t) kt * KTILE, LLAMA_M, KTILE, LLAMA_D, SPAD_XN);
      gemmini_mx_load_scales((uint64_t) (xn_scales + (size_t) kt * KGRP * LLAMA_M),
                             KGRP * LLAMA_M, 0);
      gemmini_mx_load_scales((uint64_t) (wscales[s] + (size_t) kt * KGRP * LLAMA_H),
                             KGRP * LLAMA_H, 1);
      gemmini_fence();
      mvin_B(wcodes[s] + (size_t) kt * KTILE * LLAMA_H, KTILE, LLAMA_H, 0, LLAMA_H, SPAD_PB);
      mesh_matmul(LLAMA_M, KTILE, LLAMA_H, SPAD_XN, SPAD_PB_ARG, qkv_dst,
                  OUT_BF16, (uint64_t) scale_sink, 0, kt > 0);
    }
    t_mesh += read_cycles() - t0;
    XN_PROBE("premvout", qkv_name[s]);
    mvout_bf16(qkv_hw[s], qkv_dst, LLAMA_M, LLAMA_H);
    XN_PROBE("postmvout", qkv_name[s]);
    for (int c = 0; c < 8; c++) cap_row0[s][c] = qkv_hw[s][c];     // corruption-proof snapshot
    if (s == 0) for (int c = 0; c < 8; c++) cap_qlast[c] = Q_hw[(LLAMA_M - 1) * LLAMA_H + c];
    int d = mx_count_diff_u16(qkv_hw[s], qkv_gold[s], LLAMA_M * LLAMA_H);
    qkv_diff += d;
    printf("mesh  %s = Xn @ W%s : %d/%d differ from golden\n",
           qkv_name[s], qkv_name[s], d, LLAMA_M * LLAMA_H);
    if (d) {  // localize the regression: which output rows differ, and by how much
      for (int r = 0; r < LLAMA_M; r++) {
        int rd = 0;
        for (int c = 0; c < LLAMA_H; c++)
          if (qkv_hw[s][(size_t) r * LLAMA_H + c] != qkv_gold[s][(size_t) r * LLAMA_H + c]) rd++;
        if (rd)
          printf("      %s row %2d: %d/%d cols differ  hw[0]=%04x gold[0]=%04x\n",
                 qkv_name[s], r, rd, LLAMA_H,
                 qkv_hw[s][(size_t) r * LLAMA_H], qkv_gold[s][(size_t) r * LLAMA_H]);
      }
    }
  }

#ifdef LLAMA_QKV_ONLY
  // Isolated repro: stop after the Q/K/V projections. This alone reproduces the back-to-back
  // BF16 LUT-matmul bug (Q clean, K/V one output row wrong) without scores/softmax/P@V/o_proj.
  // Leak probe (corruption-proof snapshots taken right after each mvout). Compare K/V row 0 (wrong)
  // against Q's row 0 and Q's LAST row: match Q[last] -> output-pipeline tail leak; match Q[0] ->
  // input-side replay; matches neither -> distinct stale/garbage.
  printf("  cap Q row0 = %04x %04x %04x %04x  Q rowLast = %04x %04x %04x %04x\n",
         cap_row0[0][0], cap_row0[0][1], cap_row0[0][2], cap_row0[0][3],
         cap_qlast[0], cap_qlast[1], cap_qlast[2], cap_qlast[3]);
  printf("  cap K row0 = %04x %04x %04x %04x  gold = %04x %04x %04x %04x\n",
         cap_row0[1][0], cap_row0[1][1], cap_row0[1][2], cap_row0[1][3],
         ((const uint16_t *) K_OUT_BF16)[0], ((const uint16_t *) K_OUT_BF16)[1],
         ((const uint16_t *) K_OUT_BF16)[2], ((const uint16_t *) K_OUT_BF16)[3]);
  printf("  cap V row0 = %04x %04x %04x %04x  gold = %04x %04x %04x %04x\n",
         cap_row0[2][0], cap_row0[2][1], cap_row0[2][2], cap_row0[2][3],
         ((const uint16_t *) V_OUT_BF16)[0], ((const uint16_t *) V_OUT_BF16)[1],
         ((const uint16_t *) V_OUT_BF16)[2], ((const uint16_t *) V_OUT_BF16)[3]);
  // Did a later matmul clobber an earlier output buffer in DRAM? (K_hw now vs its captured row0.)
  printf("  post-loop K_hw row0 = %04x %04x %04x %04x  (was %04x at produce time)\n",
         K_hw[0], K_hw[1], K_hw[2], K_hw[3], cap_row0[1][0]);
  printf("cycles mesh %d, host %d\n", (int) t_mesh, (int) t_host);
  if (qkv_diff == 0)
    printf("llama QKV-only test PASSED (Q, K, V projections bit-exact).\n");
  else
    printf("llama QKV-only test FAILED: %d mesh element(s) differ from golden.\n", qkv_diff);
#ifndef BAREMETAL
  exit(qkv_diff != 0);
#else
  return qkv_diff != 0;
#endif
#endif

  // ================= host: RoPE, and the K transpose the MX loop cannot do =================
  t0 = read_cycles();
  mx_rope(Q_hw, (const uint32_t *) ROPE_COS_F32, (const uint32_t *) ROPE_SIN_F32,
          LLAMA_M, LLAMA_H, q_f);
  mx_rope(K_hw, (const uint32_t *) ROPE_COS_F32, (const uint32_t *) ROPE_SIN_F32,
          LLAMA_M, LLAMA_H, k_f);
  mx_quantize_rows(q_f, LLAMA_M, LLAMA_H, q_codes, q_scales);
  mx_transpose_f32(k_f, LLAMA_M, LLAMA_H, kt_f);
  mx_quantize_cols(kt_f, LLAMA_H, LLAMA_M, kt_codes, kt_scales);
  t_host += read_cycles() - t0;
  printf("host  RoPE(Q,K) + K transpose: Q codes differ %d/%d, K^T codes differ %d/%d vs golden\n",
         mx_count_diff_u8(q_codes, (const uint8_t *) Q_CODES, LLAMA_M * LLAMA_H),
         LLAMA_M * LLAMA_H,
         mx_count_diff_u8(kt_codes, (const uint8_t *) KT_IN, LLAMA_H * LLAMA_M), LLAMA_H * LLAMA_M);

  // ================= mesh: S = Q @ K^T =================
  t0 = read_cycles();
#ifdef ATTN_NATIVE
  NAT_CONFIG_BF16();
  gemmini_extended3_config_ld(LLAMA_H * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(LLAMA_M * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(LLAMA_M * sizeof(uint16_t));
  gemmini_loop_ws_mx(LLAMA_M / DIM, LLAMA_M / DIM, LLAMA_H / DIM,
                     q_codes, kt_codes, S_hw, LLAMA_H, LLAMA_M, LLAMA_M,
                     q_scales, kt_scales, LLAMA_M, LLAMA_M, false, 1, 1);
  gemmini_fence();
  uint64_t t_s = read_cycles() - t0;
  t_mesh += t_s;
#else
  mvin_A(q_codes, LLAMA_M, LLAMA_H, SPAD_QA);
  mvin_B(kt_codes, LLAMA_H, LLAMA_M, 0, LLAMA_M, SPAD_KT);
  gemmini_mx_load_scales((uint64_t) q_scales, sizeof(q_scales), 0);
  gemmini_mx_load_scales((uint64_t) kt_scales, sizeof(kt_scales), 1);
  gemmini_fence();
  mesh_matmul(LLAMA_M, LLAMA_H, LLAMA_M, SPAD_QA, SPAD_KT_ARG, SPAD_S,
              OUT_BF16, (uint64_t) scale_sink, 0, 0);
  t_mesh += read_cycles() - t0;
  mvout_bf16(S_hw, SPAD_S, LLAMA_M, LLAMA_M);
#endif
  int s_d = mx_count_diff_u16(S_hw, (const uint16_t *) S_OUT_BF16, LLAMA_M * LLAMA_M);
  printf("mesh  S = Q @ K^T : %d/%d differ from golden\n", s_d, LLAMA_M * LLAMA_M);

  // ================= host: scale, causal mask, softmax; V as a B operand =================
  t0 = read_cycles();
  mx_softmax_causal(S_hw, LLAMA_M, 1.0f / sqrtf((float) LLAMA_H), p_f);
  mx_quantize_rows(p_f, LLAMA_M, LLAMA_M, p_codes, p_scales);
  for (int i = 0; i < LLAMA_M * LLAMA_H; i++) v_f[i] = mx_bf16_to_f32(V_hw[i]);
  mx_quantize_cols(v_f, LLAMA_M, LLAMA_H, v_codes, v_scales);
  t_host += read_cycles() - t0;
  printf("host  causal softmax + quant: P codes differ %d/%d, V codes differ %d/%d vs golden\n",
         mx_count_diff_u8(p_codes, (const uint8_t *) P_CODES, LLAMA_M * LLAMA_M), LLAMA_M * LLAMA_M,
         mx_count_diff_u8(v_codes, (const uint8_t *) V_IN, LLAMA_M * LLAMA_H), LLAMA_M * LLAMA_H);

  // ================= mesh: O = P @ V, requantized straight into the scratchpad =================
  t0 = read_cycles();
  mvin_A(p_codes, LLAMA_M, LLAMA_M, SPAD_PA);
  mvin_B(v_codes, LLAMA_M, LLAMA_H, 0, LLAMA_H, SPAD_VB);
  gemmini_mx_load_scales((uint64_t) p_scales, sizeof(p_scales), 0);
  gemmini_mx_load_scales((uint64_t) v_scales, sizeof(v_scales), 1);
  gemmini_fence();
  mesh_matmul(LLAMA_M, LLAMA_M, LLAMA_H, SPAD_PA, SPAD_VB_ARG, SPAD_O,
              OUT_FP8, (uint64_t) o_scales_dram, 1, 0);
  uint64_t t_o = read_cycles() - t0;
  t_mesh += t_o;

  // Residency check: read O back WITHOUT disturbing it, and check the scales the requantizer wrote.
  mvout_detile(O_hw, SPAD_O, LLAMA_M, LLAMA_H);
  int o_d = mx_count_diff_u8(O_hw, (const uint8_t *) O_OUT, LLAMA_M * LLAMA_H);
  int o_sd = mx_count_diff_u8((const uint8_t *) o_scales_dram, (const uint8_t *) O_SCALES_OUT,
                              LLAMA_M * LLAMA_GH);
  printf("mesh  O = P @ V   : %d/%d codes, %d/%d scales differ from golden (requant -> spad)\n",
         o_d, LLAMA_M * LLAMA_H, o_sd, LLAMA_M * LLAMA_GH);

  // ================= mesh: o_proj, reading O and its scales IN PLACE =================
  // No A mvin and no A-scale load: the codes are already resident at SPAD_O in the operand-A tiled
  // layout, and the requantizer wrote their E8M0 bytes into the act-scale window transposed
  // ([H/32][M], a_off = group * M + row), which is exactly what this matmul indexes.
#ifdef ATTN_NATIVE
  t0 = read_cycles();
  NAT_CONFIG_BF16();
  gemmini_extended3_config_ld(LLAMA_D * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(LLAMA_D * sizeof(uint16_t));
  for (int c = 0; c < NAT_NCH; c++) {
    gemmini_mx_load_scales_2d(&WO_SCALES_COL[0][c * NAT_NC], NAT_NC, LLAMA_GH, LLAMA_D, (c & 1) << 12, 1);
    gemmini_mxquant_config_mvout_wait((uint64_t) nat_sink, LLAMA_M / DIM, NAT_NC / DIM, LLAMA_H / DIM,
                                      0, c & 1, 1);
    gemmini_loop_ws(LLAMA_M / DIM, NAT_NC / DIM, LLAMA_H / DIM, 0, 0, 0,
                    NULL, (const uint8_t *) WO_IN + c * NAT_NC, NULL, Y_hw + c * NAT_NC, LLAMA_H, LLAMA_D, 0, LLAMA_D,
                    false, false, false, false, false, NO_ACTIVATION, 1, 1 + (c & 1), false);
  }
  gemmini_fence();
  uint64_t t_y = read_cycles() - t0;
  t_mesh += t_y;
  printf("phase QKV %d | S %d | O (resident) %d | o_proj x%d (resident A + scales) %d\n",
         (int) t_qkv, (int) t_s, (int) t_o, NAT_NCH, (int) t_y);
  for (int c = 0; c < 0; c++) {
#else
  for (int c = 0; c < NCHUNKS; c++) {
#endif
    for (int g = 0; g < LLAMA_GH; g++)
      memcpy(wo_scales_chunk + (size_t) g * NCHUNK, &WO_SCALES_COL[g][c * NCHUNK], NCHUNK);
    t0 = read_cycles();
    gemmini_mx_load_scales((uint64_t) wo_scales_chunk, sizeof(wo_scales_chunk), 1);
    gemmini_fence();
    mvin_B((const uint8_t *) WO_IN, LLAMA_H, LLAMA_D, c * NCHUNK, NCHUNK, SPAD_WO);
    mesh_matmul(LLAMA_M, LLAMA_H, NCHUNK, SPAD_O, SPAD_WO_ARG, SPAD_Y,
                OUT_BF16, (uint64_t) scale_sink, 0, 0);
    t_mesh += read_cycles() - t0;
    mvout_bf16(Ychunk, SPAD_Y, LLAMA_M, NCHUNK);
    for (int m = 0; m < LLAMA_M; m++)
      memcpy(&Y_hw[(size_t) m * LLAMA_D + c * NCHUNK], &Ychunk[(size_t) m * NCHUNK],
             NCHUNK * sizeof(uint16_t));
  }
  int y_d = mx_count_diff_u16(Y_hw, (const uint16_t *) Y_OUT_BF16, LLAMA_M * LLAMA_D);
  printf("mesh  Y = O @ Wo  : %d/%d differ from golden (O resident, scales resident, %d chunks)\n",
         y_d, LLAMA_M * LLAMA_D, NCHUNKS);

  // ================= host: the residual =================
  t0 = read_cycles();
  for (int i = 0; i < LLAMA_M * LLAMA_D; i++)
    OUT_hw[i] = mx_f32_to_bf16_rne(mx_bf16_to_f32(((const uint16_t *) H_PRE_BF16)[i])
                                   + mx_bf16_to_f32(Y_hw[i]));
  t_host += read_cycles() - t0;

  printf("grade attention out vs fp32 reference: rel_fro %d ppm\n",
         MX_PPM(mx_rel_fro_bf16(Y_hw, (const uint16_t *) REF_ATTN_BF16, LLAMA_M * LLAMA_D)));
  printf("grade residual out  vs fp32 reference: rel_fro %d ppm\n",
         MX_PPM(mx_rel_fro_bf16(OUT_hw, (const uint16_t *) REF_OUT_BF16, LLAMA_M * LLAMA_D)));
  printf("cycles mesh %d, host %d\n", (int) t_mesh, (int) t_host);

  int exact = qkv_diff + s_d + o_d + o_sd + y_d;
  if (exact == 0)
    printf("llama attention test PASSED (6 mesh matmuls bit-exact; O and its scales reused in "
           "place by o_proj).\n");
  else
    printf("llama attention test FAILED: %d mesh element(s) differ from golden.\n", exact);

#ifndef BAREMETAL
  exit(exact != 0);
#else
  return exact != 0;
#endif
}
