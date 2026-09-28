// DEBUG VARIANT of llama_attention_full.c -- same kernel, same plan, same data, plus diagnostics for
// the FPGA result "every stage bit-exact except O: 1/65536 codes and Y: 658/65536".
//
//   1. o_proj runs TWICE: once on the hardware's O (the real kernel, graded as before), then on the
//      GOLDEN O codes and scales. If the second Y is 0/65536, the 32-call cross-head accumulation is
//      correct on RTL and every Y mismatch is downstream of the bad O code(s).
//   2. Y mismatches are histogrammed per token row, for both runs: a single bad O_h[m][j] can only
//      reach row m of Y.
//   3. Every bad O code is printed with its head/row/col, got/expected code and got/expected E8M0.
//   4. For the FIRST bad O code, that head's P @ V is re-run with a BF16 output (no requant) and the
//      32-element E8M0 block holding it is printed -- the requantizer's exact input, so the one
//      rounding decision can be replayed against the C model offline.
//
// Nothing the original kernel computes is changed: the diagnostics run after each stage's own
// grading, re-load every operand they touch, and the PASS/FAIL verdict is the original one.
//
// ---- the original kernel's description follows ----
//
// A COMPLETE TinyLlama attention sub-layer on MxGemmini -- ALL 32 query heads, one ELF, real data.
//
// Unlike llama_attention.c, which runs one head and so grades a partial sum over 1 of 32, nothing
// here is truncated: the full 2048x2048 q_proj and o_proj, all 4 GQA kv heads, the hidden size
// whole. The result is therefore this layer's REAL attention output, and the blob carries
// ATTN_TORCH -- what TinyLlama's own self_attn module produced on the same tokens -- so for the
// first time the kernel can be graded against the model instead of against a reimplementation.
//
//   host   xn = rmsnorm(h_pre, w_in_ln)                      fp32 -> MX
//   mesh   Q = Xn @ Wq        [M,D]x[D,QD]                   N-chunked, Xn resident
//   mesh   K,V = Xn @ Wk/Wv   [M,D]x[D,KVD]                  same A, same pass
//   host   RoPE per head; K transposed per kv head           fp32 -> MX
//   mesh   per head  S_h = Q_h @ K_kv^T   [M,H]x[H,M]
//   host   per head  softmax(causal(S_h/sqrt(H)))            fp32 -> MX
//   mesh   per head  O_h = P_h @ V_kv     -> FP8 requant into the scratchpad
//   mesh   Y = sum_h O_h @ Wo[H*h : H*h+H, :]                ACCUMULATED IN SMEM ACROSS HEADS
//   host   out = h_pre + Y
//
// THE CROSS-HEAD ACCUMULATION IS THE POINT. o_proj issues loop_ws with ex_accumulate = 1 for every
// head after the first, so each head's [M,H]x[H,D] contribution ADDS into the same Y region and the
// concatenated [M,QD] operand is never materialized. Every other matmul here passes 0, which
// overwrites its output region -- that is what lets one region serve all 40 projection chunks and
// all 32 heads. Wiring that bit through the MX path is what made this kernel possible at all;
// before it, mx_loop_ws_spad discarded rs1 and ALWAYS accumulated, so no region could ever be
// reused and the full layer needed 26624 rows of output address space against 16384 available.
//
// TILING ADAPTS TO THE SCRATCHPAD. Every chunk width below is chosen at COMPILE TIME from
// BANK_NUM * BANK_ROWS, so the same source runs the largest blocking a given config can hold and
// the algorithm reshapes itself rather than overflowing. The banner prints the plan it picked.
// The data blob is independent of all of it: an output column depends only on its own column of B,
// so any chunking reproduces the same goldens and only the C replans.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "mx_host.h"
#include "llama_attn_full.h"

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
#define LOOP_WS_REQUANT_TILED (1u << 10)

// ---------------------------------------------------------------------------------------------
// The plan. Scratchpad rows an operand occupies: one 16-byte row per DIM elements, FP8 one byte
// per element and a BF16 output two.
// ---------------------------------------------------------------------------------------------
#define SPAD_ROWS    (BANK_NUM * BANK_ROWS)
#define ROWS8(m, n)  ((m) * (n) / DIM)
#define ROWS16(m, n) ((m) * (n) * 2 / DIM)

// -- phase 1: the projections, [M,D] x [D,Nc] -> BF16. Xn is resident as A across all three, so a
//    B tile and the output region must fit beside it. Larger Nc = fewer, bigger matmuls, so take the
//    largest that fits; the B tile is what dominates, being D deep. ONE output region serves every
//    chunk: loop_ws is issued with ex_accumulate = 0, which overwrites it.
#define P1_COST(nc) (ROWS8(LLAMA_M, LLAMA_D) + ROWS8(LLAMA_D, (nc)) + ROWS16(LLAMA_M, (nc)))
// The projection keeps the full D-deep contraction in ONE loop_ws, so its B-side scale window is
// (PROJ_N/16) * (D/32) rows against the 256 the hardware holds -- a second budget that binds
// independently of the scratchpad. At D = 2048 that caps PROJ_N at 64, which is what the
// scratchpad picks anyway; the term is here so a bigger scratchpad cannot silently choose 128 and
// wrap the scale rows. See planning/rtl_fault_b_kdepth.md.
#ifndef LLAMA_SCALE_ROWS_MAX
#define LLAMA_SCALE_ROWS_MAX 256
#endif
#define SCALE_ROWS(k, n) ((n) * (k) / 512)
#define P1FITS(nc) (P1_COST(nc) <= SPAD_ROWS && \
                    SCALE_ROWS(LLAMA_D, (nc)) <= LLAMA_SCALE_ROWS_MAX)
#define PROJ_N   (P1FITS(512) ? 512 : \
                  P1FITS(256) ? 256 : \
                  P1FITS(128) ? 128 : \
                  P1FITS(64)  ? 64  : \
                  P1FITS(32)  ? 32  : 16)
#define SPAD_XN     0
#define PROJ_C      ROWS8(LLAMA_M, LLAMA_D)                  // the output sits above Xn
#define PROJ_B      (SPAD_ROWS - ROWS8(LLAMA_D, PROJ_N))     // B tiles live at the top
#define PROJ_B_ARG  SPAD_ROWS

// -- phase 2: per head, S = Q_h @ K_kv^T then O = P_h @ V_kv. Every region is reused across heads.
#define H2_A_Q      0
#define H2_B_KT     (H2_A_Q + ROWS8(LLAMA_M, LLAMA_H))
#define H2_B_KT_ARG (H2_B_KT + ROWS8(LLAMA_H, LLAMA_M))
#define H2_A_P      H2_B_KT_ARG
#define H2_B_V      (H2_A_P + ROWS8(LLAMA_M, LLAMA_M))
#define H2_B_V_ARG  (H2_B_V + ROWS8(LLAMA_M, LLAMA_H))
#define H2_C_S      H2_B_V_ARG
#define H2_C_O      (H2_C_S + ROWS16(LLAMA_M, LLAMA_M))
#define H2_C_DBG    (H2_C_O + ROWS8(LLAMA_M, LLAMA_H))      // debug: P@V as BF16, no requant

// -- phase 3: o_proj. Y_c must stay resident while all NH heads accumulate into it, so only ONE
//    chunk is live at a time and the scratchpad decides how wide it is.
#define P3_COST(dc) (ROWS16(LLAMA_M, (dc)) + ROWS8(LLAMA_H, (dc)) + ROWS8(LLAMA_M, LLAMA_H))
#define YCHUNK   (P3_COST(2048) <= SPAD_ROWS ? 2048 : \
                  P3_COST(1024) <= SPAD_ROWS ? 1024 : \
                  P3_COST(512)  <= SPAD_ROWS ? 512  : \
                  P3_COST(256)  <= SPAD_ROWS ? 256  : \
                  P3_COST(128)  <= SPAD_ROWS ? 128  : 64)
#define YCHUNKS     (LLAMA_D / YCHUNK)
#define SPAD_Y      0
#define SPAD_WO     (SPAD_Y + ROWS16(LLAMA_M, YCHUNK))
#define SPAD_WO_ARG (SPAD_WO + ROWS8(LLAMA_H, YCHUNK))
#define SPAD_OA     SPAD_WO_ARG

// The plan must fit the scratchpad it was planned for -- checked at COMPILE time, because an
// overflowing region aliases silently and produces plausible numbers.
#define LLAMA_REQUIRE(name, cond) typedef char llama_plan_##name[(cond) ? 1 : -1]
LLAMA_REQUIRE(proj_fits,        PROJ_C + ROWS16(LLAMA_M, PROJ_N) <= PROJ_B);
LLAMA_REQUIRE(heads_fit,        H2_C_O + ROWS8(LLAMA_M, LLAMA_H) <= SPAD_ROWS);
LLAMA_REQUIRE(dbg_fits,         H2_C_DBG + ROWS16(LLAMA_M, LLAMA_H) <= SPAD_ROWS);
LLAMA_REQUIRE(ychunk_divides_d, YCHUNK * YCHUNKS == LLAMA_D);
LLAMA_REQUIRE(oproj_fits,       SPAD_OA + ROWS8(LLAMA_M, LLAMA_H) <= SPAD_ROWS);
LLAMA_REQUIRE(proj_n_divides,   (LLAMA_QD % PROJ_N) == 0);

// ---------------------------------------------------------------------------------------------
// Host buffers.
// ---------------------------------------------------------------------------------------------
static float    xn_f[LLAMA_M * LLAMA_D];
static uint8_t  xn_codes[LLAMA_M * LLAMA_D];
static uint8_t  xn_scales[LLAMA_GD * LLAMA_M];
static uint16_t Q_hw[LLAMA_M * LLAMA_QD], K_hw[LLAMA_M * LLAMA_KVD], V_hw[LLAMA_M * LLAMA_KVD];
static uint16_t chunk16[LLAMA_M * PROJ_N];
static float    tmp_f[LLAMA_M * LLAMA_H], tmp_t[LLAMA_H * LLAMA_M];
static uint8_t  q_codes[LLAMA_NH][LLAMA_M * LLAMA_H], q_scales[LLAMA_NH][LLAMA_GH * LLAMA_M];
static uint8_t  kt_codes[LLAMA_NKV][LLAMA_H * LLAMA_M], kt_scales[LLAMA_NKV][LLAMA_GH * LLAMA_M];
static uint8_t  v_codes[LLAMA_NKV][LLAMA_M * LLAMA_H], v_scales[LLAMA_NKV][LLAMA_GM * LLAMA_H];
static uint16_t S_hw[LLAMA_NH][LLAMA_M * LLAMA_M];
static float    p_f[LLAMA_M * LLAMA_M];
static uint8_t  p_codes[LLAMA_NH][LLAMA_M * LLAMA_M], p_scales[LLAMA_NH][LLAMA_GM * LLAMA_M];
static uint8_t  O_hw[LLAMA_NH][LLAMA_M * LLAMA_H];
static uint32_t o_scales_dram[LLAMA_NH][64] __attribute__((aligned(32)));
static uint8_t  wo_scales_chunk[LLAMA_GH * YCHUNK];
// O's requantizer scales arrive as [M][GH]; the A-side window wants [GH][M] (a_off =
// group * M + row, gemmini.cc:1240). Phase 3 re-mvins O after a reset, so it cannot use the
// resident path that writes them transposed for free -- the host transposes them here.
static uint8_t  o_scales_a[LLAMA_GH * LLAMA_M];
// A B-side scale window is [K/32][N] with b_off = group * N + col. Slicing N columns out of
// a [GD][N_full] table is therefore a GATHER, not a contiguous run -- the rows are N_full
// apart. Same shape of copy the o_proj chunk below does.
static uint8_t  proj_scales_chunk[LLAMA_GD * PROJ_N];
static uint16_t Ychunk[LLAMA_M * YCHUNK];
static uint16_t Y_hw[LLAMA_M * LLAMA_D];
static uint16_t OUT_hw[LLAMA_M * LLAMA_D];
static uint32_t scale_sink[512] __attribute__((aligned(32)));
// Debug buffers.
static uint16_t Y_g[LLAMA_M * LLAMA_D];        // o_proj fed the GOLDEN O
static uint16_t o_bf16[LLAMA_M * LLAMA_H];     // P @ V before requant, for the first bad head
#define DBG_O_PRINT_MAX 16                     // bad O codes listed individually

uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  printf("TRAP cause=%d epc=%lx\n", (int) cause, (unsigned long) epc);
  tohost_exit(1337);
  return 0;
}

// ---------------------------------------------------------------------------------------------
// Movers.
// ---------------------------------------------------------------------------------------------
static void mvin_A(const uint8_t *A, int M, int K, uint32_t a_spad) {
  gemmini_config_ld(K * sizeof(uint8_t));
  int tiles_I = M / DIM, tiles_K = K / DIM;
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void *) (A + (size_t) i * DIM * K + (size_t) k * DIM),
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

// `accum` is loop_ws's ex_accumulate (rs1 bit 0): 0 OVERWRITES the output region, 1 adds into what
// is already there. Only the cross-head o_proj wants 1 -- everything else starts a fresh result.
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

// ---------------------------------------------------------------------------------------------
// Debug helpers.
// ---------------------------------------------------------------------------------------------
// o_proj: Y = sum_h O_h @ Wo_h, every head accumulated into one resident chunk -- exactly the
// original kernel's loop, parameterised on WHERE O and its [M][GH] scales come from, so the same
// code runs on the hardware's O and on the golden O.
static void oproj(const uint8_t *o_src, size_t o_stride, const uint8_t *os_src, size_t os_stride,
                  const uint8_t *WO_CODES, const uint8_t *WO_SCALES, uint16_t *Y,
                  uint64_t *t_mesh) {
  for (int c = 0; c < YCHUNKS; c++) {
    for (int h = 0; h < LLAMA_NH; h++) {
      for (int g = 0; g < LLAMA_GH; g++)
        memcpy(wo_scales_chunk + (size_t) g * YCHUNK,
               WO_SCALES + (size_t) (h * LLAMA_GH + g) * LLAMA_D + c * YCHUNK, YCHUNK);
      const uint8_t *osrc = os_src + (size_t) h * os_stride;         // [M][GH]
      for (int g = 0; g < LLAMA_GH; g++)
        for (int m = 0; m < LLAMA_M; m++)
          o_scales_a[(size_t) g * LLAMA_M + m] = osrc[(size_t) m * LLAMA_GH + g];
      uint64_t t0 = read_cycles();
      mvin_A(o_src + (size_t) h * o_stride, LLAMA_M, LLAMA_H, SPAD_OA);
      gemmini_mx_load_scales((uint64_t) o_scales_a, sizeof(o_scales_a), 0);
      gemmini_mx_load_scales((uint64_t) wo_scales_chunk, sizeof(wo_scales_chunk), 1);
      gemmini_fence();
      mvin_B(WO_CODES + (size_t) h * LLAMA_H * LLAMA_D, LLAMA_H, LLAMA_D, c * YCHUNK, YCHUNK,
             SPAD_WO);
      mesh_matmul(LLAMA_M, LLAMA_H, YCHUNK, SPAD_OA, SPAD_WO_ARG, SPAD_Y,
                  OUT_BF16, (uint64_t) scale_sink, 0, h > 0);
      *t_mesh += read_cycles() - t0;
    }
    mvout_bf16(Ychunk, SPAD_Y, LLAMA_M, YCHUNK);
    for (int m = 0; m < LLAMA_M; m++)
      memcpy(&Y[(size_t) m * LLAMA_D + c * YCHUNK], &Ychunk[(size_t) m * YCHUNK],
             YCHUNK * sizeof(uint16_t));
  }
}

// Which token rows of Y differ, and by how many columns. One bad O_h[m][j] reaches row m only.
static void y_rows(const char *tag, const uint16_t *Y, const uint16_t *G) {
  int rows = 0;
  printf("dbg   %s: rows with diffs (row:count):", tag);
  for (int m = 0; m < LLAMA_M; m++) {
    int n = mx_count_diff_u16(Y + (size_t) m * LLAMA_D, G + (size_t) m * LLAMA_D, LLAMA_D);
    if (n) { printf(" %d:%d", m, n); rows++; }
  }
  printf("%s  (%d of %d rows)\n", rows ? "" : " none", rows, LLAMA_M);
}

// Re-run head h's P @ V with a BF16 output and print the E8M0 block holding (m, j): the exact
// values the requantizer turned into codes. Operands and scales are re-loaded, since the requant
// pass's resident scale write may have overwritten the A-side window.
static void dump_pv_block(int h, int kv, int m, int j) {
  mvin_A(p_codes[h], LLAMA_M, LLAMA_M, H2_A_P);
  mvin_B(v_codes[kv], LLAMA_M, LLAMA_H, 0, LLAMA_H, H2_B_V);
  gemmini_mx_load_scales((uint64_t) p_scales[h], LLAMA_GM * LLAMA_M, 0);
  gemmini_mx_load_scales((uint64_t) v_scales[kv], LLAMA_GM * LLAMA_H, 1);
  gemmini_fence();
  mesh_matmul(LLAMA_M, LLAMA_M, LLAMA_H, H2_A_P, H2_B_V_ARG, H2_C_DBG,
              OUT_BF16, (uint64_t) scale_sink, 0, 0);
  mvout_bf16(o_bf16, H2_C_DBG, LLAMA_M, LLAMA_H);
  const int b = j / 32;
  printf("dbg   requant INPUT, head %d row %d block %d (cols %d..%d), BF16 bits, [*] = bad col:\n",
         h, m, b, b * 32, b * 32 + 31);
  for (int c = b * 32; c < b * 32 + 32; c++) {
    printf(" %s0x%04x", c == j ? "*" : "", o_bf16[(size_t) m * LLAMA_H + c]);
    if ((c & 7) == 7) printf("\n");
  }
}

#ifdef DGB_DUMP_PV_HEAD
// -DDGB_DUMP_PV_HEAD=h: print head h's P @ V tile EXACTLY as the device is given it -- A = P codes
// [M][M] + A-side scale window [M/32][M], B = V codes [M][H] + B-side window [M/32][H] -- and the
// BF16 result of running it (no requant). gen/gen_pv_repro.py turns the log into a ladder header,
// so the one tile can be re-run on RTL simulation on its own. Build on spike: its BF16 is the golden.
static void dump_arr(const char *name, const void *p, int n, int bytes) {
  printf("dump %s %d %d\n", name, n, bytes);
  for (int i = 0; i < n; i++) {
    if (bytes == 1) printf("%02x", ((const uint8_t *) p)[i]);
    else            printf("%04x", ((const uint16_t *) p)[i]);
    printf((i & 31) == 31 || i == n - 1 ? "\n" : " ");
  }
}

static void dump_pv_inputs(int h, int kv) {
  mvin_A(p_codes[h], LLAMA_M, LLAMA_M, H2_A_P);
  mvin_B(v_codes[kv], LLAMA_M, LLAMA_H, 0, LLAMA_H, H2_B_V);
  gemmini_mx_load_scales((uint64_t) p_scales[h], LLAMA_GM * LLAMA_M, 0);
  gemmini_mx_load_scales((uint64_t) v_scales[kv], LLAMA_GM * LLAMA_H, 1);
  gemmini_fence();
  mesh_matmul(LLAMA_M, LLAMA_M, LLAMA_H, H2_A_P, H2_B_V_ARG, H2_C_DBG,
              OUT_BF16, (uint64_t) scale_sink, 0, 0);
  mvout_bf16(o_bf16, H2_C_DBG, LLAMA_M, LLAMA_H);
  printf("dump-begin head %d kv %d M %d K %d N %d\n", h, kv, LLAMA_M, LLAMA_M, LLAMA_H);
  dump_arr("A_CODES",  p_codes[h],  LLAMA_M * LLAMA_M,  1);
  dump_arr("A_SCALES", p_scales[h], LLAMA_GM * LLAMA_M, 1);
  dump_arr("B_CODES",  v_codes[kv], LLAMA_M * LLAMA_H,  1);
  dump_arr("B_SCALES", v_scales[kv], LLAMA_GM * LLAMA_H, 1);
  dump_arr("C_BF16",   o_bf16,      LLAMA_M * LLAMA_H,  2);
  printf("dump-end\n");
}
#endif

// Re-establish what a reset destroys: Xn as the resident A operand and its scale window.
static void load_xn(void) {
  mvin_A(xn_codes, LLAMA_M, LLAMA_D, SPAD_XN);
  gemmini_mx_load_scales((uint64_t) xn_scales, sizeof(xn_scales), 0);
  gemmini_fence();
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  const uint16_t *H_PRE      = LLAMA_AT(LLAMA_OFF_H_PRE, uint16_t);
  const uint16_t *W_IN_LN    = LLAMA_AT(LLAMA_OFF_W_IN_LN, uint16_t);
  const uint32_t *ROPE_COS   = LLAMA_AT(LLAMA_OFF_ROPE_COS, uint32_t);
  const uint32_t *ROPE_SIN   = LLAMA_AT(LLAMA_OFF_ROPE_SIN, uint32_t);
  const uint8_t  *XN_CODES   = LLAMA_AT(LLAMA_OFF_XN_CODES, uint8_t);
  const uint8_t  *XN_SCALES  = LLAMA_AT(LLAMA_OFF_XN_SCALES, uint8_t);
  const uint8_t  *WQ_CODES   = LLAMA_AT(LLAMA_OFF_WQ_CODES, uint8_t);
  const uint8_t  *WQ_SCALES  = LLAMA_AT(LLAMA_OFF_WQ_SCALES, uint8_t);
  const uint8_t  *WK_CODES   = LLAMA_AT(LLAMA_OFF_WK_CODES, uint8_t);
  const uint8_t  *WK_SCALES  = LLAMA_AT(LLAMA_OFF_WK_SCALES, uint8_t);
  const uint8_t  *WV_CODES   = LLAMA_AT(LLAMA_OFF_WV_CODES, uint8_t);
  const uint8_t  *WV_SCALES  = LLAMA_AT(LLAMA_OFF_WV_SCALES, uint8_t);
  const uint8_t  *WO_CODES   = LLAMA_AT(LLAMA_OFF_WO_CODES, uint8_t);
  const uint8_t  *WO_SCALES  = LLAMA_AT(LLAMA_OFF_WO_SCALES, uint8_t);
  const uint16_t *Q_OUT      = LLAMA_AT(LLAMA_OFF_Q_OUT, uint16_t);
  const uint16_t *K_OUT      = LLAMA_AT(LLAMA_OFF_K_OUT, uint16_t);
  const uint16_t *V_OUT      = LLAMA_AT(LLAMA_OFF_V_OUT, uint16_t);
  const uint8_t  *G_Q_CODES  = LLAMA_AT(LLAMA_OFF_Q_CODES, uint8_t);
  const uint8_t  *G_KT_CODES = LLAMA_AT(LLAMA_OFF_KT_CODES, uint8_t);
  const uint8_t  *G_V_CODES  = LLAMA_AT(LLAMA_OFF_V_CODES, uint8_t);
  const uint16_t *S_OUT      = LLAMA_AT(LLAMA_OFF_S_OUT, uint16_t);
  const uint8_t  *G_P_CODES  = LLAMA_AT(LLAMA_OFF_P_CODES, uint8_t);
  const uint8_t  *O_OUT      = LLAMA_AT(LLAMA_OFF_O_CODES, uint8_t);
  const uint8_t  *O_SCALES   = LLAMA_AT(LLAMA_OFF_O_SCALES, uint8_t);
  const uint16_t *Y_OUT      = LLAMA_AT(LLAMA_OFF_Y_OUT, uint16_t);
  const uint16_t *REF_ATTN   = LLAMA_AT(LLAMA_OFF_REF_ATTN, uint16_t);
  const uint16_t *ATTN_TORCH = LLAMA_AT(LLAMA_OFF_ATTN_TORCH, uint16_t);

  printf("llama attention FULL: M=%d D=%d head_dim=%d heads=%d kv_heads=%d (fp8 e4m3 + E8M0)\n",
         LLAMA_M, LLAMA_D, LLAMA_H, LLAMA_NH, LLAMA_NKV);
  printf("plan  spad %d rows: proj N=%d | %d heads | o_proj %d chunks of %d\n",
         SPAD_ROWS, PROJ_N, LLAMA_NH, YCHUNKS, YCHUNK);

  gemmini_flush(0);

  uint64_t t_host = 0, t_mesh = 0, t0;

  // ============ host: RMSNorm over the full D ============
  t0 = read_cycles();
  mx_rmsnorm(H_PRE, W_IN_LN, LLAMA_M, LLAMA_D, LLAMA_RMS_EPS, xn_f);
  mx_quantize_rows(xn_f, LLAMA_M, LLAMA_D, xn_codes, xn_scales);
  t_host += read_cycles() - t0;
  printf("host  rmsnorm+quant: codes differ %d/%d, scales differ %d/%d vs golden\n",
         mx_count_diff_u8(xn_codes, XN_CODES, LLAMA_M * LLAMA_D), LLAMA_M * LLAMA_D,
         mx_count_diff_u8(xn_scales, XN_SCALES, LLAMA_GD * LLAMA_M), LLAMA_GD * LLAMA_M);

  // ============ mesh: Q, K, V -- Xn resident, N-chunked at the width the plan picked ============
  const uint8_t *wc[3]  = { WQ_CODES, WK_CODES, WV_CODES };
  const uint8_t *ws[3]  = { WQ_SCALES, WK_SCALES, WV_SCALES };
  uint16_t *dst[3]      = { Q_hw, K_hw, V_hw };
  const uint16_t *gold[3] = { Q_OUT, K_OUT, V_OUT };
  const int wid[3]      = { LLAMA_QD, LLAMA_KVD, LLAMA_KVD };
  const char *pn[3]     = { "Q", "K", "V" };

  t0 = read_cycles();
  load_xn();
  t_mesh += read_cycles() - t0;

  int proj_diff = 0;
  for (int s = 0; s < 3; s++) {
    // A projection narrower than the planned chunk just uses its own width.
    const int nc = wid[s] < PROJ_N ? wid[s] : PROJ_N;
    for (int c = 0; c < wid[s] / nc; c++) {
      for (int g = 0; g < LLAMA_GD; g++)
        memcpy(proj_scales_chunk + (size_t) g * nc,
               ws[s] + (size_t) g * wid[s] + (size_t) c * nc, nc);
      t0 = read_cycles();
      gemmini_mx_load_scales((uint64_t) proj_scales_chunk, LLAMA_GD * nc, 1);
      gemmini_fence();
      mvin_B(wc[s], LLAMA_D, wid[s], c * nc, nc, SPAD_ROWS - ROWS8(LLAMA_D, nc));
      mesh_matmul(LLAMA_M, LLAMA_D, nc, SPAD_XN, SPAD_ROWS, PROJ_C,
                  OUT_BF16, (uint64_t) scale_sink, 0, 0);
      t_mesh += read_cycles() - t0;
      mvout_bf16(chunk16, PROJ_C, LLAMA_M, nc);
      for (int m = 0; m < LLAMA_M; m++)
        memcpy(&dst[s][(size_t) m * wid[s] + c * nc], &chunk16[(size_t) m * nc],
               nc * sizeof(uint16_t));
    }
    int d = mx_count_diff_u16(dst[s], gold[s], LLAMA_M * wid[s]);
    proj_diff += d;
    printf("mesh  %s = Xn @ W%s : %d/%d differ from golden (%d chunks of %d)\n",
           pn[s], pn[s], d, LLAMA_M * wid[s], wid[s] / nc, nc);
  }

  // ============ host: RoPE per head, the K transpose, V as a B operand ============
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
    printf("host  RoPE(%d heads) + K^T: Q codes %d/%d, K^T %d/%d, V %d/%d differ\n",
           LLAMA_NH, qd, LLAMA_NH * LLAMA_M * LLAMA_H, kd, LLAMA_NKV * LLAMA_H * LLAMA_M,
           vd, LLAMA_NKV * LLAMA_M * LLAMA_H);
  }

  // ============ mesh + host, per head: S, softmax, O = P@V requantized into the spad ============
  int s_diff = 0, o_diff = 0, os_diff = 0, dbg_o_listed = 0;
  for (int h = 0; h < LLAMA_NH; h++) {
    {
      const int kv = h / LLAMA_PER;
      const uint32_t c_s = H2_C_S, c_o = H2_C_O;

      t0 = read_cycles();
      mvin_A(q_codes[h], LLAMA_M, LLAMA_H, H2_A_Q);
      mvin_B(kt_codes[kv], LLAMA_H, LLAMA_M, 0, LLAMA_M, H2_B_KT);
      gemmini_mx_load_scales((uint64_t) q_scales[h], LLAMA_GH * LLAMA_M, 0);
      gemmini_mx_load_scales((uint64_t) kt_scales[kv], LLAMA_GH * LLAMA_M, 1);
      gemmini_fence();
      mesh_matmul(LLAMA_M, LLAMA_H, LLAMA_M, H2_A_Q, H2_B_KT_ARG, c_s,
                  OUT_BF16, (uint64_t) scale_sink, 0, 0);
      t_mesh += read_cycles() - t0;
      mvout_bf16(S_hw[h], c_s, LLAMA_M, LLAMA_M);
      s_diff += mx_count_diff_u16(S_hw[h], S_OUT + (size_t) h * LLAMA_M * LLAMA_M,
                                  LLAMA_M * LLAMA_M);

      t0 = read_cycles();
      mx_softmax_causal(S_hw[h], LLAMA_M, 1.0f / sqrtf((float) LLAMA_H), p_f);
      mx_quantize_rows(p_f, LLAMA_M, LLAMA_M, p_codes[h], p_scales[h]);
      t_host += read_cycles() - t0;

      t0 = read_cycles();
      mvin_A(p_codes[h], LLAMA_M, LLAMA_M, H2_A_P);
      mvin_B(v_codes[kv], LLAMA_M, LLAMA_H, 0, LLAMA_H, H2_B_V);
      gemmini_mx_load_scales((uint64_t) p_scales[h], LLAMA_GM * LLAMA_M, 0);
      gemmini_mx_load_scales((uint64_t) v_scales[kv], LLAMA_GM * LLAMA_H, 1);
      gemmini_fence();
      mesh_matmul(LLAMA_M, LLAMA_M, LLAMA_H, H2_A_P, H2_B_V_ARG, c_o,
                  OUT_FP8, (uint64_t) o_scales_dram[h], 1, 0);
      t_mesh += read_cycles() - t0;
      mvout_detile(O_hw[h], c_o, LLAMA_M, LLAMA_H);
      o_diff += mx_count_diff_u8(O_hw[h], O_OUT + (size_t) h * LLAMA_M * LLAMA_H,
                                 LLAMA_M * LLAMA_H);
      os_diff += mx_count_diff_u8((const uint8_t *) o_scales_dram[h],
                                  O_SCALES + (size_t) h * LLAMA_M * LLAMA_GH,
                                  LLAMA_M * LLAMA_GH);

      // ---- debug: list every bad O code; dump the requant input for the first one ----
      for (int i = 0; i < LLAMA_M * LLAMA_H; i++) {
        const uint8_t got = O_hw[h][i], exp = O_OUT[(size_t) h * LLAMA_M * LLAMA_H + i];
        if (got == exp) continue;
        const int m = i / LLAMA_H, j = i % LLAMA_H, b = j / 32;
        const uint8_t sg = ((const uint8_t *) o_scales_dram[h])[m * LLAMA_GH + b];
        const uint8_t se = O_SCALES[(size_t) h * LLAMA_M * LLAMA_GH + m * LLAMA_GH + b];
        if (dbg_o_listed < DBG_O_PRINT_MAX)
          printf("dbg   bad O: head %d row %d col %d  code got 0x%02x exp 0x%02x  "
                 "E8M0[blk %d] got 0x%02x exp 0x%02x\n", h, m, j, got, exp, b, sg, se);
        if (dbg_o_listed == 0) dump_pv_block(h, kv, m, j);
        dbg_o_listed++;
      }
#ifdef DGB_DUMP_PV_HEAD
      if (h == DGB_DUMP_PV_HEAD) dump_pv_inputs(h, kv);
#endif
    }
  }
  {
    int pd = 0;
    for (int h = 0; h < LLAMA_NH; h++)
      pd += mx_count_diff_u8(p_codes[h], G_P_CODES + (size_t) h * LLAMA_M * LLAMA_M,
                             LLAMA_M * LLAMA_M);
    printf("mesh  S = Q @ K^T : %d/%d differ   host softmax: P codes %d/%d differ\n",
           s_diff, LLAMA_NH * LLAMA_M * LLAMA_M, pd, LLAMA_NH * LLAMA_M * LLAMA_M);
  }
  printf("mesh  O = P @ V   : %d/%d codes, %d/%d scales differ (requant -> spad, %d heads)\n",
         o_diff, LLAMA_NH * LLAMA_M * LLAMA_H, os_diff, LLAMA_NH * LLAMA_M * LLAMA_GH, LLAMA_NH);

  // ============ mesh: o_proj, ACCUMULATING every head into one resident Y chunk ============
  // Y_c = sum_h O_h @ Wo[H*h : H*h+H, c-slice]. Every head targets the same C region and mx_smem
  // adds -- so the concatenated [M, QD] operand is never built. The chunk is as wide as the
  // scratchpad allows, and only one is live at a time, which is what keeps this in budget.
  oproj((const uint8_t *) O_hw, LLAMA_M * LLAMA_H, (const uint8_t *) o_scales_dram,
        sizeof(o_scales_dram[0]), WO_CODES, WO_SCALES, Y_hw, &t_mesh);
  int y_d = mx_count_diff_u16(Y_hw, Y_OUT, LLAMA_M * LLAMA_D);
  printf("mesh  Y = sum_h O_h @ Wo_h : %d/%d differ (%d heads accumulated, %d chunks of %d)\n",
         y_d, LLAMA_M * LLAMA_D, LLAMA_NH, YCHUNKS, YCHUNK);

  // ---- debug: where the Y diffs are, and o_proj again on the GOLDEN O ----
  if (dbg_o_listed > DBG_O_PRINT_MAX)
    printf("dbg   (%d bad O codes in all; first %d listed)\n", dbg_o_listed, DBG_O_PRINT_MAX);
  y_rows("Y (hardware O)", Y_hw, Y_OUT);
  {
    uint64_t t_dbg = 0;
    oproj(O_OUT, LLAMA_M * LLAMA_H, O_SCALES, LLAMA_M * LLAMA_GH, WO_CODES, WO_SCALES, Y_g,
          &t_dbg);
    int yg_d = mx_count_diff_u16(Y_g, Y_OUT, LLAMA_M * LLAMA_D);
    printf("dbg   Y (GOLDEN O) = sum_h O_h @ Wo_h : %d/%d differ -- %s\n", yg_d,
           LLAMA_M * LLAMA_D,
           yg_d != 0 ? "o_proj/accumulation itself is wrong on this bitstream"
           : y_d != 0 ? "cross-head accumulation is EXACT; the Y diffs come from the bad O"
                      : "cross-head accumulation is EXACT");
    y_rows("Y (golden O)", Y_g, Y_OUT);
  }

  // ============ host: the residual ============
  t0 = read_cycles();
  for (int i = 0; i < LLAMA_M * LLAMA_D; i++)
    OUT_hw[i] = mx_f32_to_bf16_rne(mx_bf16_to_f32(H_PRE[i]) + mx_bf16_to_f32(Y_hw[i]));
  t_host += read_cycles() - t0;

  printf("grade attention out vs fp32 reference      : rel_fro %d ppm\n",
         MX_PPM(mx_rel_fro_bf16(Y_hw, REF_ATTN, LLAMA_M * LLAMA_D)));
  printf("grade attention out vs THE MODEL's own out : rel_fro %d ppm\n",
         MX_PPM(mx_rel_fro_bf16(Y_hw, ATTN_TORCH, LLAMA_M * LLAMA_D)));
  printf("cycles mesh %d, host %d\n", (int) t_mesh, (int) t_host);

  int exact = proj_diff + s_diff + o_diff + os_diff + y_d;
  if (exact == 0)
    printf("llama FULL attention test PASSED (all %d heads, every mesh matmul bit-exact; "
           "o_proj accumulated across heads in smem).\n", LLAMA_NH);
  else
    printf("llama FULL attention test FAILED: %d mesh element(s) differ from golden.\n", exact);

#ifndef BAREMETAL
  exit(exact != 0);
#else
  return exact != 0;
#endif
}
