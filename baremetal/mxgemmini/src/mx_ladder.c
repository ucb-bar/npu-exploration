// The MX bisection ladder -- shared driver. One rung per ELF; see `src/mxl*.c`, which are two
// lines each, and `gen/gen_mx_ladder.py`, which carries the reasoning for every rung.
//
// WHY. `llama_attention` PASSES on spike and fails on the FPGA with EVERY mesh stage wrong,
// including the first projection, while `matmul_tiled_fp8_64x64_chain` and the 128x128 fp8 test
// PASS on the same bitstream. So the problem is not "MX fp8 on RTL". It is something llama's
// matmuls do that the ISA tests do not, and the deltas are few enough to enumerate:
//
//   mxl0  64x64x64, square, shallow      -- the ISA shape, through llama's own helpers
//   mxl1  M=32  -> I=2, J=4              -- a NON-SQUARE tile grid
//   mxl2  K=256 -> TK=16                 -- a deeper reduction
//   mxl3  K=1024 -> TK=64, 2 KB B scales
//   mxl4  K=2048 -> TK=128, 4 KB B scales   == llama's Q = Xn @ Wq exactly
//   mxl5  the same matmul in 2 accumulating K-tiles  -- ex_accumulate on RTL
//   mxl6  two matmuls into one C_spad                -- output-region reuse / smem clear
//   mxl7  requant -> spad, resident scales, re-read  -- the P@V -> o_proj seam
//   mxl8  a strided A mvin out of a [32][2048] array
//   mxl9  no mesh at all -- mx_host.h's fp32 glue against its golden
//
// Run them in order on the FPGA. The first one that FAILS names the feature; everything below it
// is then a consequence, not a separate bug.
//
// Every golden comes from `fp8_matmul_model.tiled_matmul_hwlike`, the same model that generates
// the llama headers, so a rung PASSING is the same kind of statement `llama_attention` makes.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"

// THE LADDER IS A DIM=16 ARTIFACT. Its goldens were computed at the dim16 precision ramp and every
// scratchpad address below counts 16-byte rows. `gemmini_params.h` is shared with the ISA suite and
// gets flipped between the dim16 and dim32 bitstreams (commits abd690d / 22be6a0 did exactly that),
// so pin DIM here -- and WARN, rather than silently disagree, which is what the llama kernels do.
#define LADDER_DIM 16
#if DIM != LADDER_DIM
#warning "gemmini_params.h DIM != 16: this ELF pins DIM=16 to match its goldens. Correct for a \
dim16 bitstream; meaningless on a dim32 one, which needs regenerated data."
#endif
#undef DIM
#define DIM LADDER_DIM

// AND THE SAME FOR THE SCRATCHPAD GEOMETRY, for a reason learned the hard way mid-bisection:
// `BANK_ROWS` flips with whatever bitstream is being built (4096 for dim16, 2048 for dim32), and
// unlike `DIM` nothing here overrode it -- so a flip retunes every scratchpad map in this file, and
// mxl4 stopped BUILDING (its LAD_REQUIRE(fits) fired) while the RTL results it produced were still
// being read. That guard doing its job is the only reason this was a build error rather than a
// silently aliased run. Pin the geometry to the bitstream the ladder targets and warn on a
// mismatch; `-DLADDER_BANK_ROWS=2048` retargets it.
#ifndef LADDER_BANK_ROWS
#define LADDER_BANK_ROWS 4096
#endif
#if BANK_ROWS != LADDER_BANK_ROWS
#warning "gemmini_params.h BANK_ROWS differs from the ladder's: pinning the ladder's value. \
Correct if that header is set for a different bitstream; -DLADDER_BANK_ROWS=N to retarget."
#endif
#undef BANK_ROWS
#define BANK_ROWS LADDER_BANK_ROWS

#include "mx_mesh.h"
#include "mx_host.h"

#define KIND_PLAIN   0
#define KIND_KTILE   1
#define KIND_REUSE   2
#define KIND_REQUANT 3
#define KIND_STRIDED 4
#define KIND_HOST    5

#ifndef LADDER_HEADER
#error "define LADDER_HEADER and LADDER_KIND, then include this file (see src/mxl0.c)"
#endif
#include LADDER_HEADER

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

// Radiance drives gemmini via an MMIO command mimic; the standalone rocket config (MX_ROCKET) and
// spike are real RoCC, so keep gemmini.h's direct-RoCC macro there.
#if !defined(SPIKE_SIM) && !defined(MX_ROCKET)
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#endif

// A region that overflows the scratchpad ALIASES silently and produces plausible numbers, which on
// a bisection ladder would be worse than useless. Each check names a live range.
#define LAD_REQUIRE(name, cond) typedef char lad_spad_##name[(cond) ? 1 : -1]

// riscv-tests' handle_trap is weak and exits 1337 with no cause, which is indistinguishable from a
// stale libgemmini.so. One diagnosable line instead.
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  printf("TRAP cause=%d epc=%lx\n", (int) cause, (unsigned long) epc);
  tohost_exit(1337);
  return 0;
}

static void banner(void) {
  printf("=== %s ===\n", LAD_NAME);
  printf("why   %s\n", LAD_WHY);
  printf("cfg   DIM=%d  spad=%d rows (BANK_NUM %d x BANK_ROWS %d)\n",
         DIM, SPAD_TOP, BANK_NUM, BANK_ROWS);
}

static int verdict(int errors) {
  if (errors == 0) printf("%s PASSED\n", LAD_NAME);
  else             printf("%s FAILED: %d element(s) differ from golden\n", LAD_NAME, errors);
#ifndef BAREMETAL
  exit(errors != 0);
#else
  return errors != 0;
#endif
}

// Print the first few mismatches. On a bisection ladder the PATTERN is the diagnosis -- all-wrong
// versus one-tile-wrong versus every-other-column-wrong are three different bugs -- and a bare
// count cannot tell them apart.
static void show_u16(const uint16_t *got, const uint16_t *exp, int M, int N, int limit) {
  int shown = 0;
  for (int i = 0; i < M * N && shown < limit; i++)
    if (got[i] != exp[i]) {
      printf("  diff @(%d,%d) hw=0x%04x golden=0x%04x\n", i / N, i % N, got[i], exp[i]);
      shown++;
    }
}

static void show_u8(const uint8_t *got, const uint8_t *exp, int M, int N, int limit) {
  int shown = 0;
  for (int i = 0; i < M * N && shown < limit; i++)
    if (got[i] != exp[i]) {
      printf("  diff @(%d,%d) hw=0x%02x golden=0x%02x\n", i / N, i % N, got[i], exp[i]);
      shown++;
    }
}

static uint32_t scale_sink[512] __attribute__((aligned(32)));

// =================================================================================================
#if LADDER_KIND == KIND_PLAIN || LADDER_KIND == KIND_KTILE || LADDER_KIND == KIND_STRIDED

// PLAIN, KTILE and STRIDED are ONE driver. A plain rung is the K-tile loop with LAD_KTILES = 1,
// which is exactly how `llama_attention.c` degenerates on a 16384-row scratchpad -- so the shared
// path is the honest one, not a convenience.
#ifndef LAD_KTILES
#define LAD_KTILES 1
#endif
#define LAD_KT   (LAD_K / LAD_KTILES)
#define LAD_KGRP (LAD_KT / 32)

#define SPAD_A    0
#define SPAD_B    (SPAD_A + ROWS8(LAD_M, LAD_KT))
#define SPAD_BARG (SPAD_B + ROWS8(LAD_KT, LAD_N))
#define SPAD_C    SPAD_BARG

LAD_REQUIRE(ktiles_divide_k, LAD_KT * LAD_KTILES == LAD_K);
LAD_REQUIRE(ktile_is_blocked, (LAD_KT % 32) == 0);
LAD_REQUIRE(fits, SPAD_C + ROWS16(LAD_M, LAD_N) <= SPAD_TOP);

static uint16_t C_hw[LAD_M * LAD_N];

int main(void) {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  banner();
  printf("plan  M=%d K=%d N=%d -> I=%d J=%d TK=%d | %d K-tile(s) of %d | A-scales %d B "
         "B-scales %d B\n",
         LAD_M, LAD_K, LAD_N, LAD_M / DIM, LAD_N / DIM, LAD_KT / DIM,
         LAD_KTILES, LAD_KT, LAD_KGRP * LAD_M, LAD_KGRP * LAD_N);
  printf("spad  A %d..%d  B %d..%d (arg %d)  C %d..%d\n",
         SPAD_A, SPAD_B - 1, SPAD_B, SPAD_BARG - 1, SPAD_BARG,
         SPAD_C, SPAD_C + ROWS16(LAD_M, LAD_N) - 1);

  gemmini_flush(0);
  uint64_t t0 = read_cycles();

  for (int kt = 0; kt < LAD_KTILES; kt++) {
#if LADDER_KIND == KIND_STRIDED
    // A is a column slice of a wider array: the DMA walks LAD_A_STRIDE bytes per tile row.
    mvin_A_strided(&LAD_A_WIDE[0][LAD_A_COL] + (size_t) kt * LAD_KT,
                   LAD_M, LAD_KT, LAD_A_STRIDE, SPAD_A);
#else
    mvin_A_strided((const uint8_t *) LAD_A + (size_t) kt * LAD_KT,
                   LAD_M, LAD_KT, LAD_K, SPAD_A);
#endif
    // Both scale windows take a CONTIGUOUS slice: the A window is [K/32][M] and the B window
    // [K/32][N], so a K-tile is whole rows of each and no gather is needed.
    gemmini_mx_load_scales((uint64_t) &LAD_A_SCALES[kt * LAD_KGRP][0], LAD_KGRP * LAD_M, 0);
    gemmini_mx_load_scales((uint64_t) &LAD_B_SCALES[kt * LAD_KGRP][0], LAD_KGRP * LAD_N, 1);
    gemmini_fence();
    mvin_B((const uint8_t *) LAD_B + (size_t) kt * LAD_KT * LAD_N, LAD_KT, LAD_N, 0, LAD_N, SPAD_B);
    // kt > 0 accumulates into the region the previous tile wrote; kt == 0 overwrites it.
    mesh_matmul(LAD_M, LAD_KT, LAD_N, SPAD_A, SPAD_BARG, SPAD_C,
                OUT_BF16, (uint64_t) scale_sink, 0, kt > 0);
  }
  uint64_t cyc = read_cycles() - t0;

  mvout_bf16(C_hw, SPAD_C, LAD_M, LAD_N);
  int d = mx_count_diff_u16(C_hw, (const uint16_t *) LAD_C_BF16, LAD_M * LAD_N);
  printf("mesh  C = A @ B : %d/%d differ from golden   (%d cycles)\n",
         d, LAD_M * LAD_N, (int) cyc);
  if (d) show_u16(C_hw, (const uint16_t *) LAD_C_BF16, LAD_M, LAD_N, 8);
#if LADDER_KIND == KIND_KTILE
  if (d)
    printf("  NOTE this is mxl4's matmul split into %d accumulating K-tiles and graded against "
           "mxl4's golden. If mxl4 PASSED and this failed, the RTL is not honouring loop_ws's "
           "ex_accumulate bit (rs1 bit 0).\n", LAD_KTILES);
#endif
  return verdict(d);
}

// =================================================================================================
#elif LADDER_KIND == KIND_REUSE

#define SPAD_A    0
#define SPAD_B    (SPAD_A + ROWS8(LAD_M, LAD_K))
#define SPAD_BARG (SPAD_B + ROWS8(LAD_K, LAD_N))
#define SPAD_C    SPAD_BARG

LAD_REQUIRE(fits, SPAD_C + ROWS16(LAD_M, LAD_N) <= SPAD_TOP);

static uint16_t C1_hw[LAD_M * LAD_N], C2_hw[LAD_M * LAD_N];

static void one(const uint8_t *A, const uint8_t *B, const uint8_t *asc, const uint8_t *bsc,
                uint16_t *out) {
  mvin_A(A, LAD_M, LAD_K, SPAD_A);
  gemmini_mx_load_scales((uint64_t) asc, LAD_GK * LAD_M, 0);
  gemmini_mx_load_scales((uint64_t) bsc, LAD_GK * LAD_N, 1);
  gemmini_fence();
  mvin_B(B, LAD_K, LAD_N, 0, LAD_N, SPAD_B);
  // accumulate = 0 on BOTH: each matmul must OVERWRITE the region, not add to what it holds.
  mesh_matmul(LAD_M, LAD_K, LAD_N, SPAD_A, SPAD_BARG, SPAD_C,
              OUT_BF16, (uint64_t) scale_sink, 0, 0);
  mvout_bf16(out, SPAD_C, LAD_M, LAD_N);
}

int main(void) {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  banner();
  printf("plan  two DISTINCT %dx%dx%d matmuls into the SAME C_spad %d, both ex_accumulate=0\n",
         LAD_M, LAD_K, LAD_N, SPAD_C);
  gemmini_flush(0);

  one((const uint8_t *) LAD_A, (const uint8_t *) LAD_B,
      (const uint8_t *) LAD_A_SCALES, (const uint8_t *) LAD_B_SCALES, C1_hw);
  int d1 = mx_count_diff_u16(C1_hw, (const uint16_t *) LAD_C_BF16, LAD_M * LAD_N);
  printf("mesh  #1 into C_spad : %d/%d differ from golden\n", d1, LAD_M * LAD_N);

  one((const uint8_t *) LAD2_A, (const uint8_t *) LAD2_B,
      (const uint8_t *) LAD2_A_SCALES, (const uint8_t *) LAD2_B_SCALES, C2_hw);
  int d2 = mx_count_diff_u16(C2_hw, (const uint16_t *) LAD2_C_BF16, LAD_M * LAD_N);
  printf("mesh  #2 into C_spad : %d/%d differ from golden\n", d2, LAD_M * LAD_N);

  if (d2) {
    // The specific way this fails is diagnostic, so test the hypothesis rather than just report
    // the count: if the region was never cleared, #2's result is #1's plus #2's. Checked at
    // runtime against a bf16 sum of the two goldens -- an approximation of the hardware's own
    // bf16 shadow accumulate, so a HIGH match rate is the signature, not necessarily 100%.
    int as_sum = 0;
    for (int i = 0; i < LAD_M * LAD_N; i++) {
      uint16_t s = mx_f32_to_bf16_rne(mx_bf16_to_f32(((const uint16_t *) LAD_C_BF16)[i])
                                      + mx_bf16_to_f32(((const uint16_t *) LAD2_C_BF16)[i]));
      if (C2_hw[i] == s) as_sum++;
    }
    printf("  of %d outputs, %d match bf16(golden#1 + golden#2) -- a high count means the RTL "
           "never cleared its shadow accumulator, i.e. ex_accumulate=0 does not overwrite\n",
           LAD_M * LAD_N, as_sum);
    show_u16(C2_hw, (const uint16_t *) LAD2_C_BF16, LAD_M, LAD_N, 8);
  }
  return verdict(d1 + d2);
}

// =================================================================================================
#elif LADDER_KIND == KIND_REQUANT

#define SPAD_O     0                                      // LIVE through the second matmul
#define SPAD_A     (SPAD_O + ROWS8(LAD_M, LAD_N))
#define SPAD_B     (SPAD_A + ROWS8(LAD_M, LAD_K))
#define SPAD_BARG  (SPAD_B + ROWS8(LAD_K, LAD_N))
#define SPAD_W2    SPAD_A                                 // A and B are dead once O exists
#define SPAD_W2ARG (SPAD_W2 + ROWS8(LAD_N, LAD_N2))
#define SPAD_Y     SPAD_W2ARG

LAD_REQUIRE(o_fits, SPAD_BARG <= SPAD_TOP);
LAD_REQUIRE(y_fits, SPAD_Y + ROWS16(LAD_M, LAD_N2) <= SPAD_TOP);

static uint8_t  O_hw[LAD_M * LAD_N], O_scratch[LAD_M * LAD_N];
static uint32_t o_scales_dram[512] __attribute__((aligned(32)));
static uint16_t Y_hw[LAD_M * LAD_N2];

int main(void) {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  banner();
  printf("plan  O = A @ B [%d,%d]x[%d,%d] -> FP8 requant TILED into spad %d, scales resident; "
         "then Y = O @ W2 [%d,%d]x[%d,%d] reading O IN PLACE\n",
         LAD_M, LAD_K, LAD_K, LAD_N, SPAD_O, LAD_M, LAD_N, LAD_N, LAD_N2);
  gemmini_flush(0);

  mvin_A((const uint8_t *) LAD_A, LAD_M, LAD_K, SPAD_A);
  gemmini_mx_load_scales((uint64_t) LAD_A_SCALES, LAD_GK * LAD_M, 0);
  gemmini_mx_load_scales((uint64_t) LAD_B_SCALES, LAD_GK * LAD_N, 1);
  gemmini_fence();
  mvin_B((const uint8_t *) LAD_B, LAD_K, LAD_N, 0, LAD_N, SPAD_B);
  mesh_matmul(LAD_M, LAD_K, LAD_N, SPAD_A, SPAD_BARG, SPAD_O,
              OUT_FP8, (uint64_t) o_scales_dram, 1, 0);

  // Read O back WITHOUT disturbing it, and check the scales the requantizer wrote to DRAM.
  mvout_detile(O_hw, O_scratch, SPAD_O, LAD_M, LAD_N);
  int od = mx_count_diff_u8(O_hw, (const uint8_t *) LAD_O_CODES, LAD_M * LAD_N);
  int os = mx_count_diff_u8((const uint8_t *) o_scales_dram, (const uint8_t *) LAD_O_SCALES,
                            LAD_M * LAD_GN);
  printf("mesh  O = A @ B requant->spad : %d/%d codes, %d/%d scales differ  (I=%d J=%d Kt=%d)\n",
         od, LAD_M * LAD_N, os, LAD_M * LAD_GN,
         LAD_M / DIM, LAD_N / DIM, LAD_K / DIM);
  if (od) show_u8(O_hw, (const uint8_t *) LAD_O_CODES, LAD_M, LAD_N, 8);

  // The E8M0 output scales are only M*GN bytes, so when they disagree, DUMP THEM ALL. A count
  // cannot distinguish a transposed write from a wrong row stride from an off-by-one, and those
  // are different bugs; the layout is small enough that the pattern is readable directly.
  if (os) {
    const uint8_t *hw = (const uint8_t *) o_scales_dram;
    const uint8_t *gd = (const uint8_t *) LAD_O_SCALES;
    printf("  O scales, expected [M][GN] = [%d][%d] (hw | golden, '.' = equal):\n",
           LAD_M, LAD_GN);
    for (int m = 0; m < LAD_M; m++) {
      printf("   row %2d: ", m);
      for (int g = 0; g < LAD_GN; g++) printf("%02x ", hw[m * LAD_GN + g]);
      printf("| ");
      for (int g = 0; g < LAD_GN; g++) {
        int k = m * LAD_GN + g;
        if (hw[k] == gd[k]) printf(" . "); else printf("%02x ", gd[k]);
      }
      printf("\n");
    }
    // Two layout hypotheses, tested rather than eyeballed. TRANSPOSED: the requantizer wrote
    // [GN][M] (which is the layout the A-side scale WINDOW wants, so confusing the two is a real
    // and already-recorded hazard -- planning/llama_layer_hw_plan.md 10.4). PITCHED: it wrote
    // [M][GN] but strided rows by J*GN instead of GN, i.e. it took the row pitch from the output
    // TILE count rather than the block count.
    int t = 0, p = 0, n = LAD_M * LAD_GN;
    for (int m = 0; m < LAD_M; m++)
      for (int g = 0; g < LAD_GN; g++) {
        if (hw[g * LAD_M + m] == gd[m * LAD_GN + g]) t++;
        int k = m * (LAD_N / DIM) * LAD_GN + g;
        if (k < 512 * 4 && hw[k] == gd[m * LAD_GN + g]) p++;
      }
    printf("  layout: %d/%d match a TRANSPOSED [GN][M] write, %d/%d match an [M][GN] write with a "
           "J*GN row pitch\n", t, n, p, n);
  }

  // No A mvin and no A-scale load: the codes are already resident at SPAD_O in the operand-A
  // tiled layout, and the requantizer wrote their E8M0 bytes into the act-scale window transposed
  // ([N/32][M], a_off = group * M + row), which is exactly what this matmul indexes. The Y check
  // is what proves the SCALE half -- the code half is proven by the mvout_detile above.
  gemmini_mx_load_scales((uint64_t) LAD_W2_SCALES, LAD_GN * LAD_N2, 1);
  gemmini_fence();
  mvin_B((const uint8_t *) LAD_W2, LAD_N, LAD_N2, 0, LAD_N2, SPAD_W2);
  mesh_matmul(LAD_M, LAD_N, LAD_N2, SPAD_O, SPAD_W2ARG, SPAD_Y,
              OUT_BF16, (uint64_t) scale_sink, 0, 0);
  mvout_bf16(Y_hw, SPAD_Y, LAD_M, LAD_N2);
  int yd = mx_count_diff_u16(Y_hw, (const uint16_t *) LAD_Y_BF16, LAD_M * LAD_N2);
  printf("mesh  Y = O @ W2 (O + scales resident) : %d/%d differ\n", yd, LAD_M * LAD_N2);
  if (yd) show_u16(Y_hw, (const uint16_t *) LAD_Y_BF16, LAD_M, LAD_N2, 8);
  if (yd && !od && !os)
    printf("  NOTE O's codes and its DRAM scales are both correct, so the requantizer works and "
           "the failure is in the RESIDENT path -- the codes left in the spad, or the E8M0 bytes "
           "written into the act-scale window.\n");

  return verdict(od + os + yd);
}

// =================================================================================================
#elif LADDER_KIND == KIND_HOST

// No gemmini instruction is issued here at all. Every stage below feeds the mesh in the real
// kernel, so a host disagreement explains "every mesh stage differs" with nothing wrong in the
// mesh -- and it is the one rung that can be believed even if the accelerator is entirely broken.
static float    xn_f[LAD_M * LAD_D], p_f[LAD_M * LAD_M], q_f[LAD_M * LAD_H];
static uint8_t  xn_codes[LAD_M * LAD_D], xn_scales[LAD_GD * LAD_M];
static uint8_t  p_codes[LAD_M * LAD_M], p_scales[LAD_GM * LAD_M];
static uint8_t  q_codes[LAD_M * LAD_H], q_scales[LAD_GH * LAD_M];

int main(void) {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  banner();
  printf("plan  host fp32 glue only, M=%d D=%d H=%d -- no gemmini instruction is issued\n",
         LAD_M, LAD_D, LAD_H);

  uint64_t t0 = read_cycles();
  mx_rmsnorm((const uint16_t *) LAD_H_PRE_BF16, LAD_W_LN_BF16, LAD_M, LAD_D, LAD_EPS, xn_f);
  mx_quantize_rows(xn_f, LAD_M, LAD_D, xn_codes, xn_scales);
  int dc = mx_count_diff_u8(xn_codes, (const uint8_t *) LAD_XN_CODES, LAD_M * LAD_D);
  int ds = mx_count_diff_u8(xn_scales, (const uint8_t *) LAD_XN_SCALES, LAD_GD * LAD_M);
  printf("host  rmsnorm + quantize : codes %d/%d, scales %d/%d differ\n",
         dc, LAD_M * LAD_D, ds, LAD_GD * LAD_M);

  // sqrtf is the only libm call besides expf; every power-of-two operation in mx_host.h is exact
  // bit manipulation precisely so a library rounding log2f(8) to 2.9999997 cannot move a block
  // scale by a whole exponent.
  mx_softmax_causal((const uint16_t *) LAD_S_BF16, LAD_M, 1.0f / sqrtf((float) LAD_H), p_f);
  mx_quantize_rows(p_f, LAD_M, LAD_M, p_codes, p_scales);
  int pc = mx_count_diff_u8(p_codes, (const uint8_t *) LAD_P_CODES, LAD_M * LAD_M);
  int ps = mx_count_diff_u8(p_scales, (const uint8_t *) LAD_P_SCALES, LAD_GM * LAD_M);
  printf("host  causal softmax (expf) + quantize : codes %d/%d, scales %d/%d differ\n",
         pc, LAD_M * LAD_M, ps, LAD_GM * LAD_M);

  mx_rope((const uint16_t *) LAD_Q_BF16, (const uint32_t *) LAD_ROPE_COS,
          (const uint32_t *) LAD_ROPE_SIN, LAD_M, LAD_H, q_f);
  mx_quantize_rows(q_f, LAD_M, LAD_H, q_codes, q_scales);
  int qc = mx_count_diff_u8(q_codes, (const uint8_t *) LAD_QR_CODES, LAD_M * LAD_H);
  int qs = mx_count_diff_u8(q_scales, (const uint8_t *) LAD_QR_SCALES, LAD_GH * LAD_M);
  printf("host  RoPE + quantize : codes %d/%d, scales %d/%d differ   (%d cycles total)\n",
         qc, LAD_M * LAD_H, qs, LAD_GH * LAD_M, (int) (read_cycles() - t0));

  return verdict(dc + ds + pc + ps + qc + qs);
}

#else
#error "unknown LADDER_KIND"
#endif
