// dual_mm.h -- one MX matmul on whichever Gemmini the params header names (XCUSTOM_ACC + DIM): the dual-Gemmini
// SoCs put a 16x16 on custom3 and an 8x8 on custom2, so a kernel built against -iquote dual/<inst> drives that one.
//
//   C[M][N] (BF16) = A[M][K] (E4M3 single, scales [K/32][M]) @ B (scales [K/32][N])
//   B is FP4 E2M1 nibble-packed [K][N/2] (A8W4, mode6) or E4M3 [K][N] (E4M3 x E4M3, mode8).
#ifndef INCLUDE_DUAL_MM_H
#define INCLUDE_DUAL_MM_H

#include <stdint.h>
#include "include/gemmini_testutils.h"

enum { MM_B_E4M3 = 0, MM_B_FP4 = 2 };

static uint32_t mm_sf_dummy[1024] __attribute__((aligned(64)));
static int mm_fail_fit = 0;

static void dual_mm(const uint8_t *A, const uint8_t *As, const uint8_t *B, const uint8_t *Bs, int M, int K, int N,
                    int bfmt, uint16_t *C) {
  const int cols_per_tile = (bfmt == MM_B_FP4) ? 2 * DIM : DIM;    // quad FP4: 2 cols/lane
  const int tI = M / DIM, tK = K / DIM, tJ = N / cols_per_tile;
  const uint32_t spad_rows = BANK_NUM * BANK_ROWS;
  const uint32_t a_base = 0, b_base = spad_rows - tK * tJ * DIM, dest = tI * tK * DIM;
  const int out_rows = M * N * 2 / DIM;                            // BF16, row-major in the spad
  if (dest + out_rows > b_base || M % DIM || K % 32 || N % cols_per_tile) { mm_fail_fit = 1; return; }

  gemmini_extended_config_st(DIM, NO_ACTIVATION, 1);
  gemmini_mxquant_config_mvout((uint64_t) mm_sf_dummy, tI, tJ, tK, 0, 0, 1);
  // config_ex: A fp8 code0 / altfmt0 / no LUT = E4M3 single; B fp4 (code2) or E4M3 single; out BF16 (3).
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
      ((uint64_t) acc_scale_t_to_acc_scale_t_bits((acc_scale_t) ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t) 1 << 16) | ((uint64_t) 3 << 14) | ((uint64_t) bfmt << 12)
    | ((uint64_t) WEIGHT_STATIONARY << 2) | CONFIG_EX,
      ((uint64_t) 1 << 48), k_CONFIG);
  // Scale loads AFTER config_ex: the scale-mem layout follows the live mode.
  gemmini_mx_load_scales((uint64_t) As, (K / 32) * M, 0);
  gemmini_mx_load_scales((uint64_t) Bs, (K / 32) * N, 1);
  gemmini_fence();

  gemmini_config_ld(K);
  for (int i = 0; i < tI; i++)
    for (int k = 0; k < tK; k++)
      gemmini_extended_mvin((void *) (A + i * DIM * K + k * DIM), a_base + (i * tK + k) * DIM, DIM, DIM);
  const int b_pitch = (bfmt == MM_B_FP4) ? N / 2 : N;
  gemmini_config_ld(b_pitch);
  for (int k = 0; k < tK; k++)
    for (int j = 0; j < tJ; j++)
      gemmini_extended_mvin((void *) (B + k * DIM * b_pitch + j * DIM), b_base + (k * tJ + j) * DIM, DIM, DIM);

  gemmini_loop_ws_spad(tI, tJ, tK, 0, 0, 0, a_base, spad_rows, 0, dest,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false,
                       0x38);   // operands already in spad: compute + acc->spad store only

  gemmini_fence();
  gemmini_config_st(DIM);
  for (int r = 0; r < out_rows; r += DIM)
    gemmini_extended_mvout((uint8_t *) C + r * DIM, dest + r, DIM, DIM);
  gemmini_fence();
}

static int dual_check_u16(const char *what, const uint16_t *got, const uint16_t *ref, int n) {
  int d = 0, first = -1;
  for (int i = 0; i < n; i++) if (got[i] != ref[i]) { if (first < 0) first = i; d++; }
  printf("mesh  %-22s: %d/%d differ", what, d, n);
  if (first >= 0) printf("  (first @%d got 0x%04x exp 0x%04x)", first, got[first], ref[first]);
  printf("\n");
  return d;
}

#endif // INCLUDE_DUAL_MM_H
