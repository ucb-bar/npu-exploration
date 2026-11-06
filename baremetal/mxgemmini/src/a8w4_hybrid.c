// A8W4 hybrid native loop test: operands from DRAM (loop DMA), loop-managed scales, BF16 C into the scratchpad.
// C = A @ B as N/NC chunks x K/KT K-tiles; operand buffers alternate between two spad regions; chunk c's C lands at
// SP_C + c * (M * NC * 2 / DIM) rows, row-major [M][NC]. Read back flat with mvout, checked against C_out_bf16.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "dual_loops.h"
#include A8W4_HDR
#ifndef A8W4_NC
#define A8W4_NC 128
#endif
#ifndef A8W4_KT
#define A8W4_KT 128
#endif
#define NC A8W4_NC
#define KT A8W4_KT
#define SP_ROWS (BANK_NUM * BANK_ROWS)
#define A_ROWS (MATMUL_M * KT / DIM)
#define B_ROWS (KT * NC / 2 / DIM)
#define BUF_ROWS (A_ROWS + B_ROWS)
#define C_ROWS (MATMUL_M * NC * 2 / DIM)
#define SP_C (2 * BUF_ROWS)
typedef char fits[(SP_C + (MATMUL_N / NC) * C_ROWS <= SP_ROWS) ? 1 : -1];

static uint16_t C[MATMUL_N / NC][MATMUL_M][NC] __attribute__((aligned(64)));

int main() {
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 2, 3, 0);
  gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, 0);
  gemmini_fence();
  gemmini_extended3_config_ld(MATMUL_K, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(MATMUL_N / 2, MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(NC * sizeof(uint16_t));

  const int nk = MATMUL_K / KT;
  int loops = 0;
  uint64_t t0 = read_cycles();
  for (int c = 0; c < MATMUL_N / NC; c++)
    for (int t = 0; t < nk; t++, loops++) {
      const uint32_t base = (loops & 1) * BUF_ROWS;
      gemmini_loop_ws_mx_spad(MATMUL_M / DIM, NC / (2 * DIM), KT / DIM,
                              (uint8_t *) A_in + t * KT, (uint8_t *) B_in + t * KT * (MATMUL_N / 2) + c * NC / 2,
                              MATMUL_K, MATMUL_N / 2, base, base + BUF_ROWS, SP_C + c * C_ROWS,
                              &A_scales_row[t * KT / 32][0], &B_scales_col[t * KT / 32][c * NC], MATMUL_M, MATMUL_N,
                              t > 0, t == nk - 1, 0);
    }
  gemmini_fence();
  uint64_t t1 = read_cycles();
  gemmini_config_st(DIM);
  for (int c = 0; c < MATMUL_N / NC; c++)
    for (int r = 0; r < C_ROWS; r += DIM)
      gemmini_extended_mvout((uint8_t *) &C[c][0][0] + r * DIM, SP_C + c * C_ROWS + r, DIM, DIM);
  gemmini_fence();

  int bad = 0;
  for (int c = 0; c < MATMUL_N / NC; c++)
    for (int i = 0; i < MATMUL_M; i++)
      for (int j = 0; j < NC; j++)
        if (C[c][i][j] != C_out_bf16[i][c * NC + j] && bad++ < 20)
          printf("C[%d][%d] got %04x exp %04x\n", i, c * NC + j, C[c][i][j], C_out_bf16[i][c * NC + j]);
  unsigned long long ideal = (unsigned long long) MATMUL_M * MATMUL_N * MATMUL_K / (2 * DIM * DIM);
  printf("a8w4 hybrid %dx%dx%d dim%d NC=%d KT=%d (%d loops): %d mismatches; %llu cycles (ideal %llu)\n", MATMUL_M,
         MATMUL_N, MATMUL_K, DIM, NC, KT, loops, bad, (unsigned long long) (t1 - t0), ideal);
  printf(bad ? "a8w4 hybrid test FAILED\n" : "a8w4 hybrid test PASSED\n");
  return bad != 0;
}
