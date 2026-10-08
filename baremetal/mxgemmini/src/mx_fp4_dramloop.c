// FP4 x FP4 through the native DRAM loop (mxn_matmul_fp4): N-chunks into a wider C (ldc > NC), several K-tiles,
// both scratchpad / scale halves. Checks C bit-exactly against the rtl_exact golden (gen/gen_mx_fp4_dramloop.py).
#include <stdint.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"
#include "mx_fp4_dramloop.h"

#define DIM 16
#undef BANK_ROWS
#define BANK_ROWS 4096
#include "mx_native.h"

static uint16_t C[FP4T_M][FP4T_N] __attribute__((aligned(64)));

int main() {
  gemmini_flush(0);
  const uint64_t t0 = read_cycles();
  if (mxn_matmul_fp4(&FP4T_A[0][0], FP4T_K, &FP4T_B[0][0], FP4T_N / 2, &C[0][0], FP4T_N,
                     &FP4T_ASC[0][0], FP4T_M, &FP4T_BSC[0][0], FP4T_N, FP4T_M, FP4T_K, FP4T_N))
    return 1;
  gemmini_fence();
  const uint64_t cyc = read_cycles() - t0;
  int bad = 0;
  for (int m = 0; m < FP4T_M; m++)
    for (int n = 0; n < FP4T_N; n++)
      if (C[m][n] != FP4T_C[m][n]) {
        if (bad < 8) printf("  C[%d][%d] = %04x, expected %04x\n", m, n, C[m][n], FP4T_C[m][n]);
        bad++;
      }
  const uint64_t ideal = (uint64_t) FP4T_M * FP4T_K * FP4T_N / (4 * DIM * DIM);
  printf("mx_fp4_dramloop M=%d K=%d N=%d: %llu cycles, quad-mesh ideal %llu (%llu%%), %d/%d wrong -> %s\n",
         FP4T_M, FP4T_K, FP4T_N, (unsigned long long) cyc, (unsigned long long) ideal,
         (unsigned long long) (100 * ideal / cyc), bad, FP4T_M * FP4T_N, bad ? "FAIL" : "PASS");
  return bad != 0;
}
