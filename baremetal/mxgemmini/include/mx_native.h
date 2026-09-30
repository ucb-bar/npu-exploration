// Native MX matmul on MxGemmini: one call = a sequence of native DRAM loops (gemmini_loop_ws_mx).
//
//   C[M][N] (BF16, row pitch ldc) = A[M][K] (fp8 E4M3, pitch lda) @ B[K][N] (fp8 E4M3, pitch ldb)
//   A_sc = [K/32][M] E8M0 rows (pitch sc_lda), B_sc = [K/32][N] E8M0 rows (pitch sc_ldb)
//
// Loop structure: N-chunks of NC columns (one accumulator half) x K-tiles of Kt (accumulate; C only on
// the last K-tile). Every loop keeps its A and B in ONE scratchpad half and consecutive loops alternate
// halves (mxn_loops is global, so this also holds across calls): the next loop's operands and its
// loop-managed scale slices stream in under the current loop's compute, the chunk's C store overlaps the
// next chunk (acc halves alternate), and the cross-loop WAR hold never triggers. Cost: A is re-read per
// loop (+M/NC of the weight traffic) and each loop pays its scale-config drain (~35 cycles).
//
// The caller defines DIM (16) and pins BANK_ROWS before including this.
#ifndef INCLUDE_MX_NATIVE_H
#define INCLUDE_MX_NATIVE_H

#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#ifndef DIM
#error "mx_native.h needs DIM defined by the caller"
#endif
#define MXN_HALF_ROWS ((BANK_NUM * BANK_ROWS) / 2)

// Tile choice: the largest N-chunk NC (<= 512) whose C fits one acc half (E4M3-single: M*NC/64 rows) --
// larger NC means less A re-read -- then the largest K-tile Kt (<= 512) whose A (M*Kt/16) + B (Kt*NC/16)
// fit one spad half and whose scale slices fit 4 KB. At M = 32 this is NC 512, Kt 128.
static int mxn_pick_nc(int M, int N) {
  for (int nc = 512; nc >= DIM; nc >>= 1)
    if (N % nc == 0 && M * nc / (DIM * 4) <= ACC_ROWS / 2) return nc;
  return 0;
}
static int mxn_pick_kt(int M, int K, int NC) {
  for (int kt = 512; kt >= 32; kt >>= 1)
    if (K % kt == 0 && M * kt / DIM + kt * NC / DIM <= MXN_HALF_ROWS && (kt / 32) * NC <= 4096 &&
        (kt / 32) * M <= 4096) return kt;
  return 0;
}

static unsigned mxn_loops;   // loops issued so far: the parity picks the scratchpad half

// blk_nc/blk_kt != 0: B is PRE-BLOCKED for that tiling -- block (c, t) = B[t*Kt:+Kt][c*NC:+NC] stored
// contiguously (row pitch NC), blocks c-major -- so each loop streams one contiguous run from DRAM instead
// of Kt rows ldb apart. B_sc stays [K/32][N] (pitch sc_ldb). The chosen tiling must equal (blk_nc, blk_kt).
static int mxn_matmul_ex(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                         const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb,
                         int M, int K, int N, int blk_nc, int blk_kt) {
  const int NC = mxn_pick_nc(M, N);
  const int Kt = NC ? mxn_pick_kt(M, K, NC) : 0;
  const int blocked = blk_nc != 0;
  if (blocked && (NC != blk_nc || Kt != blk_kt)) {
    printf("mxn_matmul: B blocked for NC=%d Kt=%d but tiling is NC=%d Kt=%d\n", blk_nc, blk_kt, NC, Kt);
    return 1;
  }
  if (!NC || !Kt || M % DIM ||
      (((uintptr_t) A_sc | (uintptr_t) B_sc | (uintptr_t) sc_lda | (uintptr_t) sc_ldb) & 7)) {
    printf("mxn_matmul: unsupported shape/alignment M=%d K=%d N=%d (Kt=%d NC=%d)\n", M, K, N, Kt, NC);
    return 1;
  }
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0,
                              /*BF16*/ 3, 0);
  gemmini_extended3_config_ld(lda * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 0);
  const int ldb_loop = blocked ? NC : ldb;
  gemmini_extended3_config_ld(ldb_loop * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(ldc * sizeof(uint16_t));
  const int kts = K / Kt;
  for (int c = 0; c < N / NC; c++)
    for (int t = 0; t < kts; t++) {
      const int h = 1 + (int) (mxn_loops++ & 1);
      gemmini_loop_ws_mx(M / DIM, NC / DIM, Kt / DIM,
                         A + (size_t) t * Kt,
                         blocked ? B + ((size_t) c * kts + t) * Kt * NC : B + (size_t) t * Kt * ldb + (size_t) c * NC,
                         t == kts - 1 ? C + (size_t) c * NC : NULL,
                         lda, ldb_loop, ldc,
                         A_sc + (size_t) (t * Kt / 32) * sc_lda,
                         B_sc + (size_t) (t * Kt / 32) * sc_ldb + (size_t) c * NC, sc_lda, sc_ldb,
                         /*accumulate*/ t > 0, h, h);
    }
  return 0;
}

static int mxn_matmul(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                      const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb, int M, int K, int N) {
  return mxn_matmul_ex(A, lda, B, ldb, C, ldc, A_sc, sc_lda, B_sc, sc_ldb, M, K, N, 0, 0);
}

// Ideal mesh cycles of one M x K x N matmul on the 16x16 mesh.
#define MXN_IDEAL(m, k, n) ((uint64_t) (m) * (k) * (n) / (DIM * DIM))

#endif  // INCLUDE_MX_NATIVE_H
