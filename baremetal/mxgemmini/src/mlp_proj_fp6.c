// The TinyLlama MLP's gate and down projections in FP6 E3M2 LUT form at RTL-simulation size (gen/gen_mlp_proj_fp6.py),
// one build per LUT update granularity G (MP_HEADER = mlp_proj_fp6_g<G>.h): native DRAM loops through mxn_matmul_lut,
// act codebooks loaded flat per projection, weight codebooks loop-managed. Each projection is timed alone (fence to
// fence, including its act codebook load) against the quad ideal and checked against the bit-exact golden's hash.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include MP_HEADER

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: needs MX_ROCKET or Spike\n"); return 0; }
#else

#ifndef MP_DIM8   // DIM=8 builds take DIM / BANK_ROWS / ACC_ROWS from the DIM=8 params header (-iquote dim8/<fmt>)
#define DIM 16
#undef BANK_ROWS
#define BANK_ROWS 4096
#endif
#include "mx_native.h"

#define LUT_BYTES 12   // FP6 codebook: 16 x 6 bit

static uint16_t Gs[MP_M * MP_NG] __attribute__((aligned(64)));
static uint16_t Ys[MP_M * MP_ND] __attribute__((aligned(64)));
#ifdef MP_CHAIN
static uint16_t Gp[MP_M * MP_ND] __attribute__((aligned(64)));   // G at down's row pitch (no config_st between)
#endif

static uint64_t fnv(const void *p, size_t n) {
  const uint64_t *w = (const uint64_t *) p; uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n / 8; i++) { h ^= w[i]; h *= 1099511628211ULL; }
  return h;
}

static int nc_of(int M, int N) {   // mxn_matmul_lut's chunk choice, for the report
  for (int nc = 512; nc >= 2 * DIM; nc >>= 1)
    if (N % nc == 0 && M * nc / (DIM * 4) <= ACC_ROWS / 2 && (nc >> MP_G) <= 32) return nc;
  return 0;
}

// One projection: act codebooks (sel 1, flat) + the matmul, timed fence to fence.
static uint64_t proj(const uint8_t *A, const uint8_t *A_sc, const uint8_t *A_lut, const uint8_t *B, const uint8_t *B_sc,
                     const uint8_t *B_lut, uint16_t *C, int K, int N, int *bad) {
  gemmini_fence();
  const uint64_t t0 = read_cycles();
  gemmini_mx_load_lut_dt((uint64_t) A_lut, MP_M >> MP_G, 1, 6);
  *bad |= mxn_matmul_lut(A, K, B, N / 2, C, N, A_sc, MP_M, B_sc, N, B_lut, LUT_BYTES, MP_G, MP_M, K, N);
  gemmini_fence();
  return read_cycles() - t0;
}

int main() {
  printf("mlp_proj_fp6: G=%d, M=%d, gate %dx%d (NC %d), down %dx%d (NC %d), FP6 E3M2 LUT x LUT, native DRAM loops\n",
         MP_G, MP_M, MP_D, MP_NG, nc_of(MP_M, MP_NG), MP_KD, MP_ND, nc_of(MP_M, MP_ND));
  gemmini_flush(0);
  // fp6 (code 1) E3M2 (altfmt 0) via LUT on A and B, BF16 out
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16) | (3 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | ((WEIGHT_STATIONARY) << 2) | CONFIG_EX,
    ((uint64_t)(1) << 48), k_CONFIG);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 3, 1);
  gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, MP_G);   // the matmuls' G (every loop's in-stream config carries it)
  int bad = 0;
#ifdef MP_CHAIN
  // gate and down back to back, no fence: both projections' act codebooks loop-managed (M >> G <= 32), one store
  // stride (G at down's pitch). Reports the pair against the summed ideal.
  gemmini_fence();
  uint64_t t0 = read_cycles();
  gemmini_config_st(2 * MP_ND * sizeof(uint16_t));
  bad |= mxn_matmul_lut_ex(MP_AT(MP_OFF_XN4, uint8_t), MP_D, MP_AT(MP_OFF_WG4, uint8_t), MP_NG / 2, Gp, MP_ND,
                           MP_AT(MP_OFF_XN_S, uint8_t), MP_M, MP_AT(MP_OFF_WG_S, uint8_t), MP_NG,
                           MP_AT(MP_OFF_WG_L, uint8_t), LUT_BYTES, MP_G, MP_M, MP_D, MP_NG, MP_AT(MP_OFF_XN_L, uint8_t), 0);
  bad |= mxn_matmul_lut_ex(MP_AT(MP_OFF_H4, uint8_t), MP_KD, MP_AT(MP_OFF_WD4, uint8_t), MP_ND / 2, Ys, MP_ND,
                           MP_AT(MP_OFF_H_S, uint8_t), MP_M, MP_AT(MP_OFF_WD_S, uint8_t), MP_ND,
                           MP_AT(MP_OFF_WD_L, uint8_t), LUT_BYTES, MP_G, MP_M, MP_KD, MP_ND, MP_AT(MP_OFF_H_L, uint8_t), 0);
  gemmini_fence();
  const uint64_t cg = read_cycles() - t0, cd = 0;
  for (int r = 0; r < MP_M; r++) memcpy(Gs + r * MP_NG, Gp + r * MP_ND, MP_NG * sizeof(uint16_t));
#else
  const uint64_t cg = proj(MP_AT(MP_OFF_XN4, uint8_t), MP_AT(MP_OFF_XN_S, uint8_t), MP_AT(MP_OFF_XN_L, uint8_t),
                           MP_AT(MP_OFF_WG4, uint8_t), MP_AT(MP_OFF_WG_S, uint8_t), MP_AT(MP_OFF_WG_L, uint8_t),
                           Gs, MP_D, MP_NG, &bad);
  const uint64_t cd = proj(MP_AT(MP_OFF_H4, uint8_t), MP_AT(MP_OFF_H_S, uint8_t), MP_AT(MP_OFF_H_L, uint8_t),
                           MP_AT(MP_OFF_WD4, uint8_t), MP_AT(MP_OFF_WD_S, uint8_t), MP_AT(MP_OFF_WD_L, uint8_t),
                           Ys, MP_KD, MP_ND, &bad);
#endif
  const uint64_t ig = (uint64_t) MP_M * MP_D * MP_NG / (4 * DIM * DIM), id = (uint64_t) MP_M * MP_KD * MP_ND / (4 * DIM * DIM);
  const uint64_t hg = fnv(Gs, sizeof(Gs)), hy = fnv(Ys, sizeof(Ys));
  printf("hash G %016llx %s\nhash Y %016llx %s\n", (unsigned long long) hg, hg == MP_HASH_G ? "match" : "MISMATCH vs golden",
         (unsigned long long) hy, hy == MP_HASH_Y ? "match" : "MISMATCH vs golden");
#ifdef MP_CHAIN
  printf("PERF fp6_lut_g%d gate+down back to back %llu cycles, quad ideal %llu -> util %llu%%\n", MP_G,
         (unsigned long long) cg, (unsigned long long) (ig + id), (unsigned long long) (100 * (ig + id) / cg));
  (void) cd;
#else
  printf("PERF fp6_lut_g%d gate %llu cycles, quad ideal %llu -> util %llu%%\n", MP_G, (unsigned long long) cg,
         (unsigned long long) ig, (unsigned long long) (100 * ig / cg));
  printf("PERF fp6_lut_g%d down %llu cycles, quad ideal %llu -> util %llu%%\n", MP_G, (unsigned long long) cd,
         (unsigned long long) id, (unsigned long long) (100 * id / cd));
#endif
  const int fail = bad || hg != MP_HASH_G || hy != MP_HASH_Y;
  printf("mlp_proj_fp6 %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
#endif
