// The FP4 MLP's two projection shapes at RTL-simulation size (gen/gen_mlp_proj_fp4.py): gate G = xn @ Wg[:, :NG]
// (K 2048) and down Y = h @ Wd[:, :ND] (K 5632), native FP4 DRAM loops on the real llama_mlp_e2e_fp4 activations.
// Each is timed alone (after a fence) against the quad mesh's ideal and checked against the golden's hash.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#ifndef MP_HEADER
#define MP_HEADER "mlp_proj_fp4.h"
#endif
#include MP_HEADER

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: needs MX_ROCKET or Spike\n"); return 0; }
#else

#define DIM 16
#undef BANK_ROWS
#define BANK_ROWS 4096
#include "mx_native.h"

static uint16_t Gs[MP_M * MP_NG] __attribute__((aligned(64)));
static uint16_t Ys[MP_M * MP_ND] __attribute__((aligned(64)));
#ifdef MP_CHAIN
static uint16_t Gp[MP_M * MP_ND] __attribute__((aligned(64)));   // G at down's row pitch
#endif

#ifdef MP_ARES   // A resident ("A anywhere"): A loaded once, every loop reads it in place; else the plain path
#define MP_MATMUL(...) (mxn_matmul_fp4_ares(__VA_ARGS__) && (printf("A resident does not fit: plain path\n"), mxn_matmul_fp4(__VA_ARGS__)))
#else
#define MP_MATMUL(...) mxn_matmul_fp4(__VA_ARGS__)
#endif

static uint64_t fnv(const void *p, size_t n) {
  const uint64_t *w = (const uint64_t *) p; uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n / 8; i++) { h ^= w[i]; h *= 1099511628211ULL; }
  return h;
}

int main() {
  printf("mlp_proj_fp4: M=%d, gate %dx%d, down %dx%d (FP4 x FP4, native DRAM loops%s)\n", MP_M, MP_D, MP_NG, MP_KD, MP_ND,
#ifdef MP_ARES
         ", A resident"
#else
         ""
#endif
         );
  gemmini_flush(0);
  int bad = 0;
#ifndef MP_CHAIN_DOWN_A
#define MP_CHAIN_DOWN_A BANK_ROWS
#endif
#ifndef MP_CHAIN_B
#define MP_CHAIN_B (2 * BANK_ROWS)
#endif
#ifdef MP_CHAIN
  // gate and down back to back, no fence. Down's A block t loads right after gate's last-chunk loop t (that chunk only
  // streams B, so the bandwidth is free); its region MP_CHAIN_DOWN_A may wrap onto gate's A blocks < t, which that loop
  // no longer holds. G is stored at down's row pitch, so down needs no config_st (that would wait for gate to drain).
  mxn_fp4_mm g, d;
  bad |= mxn_fp4_plan(&g, MP_AT(MP_OFF_XN4, uint8_t), MP_D, MP_AT(MP_OFF_WG4, uint8_t), MP_NG / 2, Gp, MP_ND,
                      MP_AT(MP_OFF_XN_S, uint8_t), MP_M, MP_AT(MP_OFF_WG_S, uint8_t), MP_NG, MP_M, MP_D, MP_NG,
                      0, MP_CHAIN_B);
  bad |= mxn_fp4_plan(&d, MP_AT(MP_OFF_H4, uint8_t), MP_KD, MP_AT(MP_OFF_WD4, uint8_t), MP_ND / 2, Ys, MP_ND,
                      MP_AT(MP_OFF_H_S, uint8_t), MP_M, MP_AT(MP_OFF_WD_S, uint8_t), MP_ND, MP_M, MP_KD, MP_ND,
                      MP_CHAIN_DOWN_A, MP_CHAIN_B);
  if (bad) printf("chain does not fit\n");
  gemmini_fence();
  uint64_t t0 = read_cycles();
  if (!bad) {
    const int gl = MP_NG / g.NC - 1;
    mxn_fp4_cfg_ex_st(&g);
    mxn_fp4_cfg_a(&g);
    mxn_fp4_cfg_b(&g);
    for (int c = 0; c <= gl; c++)
      for (int t = 0; t < g.kts; t++) {
        if (c == 0) mxn_fp4_ld_a(&g, t);
        mxn_fp4_loop(&g, c, t);
        if (c == gl && t < d.kts) {
          if (t == 0) mxn_fp4_cfg_a(&d);
          mxn_fp4_ld_a(&d, t);
        }
      }
    for (int t = g.kts; t < d.kts; t++) mxn_fp4_ld_a(&d, t);
    mxn_fp4_cfg_b(&d);
    for (int c = 0; c < MP_ND / d.NC; c++)
      for (int t = 0; t < d.kts; t++) mxn_fp4_loop(&d, c, t);
  }
  gemmini_fence();
  const uint64_t cg = read_cycles() - t0, cd = 0;
  for (int r = 0; r < MP_M; r++) memcpy(Gs + r * MP_NG, Gp + r * MP_ND, MP_NG * sizeof(uint16_t));
#else
  gemmini_fence();
  uint64_t t0 = read_cycles();
  bad |= MP_MATMUL(MP_AT(MP_OFF_XN4, uint8_t), MP_D, MP_AT(MP_OFF_WG4, uint8_t), MP_NG / 2, Gs, MP_NG,
                        MP_AT(MP_OFF_XN_S, uint8_t), MP_M, MP_AT(MP_OFF_WG_S, uint8_t), MP_NG, MP_M, MP_D, MP_NG);
  gemmini_fence();
  const uint64_t cg = read_cycles() - t0;
  t0 = read_cycles();
  bad |= MP_MATMUL(MP_AT(MP_OFF_H4, uint8_t), MP_KD, MP_AT(MP_OFF_WD4, uint8_t), MP_ND / 2, Ys, MP_ND,
                        MP_AT(MP_OFF_H_S, uint8_t), MP_M, MP_AT(MP_OFF_WD_S, uint8_t), MP_ND, MP_M, MP_KD, MP_ND);
  gemmini_fence();
  const uint64_t cd = read_cycles() - t0;
#endif
  const uint64_t ig = (uint64_t) MP_M * MP_D * MP_NG / (4 * DIM * DIM), id = (uint64_t) MP_M * MP_KD * MP_ND / (4 * DIM * DIM);
  const uint64_t hg = fnv(Gs, sizeof(Gs)), hy = fnv(Ys, sizeof(Ys));
  printf("hash G %016llx %s\nhash Y %016llx %s\n", (unsigned long long) hg, hg == MP_HASH_G ? "match" : "MISMATCH vs golden",
         (unsigned long long) hy, hy == MP_HASH_Y ? "match" : "MISMATCH vs golden");
#ifdef MP_CHAIN
  printf("PERF gate+down back to back %llu cycles, quad ideal %llu -> util %llu%%\n", (unsigned long long) cg,
         (unsigned long long) (ig + id), (unsigned long long) (100 * (ig + id) / cg));
  (void) cd;
#else
  printf("PERF gate %llu cycles, quad ideal %llu -> util %llu%%\n", (unsigned long long) cg, (unsigned long long) ig,
         (unsigned long long) (100 * ig / cg));
  printf("PERF down %llu cycles, quad ideal %llu -> util %llu%%\n", (unsigned long long) cd, (unsigned long long) id,
         (unsigned long long) (100 * id / cd));
#endif
  const int fail = bad || hg != MP_HASH_G || hy != MP_HASH_Y;
  printf("mlp_proj_fp4 %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
#endif
