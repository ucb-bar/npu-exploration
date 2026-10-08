// The FP4 MLP's two projection shapes at RTL-simulation size (gen/gen_mlp_proj_fp4.py): gate G = xn @ Wg[:, :NG]
// (K 2048) and down Y = h @ Wd[:, :ND] (K 5632), native FP4 DRAM loops on the real llama_mlp_e2e_fp4 activations.
// Each is timed alone (after a fence) against the quad mesh's ideal and checked against the golden's hash.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "mlp_proj_fp4.h"

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: needs MX_ROCKET or Spike\n"); return 0; }
#else

#define DIM 16
#undef BANK_ROWS
#define BANK_ROWS 4096
#include "mx_native.h"

static uint16_t Gs[MP_M * MP_NG] __attribute__((aligned(64)));
static uint16_t Ys[MP_M * MP_ND] __attribute__((aligned(64)));

static uint64_t fnv(const void *p, size_t n) {
  const uint64_t *w = (const uint64_t *) p; uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n / 8; i++) { h ^= w[i]; h *= 1099511628211ULL; }
  return h;
}

int main() {
  printf("mlp_proj_fp4: M=%d, gate %dx%d, down %dx%d (FP4 x FP4, native DRAM loops)\n", MP_M, MP_D, MP_NG, MP_F, MP_ND);
  gemmini_flush(0);
  int bad = 0;
  gemmini_fence();
  uint64_t t0 = read_cycles();
  bad |= mxn_matmul_fp4(MP_AT(MP_OFF_XN4, uint8_t), MP_D, MP_AT(MP_OFF_WG4, uint8_t), MP_NG / 2, Gs, MP_NG,
                        MP_AT(MP_OFF_XN_S, uint8_t), MP_M, MP_AT(MP_OFF_WG_S, uint8_t), MP_NG, MP_M, MP_D, MP_NG);
  gemmini_fence();
  const uint64_t cg = read_cycles() - t0;
  t0 = read_cycles();
  bad |= mxn_matmul_fp4(MP_AT(MP_OFF_H4, uint8_t), MP_F, MP_AT(MP_OFF_WD4, uint8_t), MP_ND / 2, Ys, MP_ND,
                        MP_AT(MP_OFF_H_S, uint8_t), MP_M, MP_AT(MP_OFF_WD_S, uint8_t), MP_ND, MP_M, MP_F, MP_ND);
  gemmini_fence();
  const uint64_t cd = read_cycles() - t0;
  const uint64_t ig = (uint64_t) MP_M * MP_D * MP_NG / (4 * DIM * DIM), id = (uint64_t) MP_M * MP_F * MP_ND / (4 * DIM * DIM);
  const uint64_t hg = fnv(Gs, sizeof(Gs)), hy = fnv(Ys, sizeof(Ys));
  printf("hash G %016llx %s\nhash Y %016llx %s\n", (unsigned long long) hg, hg == MP_HASH_G ? "match" : "MISMATCH vs golden",
         (unsigned long long) hy, hy == MP_HASH_Y ? "match" : "MISMATCH vs golden");
  printf("PERF gate %llu cycles, quad ideal %llu -> util %llu%%\n", (unsigned long long) cg, (unsigned long long) ig,
         (unsigned long long) (100 * ig / cg));
  printf("PERF down %llu cycles, quad ideal %llu -> util %llu%%\n", (unsigned long long) cd, (unsigned long long) id,
         (unsigned long long) (100 * id / cd));
  const int fail = bad || hg != MP_HASH_G || hy != MP_HASH_Y;
  printf("mlp_proj_fp4 %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
#endif
