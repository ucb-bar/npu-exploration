// dual_mlp -- a small SwiGLU MLP in A8W4 on one Gemmini of the dual-Gemmini SoC (see include/dual_mm.h).
//   G = X@Wg, U = X@Wu (A8W4) -> H = silu(G)*U, E4M3-quantized (host fp32) -> Y = H@Wd (A8W4)
// M=32 D=64 F=128. Every mesh stage is checked bit-exact against gen/gen_dual_a8w4.py's golden.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "mx_host.h"
#include "dual_mm.h"
#include DUAL_MLP_H   // data/dual_*_d<DIM>.h, set by the Makefile

static uint16_t G[MLP_M * MLP_F], U[MLP_M * MLP_F], Y[MLP_M * MLP_D] __attribute__((aligned(64)));
static float Hf[MLP_M * MLP_F];
static uint8_t Hc[MLP_M * MLP_F] __attribute__((aligned(64)));
static uint8_t Hs[(MLP_F / 32) * MLP_M] __attribute__((aligned(64)));

int main() {
  printf("dual_mlp: DIM=%d XCUSTOM_ACC=%d  M=%d D=%d F=%d\n", DIM, XCUSTOM_ACC, MLP_M, MLP_D, MLP_F);
  gemmini_flush(0);
  uint64_t t0 = read_cycles();
  dual_mm(&mlp_x[0][0], &mlp_x_s[0][0], &mlp_wg[0][0], &mlp_wg_s[0][0], MLP_M, MLP_D, MLP_F, MM_B_FP4, G);
  dual_mm(&mlp_x[0][0], &mlp_x_s[0][0], &mlp_wu[0][0], &mlp_wu_s[0][0], MLP_M, MLP_D, MLP_F, MM_B_FP4, U);
  uint64_t t1 = read_cycles();
  mx_swiglu(G, U, MLP_M * MLP_F, Hf);
  mx_quantize_rows(Hf, MLP_M, MLP_F, Hc, Hs);
  uint64_t t2 = read_cycles();
  dual_mm(Hc, Hs, &mlp_wd[0][0], &mlp_wd_s[0][0], MLP_M, MLP_F, MLP_D, MM_B_FP4, Y);
  uint64_t t3 = read_cycles();
  if (mm_fail_fit) { printf("dual_mlp FAILED: operands do not fit the scratchpad\n"); return 1; }

  int bad = 0;
  bad += dual_check_u16("G = X@Wg  (A8W4)", G, &mlp_g_ref[0][0], MLP_M * MLP_F);
  bad += dual_check_u16("U = X@Wu  (A8W4)", U, &mlp_u_ref[0][0], MLP_M * MLP_F);
  int hd = mx_count_diff_u8(Hc, &mlp_h_ref[0][0], MLP_M * MLP_F) + mx_count_diff_u8(Hs, &mlp_h_s_ref[0][0], sizeof(Hs));
  printf("host  silu(G)*U + quant    : %d byte(s) differ from golden\n", hd);
  bad += dual_check_u16("Y = H@Wd  (A8W4)", Y, &mlp_y_ref[0][0], MLP_M * MLP_D);
  printf("cycles: gate+up %llu  host %llu  down %llu  total %llu\n", (unsigned long long) (t1 - t0),
         (unsigned long long) (t2 - t1), (unsigned long long) (t3 - t2), (unsigned long long) (t3 - t0));

  if (bad == 0 && hd == 0) printf("dual_mlp PASSED (every mesh stage bit-exact, host stage matches).\n");
  else if (hd) printf("dual_mlp %s: host glue drifted by %d byte(s); %d mesh element(s) differ.\n",
                      (bad == 0) ? "PASSED WITH DRIFT" : "FAILED", hd, bad);
  else printf("dual_mlp FAILED: %d mesh element(s) differ.\n", bad);
  return (bad == 0) ? 0 : 1;
}
