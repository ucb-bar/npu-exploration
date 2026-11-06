// dual_attn -- one causal attention head on one Gemmini of the dual-Gemmini SoC (see include/dual_mm.h).
//   Q,K,V = X@W{q,k,v} (A8W4) -> S = Q@K^T (E4M3 x E4M3) -> P = softmax_causal(S/sqrt(H)) (host)
//   -> O = P@V (E4M3 x E4M3) -> Out = O@Wo (A8W4)
// T=32 tokens, model dim 64, head dim 32. Every mesh stage is checked bit-exact against gen/gen_dual_a8w4.py.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "mx_host.h"
#include "dual_mm.h"
#include DUAL_ATTN_H   // data/dual_*_d<DIM>.h, set by the Makefile

#define A64 __attribute__((aligned(64)))
static uint16_t Q[ATT_T * ATT_H] A64, K[ATT_T * ATT_H] A64, V[ATT_T * ATT_H] A64;
static uint16_t S[ATT_T * ATT_T] A64, O[ATT_T * ATT_H] A64, Out[ATT_T * ATT_DM] A64;
static float f0[ATT_T * ATT_T], f1[ATT_T * ATT_T];
static uint8_t qc[ATT_T * ATT_H] A64, qs[ATT_T] A64, ktc[ATT_H * ATT_T] A64, kts[ATT_T] A64;
static uint8_t pc[ATT_T * ATT_T] A64, ps[ATT_T] A64, vc[ATT_T * ATT_H] A64, vs[ATT_H] A64;
static uint8_t oc[ATT_T * ATT_H] A64, os_[ATT_T] A64;

static void bf16_to_f32_buf(const uint16_t *src, int n, float *dst) {
  for (int i = 0; i < n; i++) dst[i] = mx_bf16_to_f32(src[i]);
}

int main() {
  printf("dual_attn: DIM=%d XCUSTOM_ACC=%d  T=%d Dm=%d H=%d\n", DIM, XCUSTOM_ACC, ATT_T, ATT_DM, ATT_H);
  gemmini_flush(0);
  uint64_t c_mesh = 0, c_host = 0, t;
#define MESH(stmt) do { t = read_cycles(); stmt; c_mesh += read_cycles() - t; } while (0)
#define HOST(stmt) do { t = read_cycles(); stmt; c_host += read_cycles() - t; } while (0)

  MESH(dual_mm(&att_x[0][0], &att_x_s[0][0], &att_wq[0][0], &att_wq_s[0][0], ATT_T, ATT_DM, ATT_H, MM_B_FP4, Q));
  MESH(dual_mm(&att_x[0][0], &att_x_s[0][0], &att_wk[0][0], &att_wk_s[0][0], ATT_T, ATT_DM, ATT_H, MM_B_FP4, K));
  MESH(dual_mm(&att_x[0][0], &att_x_s[0][0], &att_wv[0][0], &att_wv_s[0][0], ATT_T, ATT_DM, ATT_H, MM_B_FP4, V));
  HOST({
    bf16_to_f32_buf(Q, ATT_T * ATT_H, f0);
    mx_quantize_rows(f0, ATT_T, ATT_H, qc, qs);
    bf16_to_f32_buf(K, ATT_T * ATT_H, f0);
    mx_transpose_f32(f0, ATT_T, ATT_H, f1);                 // B = K^T [H][T]
    mx_quantize_cols(f1, ATT_H, ATT_T, ktc, kts);
  });
  MESH(dual_mm(qc, qs, ktc, kts, ATT_T, ATT_H, ATT_T, MM_B_E4M3, S));
  HOST({
    mx_softmax_causal(S, ATT_T, ATT_SCALE, f0);
    mx_quantize_rows(f0, ATT_T, ATT_T, pc, ps);
    bf16_to_f32_buf(V, ATT_T * ATT_H, f1);                  // B = V [T][H]
    mx_quantize_cols(f1, ATT_T, ATT_H, vc, vs);
  });
  MESH(dual_mm(pc, ps, vc, vs, ATT_T, ATT_T, ATT_H, MM_B_E4M3, O));
  HOST({
    bf16_to_f32_buf(O, ATT_T * ATT_H, f0);
    mx_quantize_rows(f0, ATT_T, ATT_H, oc, os_);
  });
  MESH(dual_mm(oc, os_, &att_wo[0][0], &att_wo_s[0][0], ATT_T, ATT_H, ATT_DM, MM_B_FP4, Out));
  if (mm_fail_fit) { printf("dual_attn FAILED: operands do not fit the scratchpad\n"); return 1; }

  int bad = 0;
  bad += dual_check_u16("Q = X@Wq  (A8W4)", Q, &att_q_ref[0][0], ATT_T * ATT_H);
  bad += dual_check_u16("K = X@Wk  (A8W4)", K, &att_k_ref[0][0], ATT_T * ATT_H);
  bad += dual_check_u16("V = X@Wv  (A8W4)", V, &att_v_ref[0][0], ATT_T * ATT_H);
  bad += dual_check_u16("S = Q@K^T (E4M3xE4M3)", S, &att_s_ref[0][0], ATT_T * ATT_T);
  int hd = mx_count_diff_u8(pc, &att_p_ref[0][0], ATT_T * ATT_T);
  printf("host  softmax + quant      : %d P code(s) differ from golden\n", hd);
  bad += dual_check_u16("O = P@V   (E4M3xE4M3)", O, &att_o_ref[0][0], ATT_T * ATT_H);
  bad += dual_check_u16("Out = O@Wo (A8W4)", Out, &att_out_ref[0][0], ATT_T * ATT_DM);
  printf("cycles: mesh %llu  host %llu\n", (unsigned long long) c_mesh, (unsigned long long) c_host);

  if (bad == 0 && hd == 0) printf("dual_attn PASSED (every mesh stage bit-exact, host stage matches).\n");
  else if (hd) printf("dual_attn %s: host glue drifted by %d byte(s); %d mesh element(s) differ.\n",
                      (bad == 0) ? "PASSED WITH DRIFT" : "FAILED", hd, bad);
  else printf("dual_attn FAILED: %d mesh element(s) differ.\n", bad);
  return (bad == 0) ? 0 : 1;
}
