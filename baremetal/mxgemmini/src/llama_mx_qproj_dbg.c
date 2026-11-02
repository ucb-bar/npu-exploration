// Debug harness: llama_layer_mx's Q projection alone (E4M3 native loop, BF16 C to DRAM) into a guarded buffer:
// counts stores past C's end and hashes each C row, to compare RTL against Spike.
#define main lmx_full_main
#include "llama_layer_mx.c"
#undef main

#define GUARD_W 32768   // uint16 words after C (64 KB)
static uint16_t Cg[LM * LD + GUARD_W] __attribute__((aligned(64)));

int main() {
  lmx_rw = (uint8_t *) LMX_AT(0, uint8_t);
  __asm__ volatile("" : "+r"(lmx_rw));
  const uint16_t *H_PRE = LMX_AT(LMX_OFF_H_PRE, uint16_t);
  printf("llama_mx_qproj_dbg: DIM %d, Q = xn1 [%d][%d] @ Wq [%d][%d] -> BF16\n", DIM, LM, LD, LD, LD);
  gemmini_flush(0);
  gemmini_fence();
  rms_phase(H_PRE, 0, 0, LMX_AT(LMX_OFF_W_IN_LN, uint16_t), xn1_c, xn1_sr);
  gemmini_fence();
  scales_t(xn1_s, xn1_sr, LM, LD / 32);
  for (int i = 0; i < LM * LD + GUARD_W; i++) Cg[i] = 0xa5a5;
  fmt_cfg();
  int bad = MATMUL(xn1_c, WC(Q), Cg, LD, xn1_s, WS(Q), WLUT(Q), ALUT(XN1), LM, LD, LD);
  gemmini_fence();
  int over = 0, first = -1, last = -1;
  for (int i = 0; i < GUARD_W; i++)
    if (Cg[LM * LD + i] != 0xa5a5) { over++; if (first < 0) first = i; last = i; }
  int unwritten = 0;
  for (int i = 0; i < LM * LD; i++) unwritten += Cg[i] == 0xa5a5;
  printf("rejected %d; C hash %016llx (q_bf golden %016llx); unwritten C words %d; guard words written %d "
         "(first +%d, last +%d words past C)\n", bad, (unsigned long long) fnv(Cg, (size_t) LM * LD * 2),
         (unsigned long long) LMX_EXPECT[2], unwritten, over, first, last);
  for (int m = 0; m < LM; m++) printf("row %2d %016llx\n", m, (unsigned long long) fnv(Cg + (size_t) m * LD, LD * 2));
  // per row, a 4-hex hash of each 16-column group ("...." = never written)
  for (int m = 0; m <= LM; m++) {
    printf("map %2d", m);
    for (int g = 0; g < LD / 16; g++) {
      const uint16_t *p = Cg + (size_t) m * LD + g * 16;
      int un = 1;
      for (int i = 0; i < 16; i++) un &= p[i] == 0xa5a5;
      if (un) printf(" ....");
      else printf(" %04x", (unsigned) (fnv(p, 32) & 0xffff));
    }
    printf("\n");
  }
  return 0;
}
